import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import DATASET_PATH, EMBEDDING_MODEL_NAME
from .preprocess import get_embeddings, average_pool
from .data_utils import load_messages_labels


def load_encoder(model_name=EMBEDDING_MODEL_NAME, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def build_vector_db_knn(
    texts,
    labels,
    model_name=EMBEDDING_MODEL_NAME,
    test_size=0.1,
    random_state=42,
    batch_size=20,
):
    tokenizer, model, device = load_encoder(model_name=model_name)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    embeddings = get_embeddings(texts, model, tokenizer, device, batch_size=batch_size)
    indices = np.arange(len(texts))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    x_train_emb = embeddings[train_idx]
    x_test_emb = embeddings[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    train_metadata = []
    test_metadata = []

    for i in train_idx:
        train_metadata.append(
            {
                "index": int(i),
                "text": texts[i],
                "label": labels[i],
                "label_id": int(y[i]),
            }
        )

    for i in test_idx:
        test_metadata.append(
            {
                "index": int(i),
                "text": texts[i],
                "label": labels[i],
                "label_id": int(y[i]),
            }
        )

    dim = x_train_emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(x_train_emb.astype("float32"))

    state = {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "label_encoder": encoder,
        "index": index,
        "x_train_emb": x_train_emb.astype("float32"),
        "x_test_emb": x_test_emb.astype("float32"),
        "y_train": y_train,
        "y_test": y_test,
        "train_metadata": train_metadata,
        "test_metadata": test_metadata,
    }
    return state


def _majority_vote(labels):
    if not labels:
        return None, {}
    labels = list(labels)
    unique, counts = np.unique(labels, return_counts=True)
    best_idx = int(np.argmax(counts))
    prediction = int(unique[best_idx])
    distribution = {int(label): int(count) for label, count in zip(unique, counts)}
    return prediction, distribution


def _classify_embedding(query_embedding, state, k, spam_threshold=None):
    index = state["index"]
    train_metadata = state["train_metadata"]
    label_encoder = state["label_encoder"]

    scores, indices = index.search(query_embedding, k)

    neighbor_ids = []
    neighbors = []
    for j in range(k):
        neighbor_idx = int(indices[0][j])
        neighbor_score = float(scores[0][j])
        meta = train_metadata[neighbor_idx]
        neighbor_ids.append(int(meta["label_id"]))
        neighbors.append(
            {
                "index": meta["index"],
                "label": meta["label"],
                "text": meta["text"],
                "score": neighbor_score,
            }
        )

    pred_id, dist_ids = _majority_vote(neighbor_ids)

    classes = list(label_encoder.classes_)
    spam_ratio = 0.0
    spam_id = None
    total_neighbors = sum(dist_ids.values())

    if "spam" in classes and total_neighbors > 0:
        spam_id = int(label_encoder.transform(["spam"])[0])
        spam_count = int(dist_ids.get(spam_id, 0))
        spam_ratio = spam_count / float(total_neighbors)

    if spam_threshold is not None and spam_id is not None:
        if spam_ratio >= spam_threshold:
            pred_id = spam_id
        else:
            ham_candidates = [lid for lid in dist_ids.keys() if lid != spam_id]
            if ham_candidates:
                best_ham = max(ham_candidates, key=lambda lid: dist_ids[lid])
                pred_id = best_ham

    if pred_id is None:
        pred_label = None
    else:
        pred_label = label_encoder.inverse_transform([int(pred_id)])[0]

    label_distribution = {}
    for lid, count in dist_ids.items():
        lab = label_encoder.inverse_transform([int(lid)])[0]
        label_distribution[lab] = int(count)

    result = {
        "prediction_id": int(pred_id) if pred_id is not None else None,
        "prediction": pred_label,
        "neighbors": neighbors,
        "label_distribution": label_distribution,
        "spam_ratio": float(spam_ratio),
    }
    return result


def predict_knn(text, state, k=3, spam_threshold=None):
    tokenizer = state["tokenizer"]
    model = state["model"]
    device = state["device"]

    if spam_threshold is None:
        spam_threshold = state.get("spam_threshold", None)

    query_with_prefix = f"query: {text}"

    batch_dict = tokenizer(
        [query_with_prefix],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = model(**batch_dict)
        query_embedding = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        query_embedding = F.normalize(query_embedding, p=2, dim=1)

    query_embedding = query_embedding.cpu().numpy().astype("float32")

    return _classify_embedding(
        query_embedding,
        state,
        k,
        spam_threshold=spam_threshold,
    )


def evaluate_knn(state, k_values=(1, 3, 5), spam_threshold=None):
    x_test_emb = state["x_test_emb"]
    y_test = state["y_test"]
    test_metadata = state["test_metadata"]
    label_encoder = state["label_encoder"]

    results = {}
    all_errors = {}
    total = len(x_test_emb)

    for k in k_values:
        correct = 0
        errors = []
        for i in range(total):
            query_embedding = x_test_emb[i : i + 1].astype("float32")
            out = _classify_embedding(
                query_embedding,
                state,
                k,
                spam_threshold=spam_threshold,
            )
            pred_id = out["prediction_id"]
            true_id = int(y_test[i])
            if pred_id == true_id:
                correct += 1
            else:
                true_label = label_encoder.inverse_transform([true_id])[0]
                true_text = test_metadata[i]["text"]
                error_info = {
                    "true_label": true_label,
                    "true_text": true_text,
                    "prediction": out["prediction"],
                    "neighbors": out["neighbors"],
                    "label_distribution": out["label_distribution"],
                    "spam_ratio": out["spam_ratio"],
                }
                errors.append(error_info)

        accuracy = correct / total if total > 0 else 0.0
        results[k] = accuracy
        all_errors[k] = errors

    return results, all_errors


def train_and_evaluate_vector_db_knn(
    model_name=EMBEDDING_MODEL_NAME,
    test_size=0.1,
    random_state=42,
    batch_size=20,
    k_values=(1, 3, 5),
    spam_threshold=None,
):
    texts, labels = load_messages_labels()

    state = build_vector_db_knn(
        texts,
        labels,
        model_name=model_name,
        test_size=test_size,
        random_state=random_state,
        batch_size=batch_size,
    )

    results, errors = evaluate_knn(
        state,
        k_values=k_values,
        spam_threshold=spam_threshold,
    )

    state["spam_threshold"] = spam_threshold

    return state, results, errors
