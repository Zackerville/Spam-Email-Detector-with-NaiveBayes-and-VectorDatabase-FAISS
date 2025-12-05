import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from .config import *
from .preprocess import *
from .data_utils import *

def load_encoder(model_name=EMBEDDING_MODEL_NAME, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device

def build_vector_db_knn(texts, labels, model_name=EMBEDDING_MODEL_NAME, test_size=0.1, random_state=42, batch_size=20):
    tokenizer, model, device = load_encoder(model_name=model_name)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    embeddings = get_embeddings(texts, model, tokenizer, device, batch_size=batch_size)
    indices = np.arange(len(texts))
    
    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=y, random_state=random_state)
    x_train_emb = embeddings[train_idx]
    x_test_emb = embeddings[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    train_metadata = []
    test_metadata = []
    
    for i in train_idx:
        train_metadata.append({
            "index": int(i),
            "text": texts[i],
            "label": labels[i],
            "label_id": int(y[i])
        })
    for i in test_idx:
        test_metadata.append({
            "index": int(i),
            "text": texts[i],
            "label": labels[i],
            "label_id": int(y[i])
        })
        
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
        "test_metadata": test_metadata
    }
    return state

def _majority_vote(labels):
    if not labels:
        return None, {}
    labels = list(labels)
    unique, counts = np.unique(labels, return_counts=True)
    best_idx = int(np.argmax(counts))
    prediction = unique[best_idx]
    distribution = {str(label): int(count) for label, count in zip(unique, counts)}
    return prediction, distribution

def predict_knn(text, state, k=3):
    tokenizer = state["tokenizer"]
    model = state["model"]
    device = state["device"]
    index = state["index"]
    train_metadata = state["train_metadata"]
    query_with_prefix = f"query: {text}"
    batch_dict = tokenizer(
        [query_with_prefix],
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    with torch.no_grad():
        outputs = model(**batch_dict)
        query_embedding = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
    query_embedding = query_embedding.cpu().numpy().astype("float32")
    scores, indices = index.search(query_embedding, k)
    neighbor_labels = []
    neighbors = []
    for j in range(k):
        neighbor_idx = int(indices[0][j])
        neighbor_score = float(scores[0][j])
        meta = train_metadata[neighbor_idx]
        neighbor_labels.append(meta["label"])
        neighbors.append({
            "index": meta["index"],
            "label": meta["label"],
            "text": meta["text"],
            "score": neighbor_score
        })
    prediction, label_distribution = _majority_vote(neighbor_labels)
    result = {
        "prediction": prediction,
        "neighbors": neighbors,
        "label_distribution": label_distribution
    }
    return result

def evaluate_knn(state, k_values=(1, 3, 5)):
    index = state["index"]
    x_test_emb = state["x_test_emb"]
    test_metadata = state["test_metadata"]
    train_metadata = state["train_metadata"]
    results = {}
    all_errors = {}
    total = len(x_test_emb)
    for k in k_values:
        correct = 0
        errors = []
        for i in range(total):
            query_embedding = x_test_emb[i:i + 1].astype("float32")
            true_label = test_metadata[i]["label"]
            true_text = test_metadata[i]["text"]
            scores, indices = index.search(query_embedding, k)
            neighbor_labels = []
            neighbors = []
            for j in range(k):
                neighbor_idx = int(indices[0][j])
                neighbor_score = float(scores[0][j])
                meta = train_metadata[neighbor_idx]
                neighbor_labels.append(meta["label"])
                neighbors.append({
                    "index": meta["index"],
                    "label": meta["label"],
                    "text": meta["text"],
                    "score": neighbor_score
                })
            prediction, label_distribution = _majority_vote(neighbor_labels)
            if prediction == true_label:
                correct += 1
            else:
                error_info = {
                    "true_label": true_label,
                    "true_text": true_text,
                    "prediction": prediction,
                    "neighbors": neighbors,
                    "label_distribution": label_distribution
                }
                errors.append(error_info)
        accuracy = correct / total if total > 0 else 0.0
        results[k] = accuracy
        all_errors[k] = errors
    return results, all_errors

def train_and_evaluate_vector_db_knn(model_name=EMBEDDING_MODEL_NAME, test_size=0.1, random_state=42, batch_size=20, k_values=(1, 3, 5)):
    texts, labels = load_messages_labels()
    state = build_vector_db_knn(
        texts,
        labels,
        model_name=model_name,
        test_size=test_size,
        random_state=random_state,
        batch_size=batch_size
    )
    results, errors = evaluate_knn(state, k_values=k_values)
    return state, results, errors