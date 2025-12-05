import gradio as gr
from src.nb_pipeline import *
from src.vector_db_knn_pipeline import train_and_evaluate_vector_db_knn, predict_knn
from src.config import *

model_nb, vocab_nb, idf_nb, metrics_nb = train_nbclassifier()
acc_nb = metrics_nb["accuracy"]
f1_nb = metrics_nb["f1"]
roc_nb = metrics_nb["ROC"]

state_faiss, faiss_results, faiss_errors = train_and_evaluate_vector_db_knn()
if faiss_results:
    best_k = max(faiss_results, key=lambda k: faiss_results[k])
    best_acc = faiss_results[best_k]
else:
    best_k = None
    best_acc = None

def decode_faiss_output(prediction, label_distribution):
    if isinstance(prediction, (int, float)):
        pred_label = "spam" if int(prediction) == 1 else "ham"
    else:
        s = str(prediction)
        if s in ["1", "spam", "Spam"]:
            pred_label = "spam"
        elif s in ["0", "ham", "Ham"]:
            pred_label = "ham"
        else:
            pred_label = s
    spam_count = 0
    total = 0
    if isinstance(label_distribution, dict):
        for k, v in label_distribution.items():
            total += v
            if str(k) in ["1", "spam", "Spam"]:
                spam_count += v
    score = spam_count / total if total > 0 else 0.0
    dist_str = ", ".join(f"{k}: {v}" for k, v in label_distribution.items()) if label_distribution else ""
    return pred_label, float(score), dist_str

def classify_email(text, model_choice):
    if not text or not text.strip():
        return "Empty Content", 0.0, ""
    if model_choice == "Naive Bayes (TF-IDF)":
        label, prob = predict_nb(text, model=model_nb, vocab=vocab_nb, idf=idf_nb)
        return label, float(prob), ""
    
    out = predict_knn(text, state_faiss, k=3)
    prediction = out.get("prediction")
    label_distribution = out.get("label_distribution", {})
    pred_label, score, dist_str = decode_faiss_output(prediction, label_distribution)
    return pred_label, score, dist_str

description = f"NaiveBayes ACC: {acc_nb:.4f} | F1: {f1_nb:.4f} | ROC-AUC: {roc_nb:.4f}"
if best_k is not None:
    description += f" | FAISS k-NN best k={best_k}, ACC={best_acc:.4f}"

iface = gr.Interface(
    fn=classify_email,
    inputs=[
        gr.Textbox(lines=6, label="Type email content here"),
        gr.Radio(
            choices=["Naive Bayes (TF-IDF)", "FAISS k-NN (Embedding)"],
            value="Naive Bayes (TF-IDF)",
            label="Model"
        ),
    ],
    outputs=[
        gr.Textbox(label="Predicted (ham / spam)"),
        gr.Number(label="Score / Spam probability"),
        gr.Textbox(label="Label distribution (k-NN)", lines=3),
    ],
    title="Spam Email Detector",
    description=description,
)

if __name__ == "__main__":
    iface.launch()
