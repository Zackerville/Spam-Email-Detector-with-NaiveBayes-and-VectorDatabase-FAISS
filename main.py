import argparse
from src.nb_pipeline import *
from src.vector_db_knn_pipeline import *
from src.kmeans_pipeline import *

def run_naive_bayes():
    model, vocab, idf, metrics = train_nbclassifier()
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"F1 Score: {metrics['f1']}")
    print(f"ROC_AUC: {metrics['ROC']}")
    print("Classification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    plot_confusion_matrix(metrics["confusion_matrix"], ["ham", "spam"], save_path="visualize/confusion_matrix_nb.png")

    test_message = "Congratulations! You have won a free ticket. Call now to claim your prize."
    result, yprob = predict_nb(test_message, model=model, vocab=vocab, idf=idf)
    print(f"\n[Naive Bayes] Message: {test_message}")
    print(f"[Naive Bayes] Predicted: {result} {yprob}")


def run_faiss_knn():
    state, results, errors = train_and_evaluate_vector_db_knn()
    print("\nFAISS k-NN accuracy:")
    for k, acc in results.items():
        print(f"k = {k}: accuracy = {acc:.4f}")

    test_message = "Congratulations! You have won a free ticket. Call now to claim your prize."
    out = predict_knn(test_message, state, k=3)
    print(f"\n[FAISS k-NN] Message: {test_message}")

    label_map = {"0": "ham", "1": "spam", 0: "ham", 1: "spam"}
    decoded = label_map.get(out["prediction"], str(out["prediction"]))
    print(f"[FAISS k-NN] Predicted: {decoded} ({out['prediction']})")
    print(f"[FAISS k-NN] Label distribution in top-3: {out['label_distribution']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["nb", "faiss", "kmeans"], default="nb")
    args = parser.parse_args()

    if args.mode == "nb":
        run_naive_bayes()
    elif args.mode == "faiss":
        run_faiss_knn()
