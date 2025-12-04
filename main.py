from src.nb_pipeline import *

if __name__ == "__main__":
    model, vocab, idf, metrics = train_nbclassifier()
    print(f'Accuracy: {metrics['accuracy']}')
    print(f'F1 Score: {metrics['f1']}')
    print(f'ROC_AUC: {metrics['ROC']}')
    print('Classification Report:')
    print(metrics['classification_report'])
    print('Confusion Matrix:')
    print(metrics['confusion_matrix'])
    plot_confusion_matrix(metrics["confusion_matrix"], ["ham", "spam"], save_path="visualize/confusion_matrix_nb.png")

    test_message = 'Congratulations! You have won a free ticket. Call now to claim your prize.'
    result, yprob = predict_nb(test_message, model=model, vocab=vocab, idf=idf)
    print(f'\nMessage: {test_message}')
    print(f'Predicted: {result} {yprob}')
