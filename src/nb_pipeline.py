from .config import *
from .preprocess import *
from .data_utils import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, f1_score, roc_curve
import matplotlib.pyplot as plt


def train_nbclassifier(test_size=TEST_SIZE, random_state=RANDOM_STATE):
    messages, labels = load_messages_labels()
    token_all = [preprocess_text(m) for m in messages]
    
    xtrain_tokens, xtest_tokens, ytrain, ytest = train_test_split(token_all, labels, test_size=test_size, random_state=random_state, shuffle=True, stratify=labels)
    vocab = create_vocab(xtrain_tokens)
    idf = compute_idf(xtrain_tokens, vocab)
    xtrain = compute_tfidf(xtrain_tokens, vocab, idf)
    xtest = compute_tfidf(xtest_tokens, vocab, idf)
    
    model = MultinomialNB(alpha=NB_ALPHA)
    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)
    yprob = model.predict_proba(xtest)[:, 1]
    
    acc = accuracy_score(ytest, ypred)
    roc = roc_auc_score(ytest, yprob)
    f1 = f1_score(ytest, ypred)
    report = classification_report(ytest, ypred, digits=4)
    cm = confusion_matrix(ytest, ypred)
    metrics = {
        'accuracy': acc,
        'f1': f1,
        'ROC': roc,
        'classification_report': report,
        'confusion_matrix': cm
    }
    plot_roc_curve(ytest, yprob, roc, save_path="visualize/roc_curve_nb.png")

    return model, vocab, idf, metrics


def predict_nb(message, model, vocab, idf):
    tokens = preprocess_text(message)
    x = compute_tfidf([tokens], vocab, idf)
    ypred = model.predict(x)[0]
    yprob = model.predict_proba(x)[0, 1]
    label = 'spam' if ypred == 1 else 'ham'
    return label, float(yprob)


def plot_confusion_matrix(cm, class_names, save_path=None):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    vmax = cm.max()
    thresh = vmax / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)

    
    
def plot_roc_curve(y_true, y_score, auc_value=None, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig = plt.figure()
    if auc_value is not None:
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_value:.3f})")
    else:
        plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)



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

    
    


