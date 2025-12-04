import gradio as gr
from src.nb_pipeline import *

model, vocab, idf, metrics = train_nbclassifier()
acc = metrics['accuracy']
f1 = metrics['f1']
roc = metrics['ROC']

def classify_email(text):
    if not text or not text.strip():
        return 'Empty Content', 0.0
    label, prob = predict_nb(text, model=model, vocab=vocab, idf=idf)
    return label, prob

iface = gr.Interface(
    fn=classify_email,
    inputs=gr.Textbox(lines=6, label='Type email content here'),
    outputs=[
        gr.Textbox(label='Predicted (ham / spam)'),
        gr.Number(label='Spam probability'),
    ],
    title='Spam Email Detector - NaiveBayes + TFIDF',
    description=f'NaiveBayes. ACC: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc:.4f}',
)

if __name__ == '__main__':
    iface.launch()
