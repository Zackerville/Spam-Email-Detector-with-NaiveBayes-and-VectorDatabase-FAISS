import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from .config import *

def load_data():
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['Message', 'Category'])
    df = df.reset_index(drop=True)
    return df


def load_messages_labels():
    df = load_data()
    messages = df['Message'].astype('str').tolist()
    labels = df['Category'].tolist()
    encoder = LabelEncoder()
    label_encoded = encoder.fit_transform(labels)
    return messages, label_encoded

# def train_test_split_data():
#     messages, label_encoded, encoder = load_messages_labels()
#     xtrain, xtest, ytrain, ytest = train_test_split(messages, label_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=label_encoded)
#     return xtrain, xtest, ytrain, ytest, encoder  
    
    