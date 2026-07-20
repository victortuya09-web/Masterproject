# ===============================
# PROYECTO FINAL NLP - SENTIMENT ANALYSIS
# ===============================

import pandas as pd
import numpy as np
import re
import nltk
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# NLP clásico
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Deep Learning
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

# Transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Descargar recursos NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# ===============================
# 1. CARGA DE DATOS
# ===============================

df = pd.read_csv("yelp.csv")

print(df.head())
print(df.info())

# ===============================
# 2. CREAR VARIABLE OBJETIVO
# ===============================

def map_sentiment(stars):
    if stars <= 2:
        return 0  # negativo
    elif stars == 3:
        return 1  # neutral
    else:
        return 2  # positivo

df['sentiment'] = df['stars'].apply(map_sentiment)

print(df['sentiment'].value_counts())

# ===============================
# 3. PREPROCESAMIENTO (LSTM)
# ===============================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)

# ===============================
# 4. SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
)

# ===============================
# 5. TOKENIZACIÓN Y SECUENCIAS
# ===============================

max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# ===============================
# 6. MODELO LSTM
# ===============================

model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Entrenamiento
model.fit(
    X_train_pad, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1
)

# Evaluación LSTM
y_pred_lstm = np.argmax(model.predict(X_test_pad), axis=1)

print("\n=== LSTM RESULTS ===")
print(classification_report(y_test, y_pred_lstm))
print(confusion_matrix(y_test, y_pred_lstm))

# ===============================
# 7. TRANSFORMER (DISTILBERT)
# ===============================

model_name = "distilbert-base-uncased"

tokenizer_bert = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(texts):
    return tokenizer_bert(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=128
    )

train_encodings = tokenize_function(X_train)
test_encodings = tokenize_function(X_test)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels.values

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, y_train)
test_dataset = Dataset(test_encodings, y_test)

model_bert = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model_bert,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Evaluación BERT
preds = trainer.predict(test_dataset)
y_pred_bert = np.argmax(preds.predictions, axis=1)

print("\n=== BERT RESULTS ===")
print(classification_report(y_test, y_pred_bert))
print(confusion_matrix(y_test, y_pred_bert))

# ===============================
# 8. COMPARACIÓN FINAL
# ===============================

print("\n=== COMPARACIÓN ===")
print("Modelo LSTM vs Transformer evaluados correctamente.")