import numpy as np 
import tensorflow as tf 
from tensorflow.keras import layers, models 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

from tensorflow.python.ops.gen_data_flow_ops import padding_fifo_queue

fra_path = "fra.txt"
deu_path = "deu.txt"
LIMIT = 5000

sentences = []
labels = []

def load_from_file(filepath, target_lang):
    sents = []
    lbls = []
    engs = []

    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= LIMIT: break
                parts = line.split('\t')
                if len(parts) >= 2:
                    engs.append(parts[0])
                    sents.append(parts[1])
                    lbls.append(target_lang)
    else:
        print(f"Nie znaleziono pliku: {filepath}")

    return sents, lbls, engs


fr_sents, fr_lbls, en_sents = load_from_file(fra_path, "fr")
de_sents, de_lbls, _ = load_from_file(deu_path, "de")
en_lbls = ["en"] * len(en_sents)

if fr_sents and de_sents:
    sentences = fr_sents + de_sents + en_sents
    labels = fr_lbls + de_lbls + en_lbls

    combined = list(zip(sentences, labels))
    np.random.shuffle(combined)
    sentences, labels = zip(*combined)

    print(f'Wczytano łącznie zdan: {len(sentences)}')
else:
    print("Brak plików fra.txt / deu.txt")

uniq_chars = sorted(set(c for s in sentences for c in s))

all_chars = ["<pad>", "<unk>"] + uniq_chars


char_to_id = {ch: i for i, ch in enumerate(all_chars)}
id_to_char = {i: ch for ch, i in char_to_id.items()}


vocab_size = len(all_chars)
print(f'Rozmiarr słownia znakow: {vocab_size}')


def encode_sentence(sent, mapping):
    
    return [mapping.get(ch, mapping["<unk>"]) for ch in sent]

encoded_sentence = [encode_sentence(s, char_to_id) for s in sentences]

max_len = max(len(s) for s in encoded_sentence)

X = pad_sequences(encoded_sentence, maxlen=max_len, padding="post", value=char_to_id["<pad>"])

X = np.array(X, dtype=np.int32)


print(f'Kształt danych wejsciowych X: {X.shape}')

label_map = {label: i for i, label in enumerate(sorted(set(labels)))}
inverse_label_map = {i: label for label, i in label_map.items()}

y = np.array([label_map[l] for l in labels], dtype=np.int32)
num_classes = len(label_map)

print(f'Klasy: {label_map}')


embedding_dim = 64
lstm_units = 64

model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True),
    layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
    ])

model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
        )

model.summary()

history = model.fit(X, y, epochs=25, batch_size=8, verbose=1)

def Predict_language(text):
    seq = encode_sentence(text, char_to_id)
    padded = pad_sequences([seq], maxlen=max_len, padding='post', value=char_to_id["<pad>"])

    padded = np.array(padded, dtype=np.int32)

    pred_prob = model.predict(padded, verbose=0)[0]
    pred_class = np.argmax(pred_prob)

    return inverse_label_map[pred_class], pred_prob[pred_class]

print(f'test predykcji')


test_sentences = [
    "Hello world",         # Oczekiwane: en
    "Guten Tag",           # Oczekiwane: de
    "Ça va bien",          # Oczekiwane: fr
    "s'il vous plaît",     # Oczekiwane: fr
    "Das ist gut",         # Oczekiwane: de
    "Wie Gehts!",          # Oczekiwane: de
    "What's down with?"    # Oczekiwane: en
]

for s in test_sentences:
    lang, conf = Predict_language(s)
    print(f"Zdanie: '{s}' -> Wykryto: {lang.upper()} ({conf:0.2f})")
