import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from dcnn import DCNN

# Load data
cols = ["sentiment", "id", "date", "query", "user", "text"]
train_data = pd.read_csv(
    "data/train.csv",
    header=None,
    names=cols,
    engine="python",
    encoding="latin1"
)

test_data = pd.read_csv(
    "data/test.csv",
    header=None,
    names=cols,
    engine="python",
    encoding="latin1"
)

# print(train_data.head(5))

data = train_data

# Limpieza

data.drop(["id", "date", "query", "user"],
          axis=1,
          inplace=True)
# print(data.head(5))


def clean_tweet(tweet):
    tweet = BeautifulSoup(tweet, "lxml").get_text()
    # Eliminamos la @ y su mención
    tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)
    # Eliminamos los links de las URLs
    tweet = re.sub(r"https?://[A-Za-z0-9]+", ' ', tweet)
    # Nos quedamos solamente con los caracteres
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)
    # Eliminamos los espacios en blanco adicionales
    tweet = re.sub(r" +", ' ', tweet)
    return tweet


data_clean = [clean_tweet(tweet) for tweet in data.text]
data_labels = data.sentiment.values
data_labels[data_labels == 4] = 1

# print(data_clean)
# Data clean: obtenemos el corpus limpio.

# Tokenización
tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    # 2**16: dos elevado a 16, coge las 65536 palabras más frecuentes.
    data_clean, target_vocab_size=2**16
)

# Cada palabra se convierte en un número.
data_inputs = [tokenizer.encode(sentence) for sentence in data_clean]

# Padding: para que todas las palabras tengan la misma longitud. Añade ceros.
MAX_LEN = max([len(sentence) for sentence in data_inputs])
data_inputs = tf.keras.preprocessing.sequence.pad_sequences(
    data_inputs,
    value=0,
    padding="post",
    maxlen=MAX_LEN)

# Dividimos en los conjuntos de entrenamiento y de testing

test_idx = np.random.randint(0, 800000, 8000)
test_idx = np.concatenate((test_idx, test_idx+800000))

test_inputs = data_inputs[test_idx]
test_labels = data_labels[test_idx]
train_inputs = np.delete(data_inputs, test_idx, axis=0)
train_labels = np.delete(data_labels, test_idx)

# Estos números los obtenemos mediante ensayo y error.
# Cuántas palabras tieen que identificar el modelo
VOCAB_SIZE = tokenizer.vocab_size
EMB_DIM = 200  # Todas y cada una de las palabras se mapearán a un espacio de dimensión 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = 2  # len(set(train_labels))
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
NB_EPOCHS = 5

# FASE DE ENTRENAMIENTO:

Dcnn = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)

if NB_CLASSES == 2:
    Dcnn.compile(loss="binary_crossentropy",
                 optimizer="adam",
                 metrics=["accuracy"])
else:
    Dcnn.compile(loss="sparse_categorical_crossentropy",
                 optimizer="adam",
                 metrics=["sparse_categorical_accuracy"])

# Sistema que nos permite marcar checkpoints en
# el entrenamiento para que cada cierto tiempo
# se vaya guardando la información y podamos
# continuar entrenamientos o añadir nuevos.
checkpoint_path = "./ckpt"
ckpt = tf.train.Checkpoint(Dcnn=Dcnn)

ckpt_manager = tf.train.CheckpointManager(
    ckpt, checkpoint_path, max_to_keep=5)  # Guarda sólo los cinco últimos

# Si hay checkpoint, restauro
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Último checkpoint restaurado!!')

Dcnn.fit(train_inputs,
         train_labels,
         batch_size=BATCH_SIZE,
         epochs=NB_EPOCHS)  # pasará por cada tweet cinco veces.
ckpt_manager.save()

# FASE DE EVALUACIÓN:
results = Dcnn.evaluate(test_inputs, test_labels, batch_size=BATCH_SIZE)
print(results)

# Hacemos una predicción:
Dcnn(np.array([tokenizer.encode("I hate you")]), training=False).numpy()
# Devuelve la probabilidad de que sea positivo
tokenizer.encode("bad")