# FASE 1 DE IMPORTAR DEPENDENCIAS:
import numpy as np
import math
import re
import time
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from transformer import Transformer

print('Importación de dependencias')

# FASE 2 PRE PROCESADO DE DATOS:
# Carga de Ficheros
print('Carga de Ficheros')

with open("data/europarl-v7.es-en.en",
          mode="r", encoding="utf-8") as f:
    europarl_en = f.read()
with open("data/europarl-v7.es-en.es",
          mode="r", encoding="utf-8") as f:
    europarl_es = f.read()
with open("data/nonbreaking_prefix.en",
          mode="r", encoding="utf-8") as f:
    non_breaking_prefix_en = f.read()
with open("data/nonbreaking_prefix.es",
          mode="r", encoding="utf-8") as f:
    non_breaking_prefix_es = f.read()

print(europarl_en[:100])
print(europarl_es[:100])

# Limpieza de datos
print('Limpieza de datos')

non_breaking_prefix_en = non_breaking_prefix_en.split("\n")
non_breaking_prefix_en = [' ' + pref + '.' for pref in non_breaking_prefix_en]
non_breaking_prefix_es = non_breaking_prefix_es.split("\n")
non_breaking_prefix_es = [' ' + pref + '.' for pref in non_breaking_prefix_es]

corpus_en = europarl_en
# Añadimos $$$ después de los puntos de frases sin fin
for prefix in non_breaking_prefix_en:
    corpus_en = corpus_en.replace(prefix, prefix + '$$$')
corpus_en = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_en)
# Eliminamos los marcadores $$$
corpus_en = re.sub(r"\.\$\$\$", '', corpus_en)
# Eliminamos espacios múltiples
corpus_en = re.sub(r"  +", " ", corpus_en)
corpus_en = corpus_en.split('\n')

print('Non Breaking Prefix')
corpus_es = europarl_es
for prefix in non_breaking_prefix_es:
    corpus_es = corpus_es.replace(prefix, prefix + '$$$')
corpus_es = re.sub(r"\.(?=[0-9]|[a-z]|[A-Z])", ".$$$", corpus_es)
corpus_es = re.sub(r"\.\$\$\$", '', corpus_es)
corpus_es = re.sub(r"  +", " ", corpus_es)
corpus_es = corpus_es.split('\n')

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    corpus_en, target_vocab_size=2**13)
tokenizer_es = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    corpus_es, target_vocab_size=2**13)

#  Estos dos que sumo son los tokens de inicio y fin de frase.
VOCAB_SIZE_EN = tokenizer_en.vocab_size + 2  # = 8198
VOCAB_SIZE_ES = tokenizer_es.vocab_size + 2  # = 8225

# Tokenizo cada frase y añadimos los tokens de inicio y fin.
inputs = [[VOCAB_SIZE_EN-2] + tokenizer_en.encode(sentence) + [VOCAB_SIZE_EN-1]
          for sentence in corpus_en]
outputs = [[VOCAB_SIZE_ES-2] + tokenizer_es.encode(sentence) + [VOCAB_SIZE_ES-1]
           for sentence in corpus_es]
print('outputs')
print(outputs[0])
print('ok')
MAX_LENGTH = 20
idx_to_remove = [count for count, sent in enumerate(inputs)
                 if len(sent) > MAX_LENGTH]
for idx in reversed(idx_to_remove):
    del inputs[idx]
    del outputs[idx]
idx_to_remove = [count for count, sent in enumerate(outputs)
                 if len(sent) > MAX_LENGTH]
for idx in reversed(idx_to_remove):
    del inputs[idx]
    del outputs[idx]
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=MAX_LENGTH)
outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs,
                                                        value=0,
                                                        padding='post',
                                                        maxlen=MAX_LENGTH)
BATCH_SIZE = 64
BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# FASE 3 ENTRENAMIENTO
print('Entrenamiento')
tf.keras.backend.clear_session()

# Hiper Parámetros
D_MODEL = 128  # 512
NB_LAYERS = 4  # 6
FFN_UNITS = 512  # 2048
NB_PROJ = 8  # 8
DROPOUT_RATE = 0.1  # 0.1

transformer = Transformer(vocab_size_enc=VOCAB_SIZE_EN,
                          vocab_size_dec=VOCAB_SIZE_ES,
                          d_model=D_MODEL,
                          nb_layers=NB_LAYERS,
                          FFN_units=FFN_UNITS,
                          nb_proj=NB_PROJ,
                          dropout_rate=DROPOUT_RATE)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction="none")

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name="train_accuracy")


leaning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(leaning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
checkpoint_path = "./ckpt/"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Último checkpoint restaurado!!")

EPOCHS = 10
for epoch in range(EPOCHS):
    print("Inicio del epoch {}".format(epoch+1))
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (enc_inputs, targets)) in enumerate(dataset):
        dec_inputs = targets[:, :-1]
        dec_outputs_real = targets[:, 1:]
        with tf.GradientTape() as tape:
            predictions = transformer(enc_inputs, dec_inputs, True)
            loss = loss_function(dec_outputs_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(dec_outputs_real, predictions)

        if batch % 50 == 0:
            print("Epoch {} Lote {} Pérdida {:.4f} Precisión {:.4f}".format(
                epoch+1, batch, train_loss.result(), train_accuracy.result()))

    ckpt_save_path = ckpt_manager.save()
    print("Guardando checkpoint para el epoch {} en {}".format(epoch+1,
                                                               ckpt_save_path))
    print("Tiempo que ha tardado 1 epoch: {} segs\n".format(time.time() - start))

translate("This is a problem we have to solve.")
translate("This is a really powerful tool!")
translate("This is an interesting course about Natural Language Processing")