import tensorflow as tf
from tensorflow.keras import layers


class DCNN(tf.keras.Model):

    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,  # nb_filters: filtros de 2, 3 o 4 palabras.
                 # número de neuronas de la Feed Fordward Network de la capa oculta de la RNN totalmente conectada.
                 FFN_units=512,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,  # si se usa para predecir o entrenar.
                 name="dcnn"):
        super(DCNN, self).__init__(name=name)
        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim)
        # Filtro que analiza palabras de dos en dos
        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,  # Filtra elementos de dos en dos
                                    # Siempre pille dos palabras válidas las unas con las otras.
                                    padding="valid",
                                    activation="relu")

        # Filtro que analiza palabras de tres en tres
        self.trigram = layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,  # Filtra elementos de dos en dos
                                     # Siempre pille dos palabras válidas las unas con las otras.
                                     padding="valid",
                                     activation="relu")

        # Filtro que analiza palabras de cuatro en cuatro
        self.fourgram = layers.Conv1D(filters=nb_filters,
                                      kernel_size=4,  # Filtra elementos de dos en dos
                                      # Siempre pille dos palabras válidas las unas con las otras.
                                      padding="valid",
                                      activation="relu")

        self.pool = layers.GlobalMaxPool1D()    # No tenemos variable de entrenamiento
        # así que podemos usar la misma capa
        # para cada paso de pooling
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        # para prevenir overfitting
        self.dropout = layers.Dropout(rate=dropout_rate)
        # Capa de salida:
        if nb_filters == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")

    def call(self, inputs, training):
        x = self.embedding(input)
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        x_3 = self.fourgram(x)
        x_3 = self.pool(x_3)

        merged = tf.concat([x_1, x_2, x_3], axis=-1)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)

        return output
