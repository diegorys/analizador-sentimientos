import tensorflow as tf

'''
$Attention(Q, K, V ) = \text{softmax}\left(\dfrac{QK^T}{\sqrt{d_k}}\right)V $
'''


def scaled_dot_product_attention(queries, keys, values, mask):
    product = tf.matmul(queries, keys, transpose_b=True)

    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(keys_dim)

    if mask is not None:
        scaled_product += (mask * -1e9) # Por - infinito para que al aplicar softmax devuelva 0.

    attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values)

    return attention


def loss_function(target, pred):
    mask = tf.math.logical_not(tf.math.equal(target, 0))
    loss_ = loss_object(target, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def evaluate(inp_sentence):
    inp_sentence = \
        [VOCAB_SIZE_EN-2] + \
        tokenizer_en.encode(inp_sentence) + [VOCAB_SIZE_EN-1]
    enc_input = tf.expand_dims(inp_sentence, axis=0)

    output = tf.expand_dims([VOCAB_SIZE_ES-2], axis=0)

    for _ in range(MAX_LENGTH):
        # (1, seq_length, VOCAB_SIZE_ES)
        predictions = transformer(enc_input, output, False)

        prediction = predictions[:, -1:, :]

        predicted_id = tf.cast(tf.argmax(prediction, axis=-1), tf.int32)

        if predicted_id == VOCAB_SIZE_ES-1:
            return tf.squeeze(output, axis=0)

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def translate(sentence):
    output = evaluate(sentence).numpy()

    predicted_sentence = tokenizer_es.decode(
        [i for i in output if i < VOCAB_SIZE_ES-2]
    )

    print("Entrada: {}".format(sentence))
    print("Traducción predicha: {}".format(predicted_sentence))
