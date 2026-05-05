import tensorflow as tf
from tensorflow.keras import layers, models


class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(input_shape[-1],),
                                 initializer='zeros',
                                 trainable=True)
        self.v = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        weights = tf.nn.softmax(tf.tensordot(score, self.v, axes=1), axis=1)
        context = weights * inputs
        return tf.reduce_sum(context, axis=1)


def build_model():

    mel_input = layers.Input(shape=(128, None, 1))

    x = layers.Conv2D(32, (3,3), padding='same')(mel_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(96, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((-1, x.shape[2] * x.shape[3]))(x)

    # 🔥 2-layer BiLSTM
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)

    deep_output = AttentionLayer()(x)

    hand_input = layers.Input(shape=(70,))

    h = layers.Dense(96)(hand_input)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)

    h = layers.Dense(48)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.Dropout(0.4)(h)

    combined = layers.concatenate([deep_output, h])

    z = layers.Dense(192, activation='relu')(combined)
    z = layers.Dropout(0.4)(z)

    output = layers.Dense(4, activation='softmax')(z)

    return models.Model(inputs=[mel_input, hand_input], outputs=output)