import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- ATTENTION LAYER ---------------- #
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
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.v, axes=1), axis=1)
        context = attention_weights * inputs
        context = tf.reduce_sum(context, axis=1)
        return context


# ---------------- MODEL ---------------- #
def build_model():

    # ----------- MEL INPUT ----------- #
    mel_input = layers.Input(shape=(128, 251, 1))

    x = layers.Conv2D(32, (3,3), padding='same')(mel_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.2)(x)

    # reshape for LSTM
    x = layers.Reshape((x.shape[2], x.shape[1] * x.shape[3]))(x)

    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)

    # Attention
    deep_output = AttentionLayer()(x)

    # ----------- HANDCRAFTED INPUT ----------- #
    hand_input = layers.Input(shape=(70,))

    h = layers.Dense(128)(hand_input)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)

    h = layers.Dense(64)(h)
    h = layers.BatchNormalization()(h)
    h = layers.ReLU()(h)
    h = layers.Dropout(0.3)(h)

    # ----------- FUSION ----------- #
    combined = layers.concatenate([deep_output, h])

    z = layers.Dense(256, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)

    output = layers.Dense(4, activation='softmax')(z)

    model = models.Model(inputs=[mel_input, hand_input], outputs=output)

    return model


# ---------------- TEST ---------------- #
if __name__ == "__main__":
    model = build_model()
    model.summary()