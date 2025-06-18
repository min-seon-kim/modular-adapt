import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from sklearn.neighbors import KernelDensity

class OnlineOODDetector:
    def __init__(self, alpha=0.05, tau=3.0, epsilon=1e-6):
        self.alpha = alpha
        self.tau = tau
        self.epsilon = epsilon
        self.kde = None
        self.mu = None
        self.sigma = None

    def initialize(self, indicator_samples):
        self.kde = KernelDensity(kernel='gaussian', bandwidth=2.0)
        self.kde.fit(indicator_samples)
        log_density = self.kde.score_samples(indicator_samples)
        energy = -log_density
        self.mu = np.mean(energy)
        self.sigma = np.mean(np.abs(energy - self.mu))

    def compute_energy(self, x):
        log_density = self.kde.score_samples(x)
        return -log_density[0]

    def update_statistics(self, energy):
        delta = abs(energy - self.mu)
        self.sigma = (1 - self.alpha) * self.sigma + self.alpha * delta
        self.sigma = max(self.sigma, 1e-3)
        self.mu = (1 - self.alpha) * self.mu + self.alpha * energy

    def detect(self, x):
        energy = self.compute_energy(x)
        z = (energy - self.mu) / (self.sigma + self.epsilon)
        self.update_statistics(energy)
        return abs(z) > self.tau, energy, z


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embedding_matrix):
        super().__init__()
        self.token_emb = layers.Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=200,
                                trainable=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embedding_matrix.shape[1], trainable=True)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def get_cnn_model(embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model
           

def get_lstm_model(embedding_layer):
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(32,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    return model
           

def get_transformer(embedding_matrix):
    num_heads = 2
    ff_dim = 32
    maxlen = 200
    embed_dim = 100

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, embedding_matrix)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model