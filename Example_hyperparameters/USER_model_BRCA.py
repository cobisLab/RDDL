from tensorflow.keras.layers import *
import tensorflow as tf

from training_func.custom_model import CustomModel


def self_defined_model(dropout_rate, model_name=None):

    inputs = Input(shape=(26380, ))

    x = Dense(16, activation='tanh', kernel_initializer='glorot_uniform')(inputs)
    x = Dropout(dropout_rate)(x)
    x = Dense(16, activation='tanh', kernel_initializer='glorot_uniform')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(8, activation='tanh', kernel_initializer='glorot_uniform')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(8, activation='tanh', kernel_initializer='glorot_uniform')(x)
    x = Dropout(dropout_rate)(x)

    outputs = Dense(2, activation='softmax', kernel_initializer='glorot_uniform')(x)

    model = CustomModel(inputs=inputs, outputs=outputs, name=model_name)

    return model
