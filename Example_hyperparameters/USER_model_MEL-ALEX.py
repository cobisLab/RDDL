from tensorflow.keras.layers import *
import tensorflow as tf

from training_func.custom_model import CustomModel


def self_defined_model(dropout_rate, model_name=None):

    inputs = Input(shape=(224, 224, 3))

    x = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(256, kernel_initializer='glorot_normal')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(2, activation='softmax', kernel_initializer='glorot_normal')(x)

    model = CustomModel(inputs=inputs, outputs=outputs, name=model_name)

    return model
