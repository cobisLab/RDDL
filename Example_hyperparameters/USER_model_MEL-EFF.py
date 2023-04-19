from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import *

from training_func.custom_model import CustomModel


def self_defined_model(dropout_rate, model_name=None):

    inputs = Input(shape=(224, 224, 3))
    model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)

    model.trainable = False

    for layer in model.layers[-16:]:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = True

    x = GlobalAveragePooling2D()(model.output)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(2, activation='softmax', kernel_initializer='glorot_normal')(x)

    model = CustomModel(inputs=inputs, outputs=outputs, name=model_name)

    return model
