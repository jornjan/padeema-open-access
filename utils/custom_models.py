import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Model
from tensorflow import Tensor

from typing import Dict, List


def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3, kernel_init='he_normal') -> Tensor:
    # Full pre-activation variation

    y = layers.BatchNormalization()(x)
    y = layers.ReLU()(y)
    y = layers.Conv2D(kernel_size=kernel_size,
                      kernel_initializer=kernel_init,
                      strides=(1 if not downsample else 2),
                      filters=filters,
                      padding="same")(y)

    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(kernel_size=kernel_size,
                      kernel_initializer=kernel_init,
                      strides=1,
                      filters=filters,
                      padding="same")(y)

    if downsample:
        x = layers.Conv2D(kernel_size=1,
                          kernel_initializer=kernel_init,
                          strides=2,
                          filters=filters,
                          padding="same")(x)

    out = layers.Add()([x, y])
    return out


def ResNet18() -> Model:

    inputs = layers.Input(shape=(224, 224, 1))
    num_filters = 64
    num_blocks_list = [2, 2, 2, 2]

    x = layers.BatchNormalization()(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(kernel_size=7,
                      kernel_initializer='he_normal',
                      strides=2,
                      padding='same',
                      filters=num_filters
                      )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    for i, num_blocks in enumerate(num_blocks_list):
        for j in range(num_blocks):
            x = residual_block(x, downsample=(
                j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    x = layers.GlobalAveragePooling2D()(x)
    outputs = x

    return Model(inputs, outputs)


def AugmentationLayer(input_shape=(256, 256, 1), target_size=(224, 224), **params) -> Model:

    inputs = layers.Input(shape=input_shape)
    x = inputs
    x = preprocessing.Resizing(target_size[0], target_size[1])(x)
    x = preprocessing.Rescaling(scale=1./255)(x)
    x = preprocessing.RandomFlip('horizontal')(x)
    x = preprocessing.RandomRotation(0.1)(x)
    outputs = x

    return Model(inputs=inputs, outputs=outputs, name='augmentation')


def build_resnet(aug_params: dict={}) -> Model:
    inputs = layers.Input(shape=(256, 256,1))
    x = AugmentationLayer(**aug_params)(inputs)
    x = ResNet18()(x)
    # x = layers.Dense(8, kernel_initializer='he_normal', activation='relu', name='feature_layer')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs, name='resnet')


def build_cnnlstm(ct_input=False, lstm_units=8, hidden_units=[32], dropout=0.1, resnet=None, aug_params: dict={}) -> Model:
    thorax_input = layers.Input(shape=(None, 256, 256,1), name='thorax_img')
    mask = layers.Masking().compute_mask(thorax_input)
    mask = mask[:,:,1,1]
    
    if resnet is  None:
        resnet = build_resnet(aug_params)
    
    resnet = Model(resnet.input, resnet.layers[-1].input)
    x = layers.TimeDistributed(resnet, name='td_resnet')(thorax_input)

    
    # Hidden layers
    for nunits in hidden_units:
        x = layers.Dense(nunits, kernel_initializer='he_normal', activation='relu')(x)

    if dropout>0:
        x = layers.Dropout(dropout)(x)
        
    if ct_input:
        ct_input = layers.Input(shape=(None, 1), name='ct_img')
        x = layers.concatenate([x, ct_input])
        inputs = [thorax_input, ct_input]
    else:
        inputs = thorax_input
        
    x = layers.LSTM(lstm_units, name='feature_layer')(x, mask=mask)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs)
    

def LabLSTM(n_features: int,
            units: int=4,
            dropout: float=0.,
            bd: bool=False,
            normalize='batch',
            ) -> Model:
    inputs = layers.Input(shape=(None, n_features))
    x = layers.Masking()(inputs)
    
    if normalize is not None and normalize=='batch':
        x = layers.BatchNormalization()(x)
    elif normalize is not None and normalize=='layer':
        x = layers.LayerNormalization()(x)
    
    lstm_layer = layers.LSTM(units, dropout=dropout, name='feature_layer')
    x = layers.Bidirectional(lstm_layer, name='feature_layer')(x) if bd else lstm_layer(x)
    
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=inputs, outputs=outputs)


def StaticModel(n_features: int,
                hidden_units: int=4):
    inputs = layers.Input((n_features,))
    x = layers.Dense(hidden_units, activation='relu', name='feature_layer')(inputs)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs) 


class TensorFusionLayer(layers.Layer):    
    def __init__(self, name=None):
        super().__init__(trainable=False, name=name)
        
    def build(self, input_shape):
        for shape in input_shape:
            assert len(shape) <= 2, 'Input shapes must be of length 2 or smaller'
        return super().build(input_shape)
    
    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=False):
        n_inputs = len(inputs)
        batch_size = tf.shape(inputs[0])[0]
        
        ones_shape = tf.stack([batch_size, 1])
        ones = tf.ones(shape=ones_shape)
        inputs = [tf.concat([ones, x], axis=-1) for x in inputs]
        
        shaped_inputs = list() 
        for i, x in enumerate(inputs):
            new_shape = n_inputs*[1]
            new_shape[i] = x.get_shape()[-1]
            new_shape = tf.stack([batch_size, *new_shape])
            shaped_inputs.append(tf.reshape(x, shape=new_shape))
            
        result = shaped_inputs[0]
        for i in range(1, n_inputs):
            result *= shaped_inputs[i]
        
        return result
    

def clip_model(model: Model) -> Model:
    feature_layer = model.get_layer('feature_layer')
    return Model(model.input, feature_layer.output)



def build_multimodal_model(um_models: Dict[str, Model], 
                           fusion: str, 
                           hidden_units: List[int]=[],
                           include_um_outputs: bool = False) -> Model:
    inputs, um_outputs = [], []
    for name, model in um_models.items():
        model = model if fusion=='late' else clip_model(model)

        um_input = layers.Input(shape=model.input_shape[1:], name=name)
        model._name = f'{name}_model'
        inputs.append(um_input)
        um_outputs.append(model(um_input))
        
        
    if fusion == 'tensor':
        x = TensorFusionLayer(name='fusion_layer')(um_outputs)
        x = layers.Flatten()(x)
        
    elif fusion=='mid': 
        x = layers.concatenate(um_outputs, name='fusion_layer')
    
    elif fusion=='late':
        x = layers.concatenate(um_outputs, name='fusion_layer')
    else:
        raise NotImplementedError('Unknown fusion strategy')
    
    for units in hidden_units:
        if units==0: break
        x = layers.Dense(units, activation='relu', kernel_initializer='he_normal')(x)

    outputs = layers.Dense(1, activation='sigmoid', name='prediction')(x)
    
    if include_um_outputs:
        outputs = layers.concatenate([outputs] + um_outputs, name='all_predictions')
    return Model(inputs, outputs)



def build_partial_multimodal_model(input_sizes: Dict[str, int], fusion: str, hidden_units: List[int]=[],
                                   dropout: float=0.0, l2reg: float=0.0, normalise=False) -> Model:
    inputs = [layers.Input(shape=(size,), name=name) for name, size in input_sizes.items()]
    if normalise:
        fusion_inputs = [layers.BatchNormalization()(x) for x in inputs]
    else:
        fusion_inputs = inputs
    
    if fusion=='tensor':
        x = TensorFusionLayer(name='fusion_layer')(fusion_inputs)
        x = layers.Flatten()(x)
    elif fusion=='mid' or fusion=='late': 
        x = layers.concatenate(fusion_inputs, name='fusion_layer')
        
    if dropout > 0.0:
        x = layers.Dropout(dropout)(x)
        
    for units in hidden_units:
        if units==0: break
        x = layers.Dense(units, activation='relu', kernel_initializer='he_normal')(x)
    
    outputs = layers.Dense(1, activation='sigmoid', name='prediction', kernel_regularizer=regularizers.l2(l2reg))(x)
    return Model(inputs, outputs)