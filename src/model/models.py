import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
from typing import Dict, Any

from app.config import IMG_SIZE, CLASS_NAMES_DICT

IMG_SHAPE = IMG_SIZE + (3,)

# ---- HELPERS ----- 
def early_stopping(patience: int) -> tf.keras.callbacks.EarlyStopping:
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=True
    )


def lr_decay() -> tf.keras.callbacks.ReduceLROnPlateau:
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.2, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )

def checkpoint(filepath)-> tf.keras.callbacks.ModelCheckpoint: 
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

def get_preds(dataset, model): 
    y_true = []
    for images, labels in dataset:
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)

    predictions = model.predict(dataset)

    y_pred = np.argmax(predictions, axis=1)
    return y_true, y_pred, predictions


def predict_single_image(imagepath: str, model: tf.keras.models.Model) -> None:
    ori_img = tf.keras.utils.load_img(imagepath)
    img = tf.keras.utils.load_img(imagepath, target_size=IMG_SIZE)

    img_array = tf.keras.utils.img_to_array(img)

    img_array = tf.expand_dims(img_array, 0)

    probabilities = model.predict(img_array)
    pred = np.argmax(probabilities)
    pred_proba = np.max(probabilities)
    plt.figure(figsize=(10,10))
    plt.imshow(ori_img)
    plt.title(f'Predicted: {CLASS_NAMES_DICT[pred]} with {(100*pred_proba):.2f}% probability')
    plt.show()



# ------------------

def get_model_v1(learning_rate = 0.0001, dropout_rate = 0.2) -> tf.keras.Model:
    #Loading the weights of the model 
    base_model = tf.keras.applications.MobileNetV3Large(input_shape = IMG_SHAPE,
                                                include_top = False,
                                                weights = 'imagenet')
  
    # Freezing the weights of the model
    base_model.trainable = False


    inputs = tf.keras.Input(shape = (224, 224, 3))
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomZoom(0.1,0.3), 
        tf.keras.layers.RandomRotation(0.1)
    ])
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input # Rescaling is happening here 
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(3, activation='softmax')
 

    # Model Logic 
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training = False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs)


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Changes the labels from integers to relatable format
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
    
    return model

def get_model_v2(fine_tune_at : int, learning_rate : float = 0.0001, dropout_rate : float = 0.2) -> tf.keras.Model:
    #Loading the weights of the model 
    base_model = tf.keras.applications.MobileNetV3Large(input_shape = IMG_SHAPE,
                                                include_top = False,
                                                weights = 'imagenet')
  
    # Freezing the weights of the model

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False



    inputs = tf.keras.Input(shape = (224, 224, 3))
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomZoom(0.1,0.3), 
        tf.keras.layers.RandomRotation(0.1)
    ])
    preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input # Rescaling is happening here 
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(3, activation='softmax')
 

    # Model Logic 
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training = False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs)


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Changes the labels from integers to relatable format
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
    
    return model
