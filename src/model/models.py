import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
from typing import Dict, Any, Tuple

from app.config import IMG_SIZE, CLASS_NAMES_DICT

IMG_SHAPE = IMG_SIZE + (3,)

# ---- HELPERS ----- 
def early_stopping(patience: int) -> tf.keras.callbacks.EarlyStopping:
    """
    Creates an early stopping callback to prevent overfitting.

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        tf.keras.callbacks.EarlyStopping: The early stopping callback object.
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=True
    )


def lr_decay() -> tf.keras.callbacks.ReduceLROnPlateau:
    """
    Creates a learning rate reduction callback that triggers when a metric has stopped improving.

    Returns:
        tf.keras.callbacks.ReduceLROnPlateau: The learning rate reduction callback object.
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', 
        factor=0.2, 
        patience=5, 
        min_lr=1e-6,
        verbose=1
    )

def checkpoint(filepath: str) -> tf.keras.callbacks.ModelCheckpoint: 
    """
    Creates a model checkpoint callback to save the model weights based on best validation accuracy.

    Args:
        filepath (str): Path to save the model weights.

    Returns:
        tf.keras.callbacks.ModelCheckpoint: The model checkpoint callback object.
    """
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

def get_preds(dataset: tf.data.Dataset, model: tf.keras.Model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    Gets model predictions and true labels from a dataset.

    Args:
        dataset (tf.data.Dataset): The dataset containing images and labels.
        model (tf.keras.Model): The trained model to generate predictions.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing true labels, predicted labels, and raw prediction probabilities.
    """
    y_true = []
    for images, labels in dataset:
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)

    predictions = model.predict(dataset)

    y_pred = np.argmax(predictions, axis=1)
    return y_true, y_pred, predictions


def predict_single_image(imagepath: str, model: tf.keras.models.Model) -> None:
    """
    Predicts the class of a single image using the given model and displays the result.

    Args:
        imagepath (str): The path to the image file.
        model (tf.keras.models.Model): The trained classification model.

    Returns:
        None
    """
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

def get_model_v1(learning_rate: float = 0.0001, dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Builds and compiles MobileNetV3Large classification model version 1 (no fine-tuning).
    The base model weights are frozen.

    Args:
        learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 0.0001.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.

    Returns:
        tf.keras.Model: The compiled classification model.
    """
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

def get_model_v2(fine_tune_at: int, learning_rate: float = 0.0001, dropout_rate: float = 0.2) -> tf.keras.Model:
    """
    Builds and compiles MobileNetV3Large classification model version 2 (with fine-tuning).
    Allows fine-tuning from a specific layer onwards.

    Args:
        fine_tune_at (int): The index of the layer from which to start unfreezing weights for fine-tuning.
        learning_rate (float, optional): Learning rate for the Adam optimizer. Defaults to 0.0001.
        dropout_rate (float, optional): Dropout rate for regularization. Defaults to 0.2.

    Returns:
        tf.keras.Model: The compiled classification model ready for fine-tuning.
    """
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
