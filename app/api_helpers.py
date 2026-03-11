import tensorflow as tf
import numpy as np 
from typing import Dict, Any, List
from app.config import IMG_SIZE, CLASS_NAMES_DICT

IMG_SHAPE = IMG_SIZE + (3,)




def predict_image_api(fname: str, image: Any, model: tf.keras.models.Model) -> Dict[str, Any]:
    """
    Predicts the class of a single image for the API.

    Args:
        fname (str): The filename of the image.
        image (Any): The image object (e.g., PIL Image) to be predicted.
        model (tf.keras.models.Model): The trained classification model.

    Returns:
        Dict[str, Any]: A dictionary containing the filename, predicted class, and confidence score.
    """
    img_array = tf.keras.utils.img_to_array(image)

    img_array = tf.expand_dims(img_array, 0)

    probabilities = model.predict(img_array)
    pred = np.argmax(probabilities)
    pred_proba = np.max(probabilities)
    return {'filename': fname,
            'prediction': CLASS_NAMES_DICT[pred],
            'confidence': float(pred_proba)}

def predict_batch_api(fnames: List[str], batch_tensor: tf.Tensor, model: tf.keras.models.Model) -> List[Dict[str, Any]]:
    """
    Predicts the classes of a batch of images for the API.

    Args:
        fnames (List[str]): A list of filenames corresponding to the images in the batch.
        batch_tensor (tf.Tensor): The preprocessed image batch tensor.
        model (tf.keras.models.Model): The trained classification model.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing filename, prediction, and confidence score for each image.
    """
    all_probabilities = model.predict(batch_tensor)
    
    batch_result = []
    for fname, proba in zip(fnames, all_probabilities): 
        pred = np.argmax(proba)
        pred_proba = proba[pred]

        batch_result.append({'filename': fname,
                    'prediction': CLASS_NAMES_DICT[pred],
                    'confidence': float(pred_proba)})
        
    return batch_result
   