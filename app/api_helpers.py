import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
from typing import Dict, Any
from src.model.config import IMG_SIZE, CLASS_NAMES_DICT

IMG_SHAPE = IMG_SIZE + (3,)




def predict_image_api(fname: str, image, model: tf.keras.models.Model) -> Dict[str,Any]:

    img_array = tf.keras.utils.img_to_array(image)

    img_array = tf.expand_dims(img_array, 0)

    probabilities = model.predict(img_array)
    pred = np.argmax(probabilities)
    pred_proba = np.max(probabilities)
    return {'filename': fname,
            'prediction': CLASS_NAMES_DICT[pred],
            'confidence': float(pred_proba)}

def predict_batch_api(fnames, batch_tensor, model):
    all_probabilities = model.predict(batch_tensor)
    
    batch_result = []
    for fname, proba in zip(fnames, all_probabilities): 
        pred = np.argmax(proba)
        pred_proba = proba[pred]

        batch_result.append({'filename': fname,
                    'prediction': CLASS_NAMES_DICT[pred],
                    'confidence': float(pred_proba)})
        
    return batch_result
   