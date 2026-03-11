from PIL import Image
import numpy as np
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, status

from pathlib import Path
import tensorflow as tf
from app.config import IMG_SIZE, CLASS_NAMES_DICT
from typing import List, Dict, Any

from app.api_helpers import predict_image_api, predict_batch_api

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / 'final_model.keras'

app = FastAPI(title='Flying Objects Classification API', version='1.0')

try: 
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Couldn't load the model")


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Endpoint for reading and predicting the class of a single image file.

    Args:
        file (UploadFile): Image file in format such as png, jpg, jpeg, or webp.

    Returns:
        Dict[str, Any]: A dictionary containing the filename, predicted class, and confidence score.
    """
    # 1. Reading the bytes of the file 
    contents = await file.read()
    # 2. Converting the contents to image
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    # 3. Preprocess and predict
    result = predict_image_api(fname=file.filename, 
                               image=image,
                               model = model)
    return result 


@app.post("/predict-batch")
async def predict_batch_of_images(
    files: List[UploadFile] = File(...)
) -> List[Dict[str, Any]]:
    """
    Endpoint for reading and predicting the classes of a batch of images (maximum 32).

    Args:
        files (List[UploadFile]): A list of image files in format such as png, jpg, jpeg, or webp.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each containing the filename, predicted class, and confidence score.

    Raises:
        HTTPException: If the requested batch size exceeds the maximum limit (32).
    """
    MAX_BATCH_SIZE = 32
    all_images = []
    filenames = []
    if len(files) > 32: 
        raise HTTPException(
            status_code = status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f'Requested batch is too big! Max valid number of images is{MAX_BATCH_SIZE}'
        )

    for file in files: 
        contents = await file.read()
    # 2. Converting the contents to image
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)

        img_array = tf.keras.utils.img_to_array(image)

        all_images.append(img_array)
        filenames.append(file.filename)

    batch_tensor = np.stack(all_images, axis=0)
    result = predict_batch_api(fnames=filenames,
                               batch_tensor=batch_tensor,
                               model=model)
    return result




if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host = '0.0.0.0', port=8000)

    