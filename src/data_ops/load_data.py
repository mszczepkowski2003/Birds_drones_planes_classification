import splitfolders
import os
import sys
from pathlib import Path
from PIL import Image
import shutil
from dotenv import load_dotenv
import tensorflow as tf 
from src.model.config import RANDOM_STATE, BATCH_SIZE, IMG_SIZE
from typing import Tuple, List




def split_data() -> None:
    input_dir = '../data/' 
    output_dir = '../data_split/' 

    splitfolders.ratio(
        input_dir, 
        output=output_dir, 
        seed=RANDOM_STATE, 
        ratio=(.7, .15, .15), 
        move=True)
    print('Data split succesfull')


def del_broken_images(data_path: str) -> None:

    data_dir = Path(data_path)

    for img_path in data_dir.rglob('*'):
        if img_path.is_file() and img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            try:
                with Image.open(img_path) as img:
                    # Konwertujemy na RGB (usuwa kanał alfa z PNG i błędy CMYK)
                    rgb_img = img.convert('RGB')
                    # Nadpisujemy jako czysty, standardowy JPEG
                    rgb_img.save(img_path, 'JPEG')
            except Exception as e:
                print(f"File: {img_path} is broken. Deleting it... Error: {e}")
                os.remove(img_path)
    print('Image cleaning process finished')


def get_data(data_path: str) -> Tuple[List[str], tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    train_dir = f'{data_path}/train'
    val_dir = f'{data_path}/val'
    test_dir = f'{data_path}/test'

    loader_params = {
        'batch_size': BATCH_SIZE,
        'image_size': IMG_SIZE,
        'label_mode': 'int'
    }
    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                shuffle=True, #Model can't learn the order of the pictures
                                                                **loader_params)
    val_dataset = tf.keras.utils.image_dataset_from_directory(val_dir,
                                                                shuffle=False, 
                                                                **loader_params)
    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                shuffle=False,
                                                                **loader_params)
    
    cm = train_dataset.class_names
    AUTOTUNE = tf.data.AUTOTUNE # Dynamically checks How many batches CPU should hold in buffer

    #cache and prefetch for faster learning
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE) 
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return cm, train_dataset, val_dataset, test_dataset
    