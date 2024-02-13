"""Load dataset for training and testing."""
import tensorflow as tf
import pathlib
import numpy as np
from PIL import Image
import pandas as pd
import optuna.integration.lightgbm as lgb_o

batch_size = 32
img_height = 384
img_width = 384


data_dir = "data"


def load_data():
    """Load dataset for training and testing."""
    target_dir = pathlib.Path(data_dir + "/images/202309_001")
    target_csv = pathlib.Path(data_dir + "/annotations/202309_001_multilabel.csv")
    image_files = sorted(target_dir.glob("*.*"))
    images = []
    for file in image_files:
        image = np.array(Image.open(file).resize((384, 384))).flatten()
        images.extend(image)
    label_df = pd.read_csv(target_csv)
    print(label_df)

    # train_ds = tf.keras.utils.image_dataset_from_directory(
    #     data_dir,
    #     validation_split=0.2,
    #     subset="training",
    #     seed=123,
    #     image_size=(img_height, img_width),
    #     batch_size=batch_size,
    # )
    # print(train_ds.shape)


load_data()
