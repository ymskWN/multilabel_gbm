"""Load dataset for training and testing."""
import tensorflow as tf

batch_size = 32
img_height = 384
img_width = 384


data_dir = "data/"


def load_data():
    """Load dataset for training and testing."""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )
    print(train_ds.shape)
