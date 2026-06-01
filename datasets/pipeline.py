# ============================================================================
# DATA SPLITTING & TF DATASET PIPELINE
# ============================================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from configs.config import IMG_SIZE, BATCH_SIZE

# Image Parsing Function
def process_path(path, ingredients_target, nutrition_target):
    # Load Image
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]) # Resize
    img = tf.cast(img, tf.float32)

    # Return inputs and dictionary of outputs
    return img, {
        'ingredients_output': ingredients_target,
        'nutrition_output': nutrition_target
    }

# Augmentation Layer (Applied inside dataset pipeline)
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

def make_dataset(dataframe, labels, is_train=False):
    paths = dataframe['path'].values

    # Stack nutrition lists into a proper Numpy Matrix (N, 4)
    nutrition_stack = np.vstack(dataframe['nutrition'].values).astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, nutrition_stack))

    # Map raw paths to images
    ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

    if is_train:
        # Apply augmentation only to training set
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                   num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=1000)

    # Batch and Prefetch
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
