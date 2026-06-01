# ============================================================================
# MODEL ARCHITECTURE (EfficientNet Multi-Task)
# ============================================================================
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers
import tensorflow as tf
from configs.config import IMG_SIZE

def build_model(num_classes):
    # Backbone: EfficientNetB0 (As per PDF Section 6.2)
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = True # Fine-tuning

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    # --- HEAD 1: Ingredients ---
    ingredients_output = layers.Dense(num_classes, activation='sigmoid', name='ingredients_output')(x)

    # --- HEAD 2: Nutrition (Regression) ---
    # Output size is 4: [Calories, Fat, Carb, Protein]
    nutrition_output = layers.Dense(4, activation='linear', name='nutrition_output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[ingredients_output, nutrition_output])
    return model
