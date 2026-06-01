# ============================================================================
# MODEL TRAINING
# ============================================================================
from tensorflow.keras import optimizers, callbacks

def train_model(model, train_ds, val_ds, epochs, save_path):
    # Loss Functions
    losses = {
        'ingredients_output': 'binary_crossentropy',
        'nutrition_output': 'mean_absolute_error'
    }

    # Loss Weights
    loss_weights = {'ingredients_output': 1.0, 'nutrition_output': 0.01}

    # Compilation
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            'ingredients_output': 'binary_accuracy',
            'nutrition_output': 'mae'
        }
    )

    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        callbacks.ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss')
    ]

    print("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1
    )
    print("✓ Training Complete")
    return history
