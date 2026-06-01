# ============================================================================
# EVALUATION & PREDICTION VISUALIZATION
# ============================================================================
import matplotlib.pyplot as plt

def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Total Loss
    ax[0].plot(history.history['loss'], label='Train Loss')
    ax[0].plot(history.history['val_loss'], label='Val Loss')
    ax[0].set_title('Total Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()

    # Plot Nutrition MAE
    ax[1].plot(history.history['val_nutrition_output_mae'], label='Val MAE (Nutrition)', color='orange')
    ax[1].set_title('Nutrition Estimation Error (MAE)')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()
    plt.show()

def run_evaluation(model, test_ds, mlb):
    print("\n" + "="*50)
    print("SAMPLE PREDICTION (Ingredients + Nutrition)")
    print("="*50)

    # Grab a random test batch
    for imgs, labels in test_ds.take(1):
        # Predict
        preds = model.predict(imgs)
        pred_ing = preds[0]  # Ingredients Probabilities
        pred_nut = preds[1]  # Nutrition Vector [4]

        # Display the first image in the batch
        plt.figure(figsize=(6,6))
        plt.imshow(imgs[0].numpy().astype("uint8"))
        plt.axis('off')
        plt.show()

        # 1. Decode Ingredients
        top_indices = pred_ing[0].argsort()[-5:][::-1]
        detected_food = [mlb.classes_[i] for i in top_indices]

        # 2. Decode Nutrition
        # Actual Labels: [Calories, Fat, Carb, Protein]
        actual_nut = labels['nutrition_output'][0].numpy()
        predicted_nut = pred_nut[0]

        print(f"Recognized Ingredients: {', '.join(detected_food)}")
        print("-" * 65)
        print(f"{'Nutrient':<12} | {'Actual':<12} | {'Predicted':<12} | {'Diff':<12}")
        print("-" * 65)

        # Only 4 nutrients now (No Mass)
        nutrients = ['Calories', 'Fat (g)', 'Carbs (g)', 'Protein (g)']

        for i, nutrient in enumerate(nutrients):
            act = actual_nut[i]
            pre = predicted_nut[i]
            diff = abs(act - pre)
            print(f"{nutrient:<12} | {act:<12.1f} | {pre:<12.1f} | {diff:<12.1f}")

        print("-" * 65)
        break
