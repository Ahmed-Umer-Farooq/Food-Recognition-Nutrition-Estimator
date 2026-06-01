# ============================================================================
# EVALUATION & ADVANCED PREDICTION VISUALIZATION
# ============================================================================
import os
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             classification_report)

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

def run_advanced_evaluation(model, test_ds, mlb, output_dir="advanced_evaluation_outputs"):
    print("🚀 Starting Advanced Evaluation Pipeline...")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("⚡ Running Inference on Test Set...")
    all_pred_ing, all_true_ing = [], []
    all_pred_nut, all_true_nut = [], []

    for images, targets in test_ds:
        preds = model.predict(images, verbose=0)
        all_pred_ing.append(preds[0])
        all_true_ing.append(targets['ingredients_output'].numpy())
        all_pred_nut.append(preds[1])
        all_true_nut.append(targets['nutrition_output'].numpy())

    y_pred_ing = np.vstack(all_pred_ing)
    y_true_ing = np.vstack(all_true_ing)
    y_pred_nut = np.vstack(all_pred_nut)
    y_true_nut = np.vstack(all_true_nut)

    print("📊 Generating Detailed Report...")
    metrics_report = "=== ADVANCED MODEL EVALUATION REPORT ===\n\n"

    # A. Ingredients
    y_pred_ing_bin = (y_pred_ing > 0.3).astype(int)
    metrics_report += "--- 1. INGREDIENT CLASSIFICATION ---\n"
    metrics_report += f"Micro Precision: {precision_score(y_true_ing, y_pred_ing_bin, average='micro'):.4f}\n"
    metrics_report += f"Micro Recall:    {recall_score(y_true_ing, y_pred_ing_bin, average='micro'):.4f}\n"
    metrics_report += f"Micro F1 Score:  {f1_score(y_true_ing, y_pred_ing_bin, average='micro'):.4f}\n\n"

    # Per-Class Report (Top 50 ingredients by frequency)
    class_names = list(mlb.classes_)
    metrics_report += "--- PER-CLASS PERFORMANCE (Top 50 Ingredients) ---\n"
    class_counts = np.sum(y_true_ing, axis=0)
    sorted_indices = np.argsort(class_counts)[::-1][:50]
    
    cls_report = classification_report(y_true_ing, y_pred_ing_bin, target_names=class_names, output_dict=True, zero_division=0)

    metrics_report += f"{'Ingredient':<20} {'F1-Score':<10} {'Support':<10}\n"
    metrics_report += "-"*45 + "\n"
    top_50_names = []
    top_50_f1 = []

    for idx in sorted_indices:
        name = class_names[idx]
        score = cls_report[name]['f1-score']
        support = cls_report[name]['support']
        metrics_report += f"{name:<20} {score:<10.2f} {support:<10}\n"
        top_50_names.append(name)
        top_50_f1.append(score)
    metrics_report += "\n"

    # B. Nutrition (Trained in raw units)
    y_pred_real = y_pred_nut
    y_true_real = y_true_nut
    nutrients = ['Calories', 'Fat', 'Carbs', 'Protein']

    metrics_report += "--- 2. NUTRITION REGRESSION ---\n"
    metrics_report += f"{'Nutrient':<15} {'MAE':<10} {'RMSE':<10} {'R2':<10} {'Mean%Err':<10}\n"
    metrics_report += "-"*60 + "\n"

    for i, n in enumerate(nutrients):
        t = y_true_real[:, i]
        p = y_pred_real[:, i]
        mae = mean_absolute_error(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        r2 = r2_score(t, p)
        # Mean Percentage Error (handling zeros safely by adding 1.0)
        mpe = np.mean(np.abs((t - p) / (t + 1.0))) * 100
        metrics_report += f"{n:<15} {mae:<10.2f} {rmse:<10.2f} {r2:<10.4f} {mpe:<10.1f}%\n"

    report_path = os.path.join(output_dir, "detailed_metrics.txt")
    with open(report_path, "w") as f:
        f.write(metrics_report)
    print(f"✓ Detailed report saved to {report_path}")

    # --- PLOTS ---
    print("🎨 Generating Advanced Visualizations...")
    sns.set_style("whitegrid")

    # Graph 1: Top 20 Ingredients F1 Score
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_50_f1[:20], y=top_50_names[:20], palette="viridis")
    plt.title("Top 20 Ingredients Performance (F1 Score)")
    plt.xlabel("F1 Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_1_ingredient_performance.png"))
    plt.close()

    # Graph 2: Nutrition Regression Scatter (4 Subplots)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    for i, n in enumerate(nutrients):
        t = y_true_real[:, i]
        p = y_pred_real[:, i]
        ax = axes[i]
        ax.scatter(t, p, alpha=0.3, s=15, color='royalblue')
        max_val = max(t.max(), p.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2)
        ax.set_title(f"{n}: Actual vs Predicted")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        r2 = r2_score(t, p)
        ax.text(0.05, 0.9, f"R² = {r2:.2f}", transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_2_nutrition_scatter.png"))
    plt.close()

    # Graph 3: Error Distribution Histograms
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    for i, n in enumerate(nutrients):
        errors = p - t
        sns.histplot(errors, kde=True, ax=axes[i], color='purple', bins=50)
        axes[i].set_title(f"{n} Error Distribution")
        axes[i].set_xlabel("Error (Predicted - Actual)")
        axes[i].axvline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "graph_3_error_distribution.png"))
    plt.close()
    
    print(f"✓ Saved advanced plots in {output_dir}")

    # --- 10 Visual Samples ---
    print("📸 Saving 10 visual prediction samples...")
    count = 0
    for images, targets in test_ds:
        b_preds = model.predict(images, verbose=0)
        
        for i in range(len(images)):
            if count >= 10:
                break
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

            # Image
            ax1.imshow(images[i].numpy().astype("uint8"))
            ax1.axis('off')
            ax1.set_title(f"Test Sample #{count+1}")

            # Text Info
            ax2.axis('off')
            true_n = targets['nutrition_output'][i].numpy()
            pred_n = b_preds[1][i]

            text = "NUTRITION ANALYSIS:\n" + "-"*35 + "\n"
            text += f"{'Item':<10} | {'Actual':>8} | {'Pred':>8} | {'Diff':>6}\n"
            text += "-"*35 + "\n"
            
            nut_labels = ['Calories', 'Fat (g)', 'Carbs (g)', 'Protein (g)']
            for j, m in enumerate(nut_labels):
                diff = abs(true_n[j] - pred_n[j])
                text += f"{m:<10} | {true_n[j]:>8.1f} | {pred_n[j]:>8.1f} | {diff:>6.1f}\n"

            text += "\nINGREDIENTS:\n" + "-"*35 + "\n"
            t_idxs = np.where(targets['ingredients_output'][i].numpy() == 1)[0]
            p_idxs = np.where(b_preds[0][i] > 0.3)[0]
            t_names = [class_names[x] for x in t_idxs]
            p_names = [f"{class_names[x]} ({b_preds[0][i][x]:.2f})" for x in p_idxs]

            def wrap(lst):
                line = ", ".join(lst)
                return "\n".join([line[k:k+40] for k in range(0, len(line), 40)])
                
            text += f"True:\n{wrap(t_names)}\n\nPred:\n{wrap(p_names)}"

            ax2.text(0, 1, text, fontsize=11, family='monospace', verticalalignment='top')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{count+1}.png"))
            plt.close()
            
            count += 1
            
        if count >= 10:
            break
            
    print(f"✓ Saved 10 sample prediction images in {output_dir}")
