# ============================================================================
# ENTRY POINT - EFFICIENTNET PIPELINE
# ============================================================================
from configs.config import EPOCHS, RANDOM_SEED, set_seed
from data.preprocessing import load_and_clean_metadata, prepare_labels
from datasets.pipeline import make_dataset
from models.efficientnet_model import build_model
from training.train import train_model
from evaluation.evaluate import run_evaluation, plot_history
from utils.save_labels import save_ingredients
from sklearn.model_selection import train_test_split

def main():
    # 1. Initialize configuration & reproducibility seed
    set_seed(RANDOM_SEED)

    # 2. Data Cleaning & Metadata Parsing
    df = load_and_clean_metadata()
    mlb, binary_labels, num_classes = prepare_labels(df)
    
    # 3. Train/Val/Test Split: Train (70%), Val (15%), Test (15%)
    train_df, test_df, train_labels, test_labels = train_test_split(
        df, binary_labels, test_size=0.3, random_state=RANDOM_SEED
    )
    val_df, test_df, val_labels, test_labels = train_test_split(
        test_df, test_labels, test_size=0.5, random_state=RANDOM_SEED
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 4. Dataset Pipelines creation
    train_ds = make_dataset(train_df, train_labels, is_train=True)
    val_ds = make_dataset(val_df, val_labels)
    test_ds = make_dataset(test_df, test_labels)
    
    print("✓ Data pipeline ready: Predicts Ingredients + 4 Nutrition Values")
    
    # 5. Build Model
    model = build_model(num_classes)
    model.summary()
    
    # 6. Model Training
    save_path = 'best_food_model.keras'
    history = train_model(model, train_ds, val_ds, EPOCHS, save_path)
    
    # 7. Model Evaluation & Visualization
    plot_history(history)
    run_evaluation(model, test_ds, mlb)
    
    # 8. Save Label Definitions
    save_ingredients(mlb, 'ingredients.json')

if __name__ == '__main__':
    main()
