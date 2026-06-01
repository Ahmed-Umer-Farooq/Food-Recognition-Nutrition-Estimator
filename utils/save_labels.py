# ============================================================================
# UTILITIES - LABEL EXPORT
# ============================================================================
import json

def save_ingredients(mlb, path='ingredients.json'):
    # Export all unique ingredient class labels to json
    labels = list(mlb.classes_)
    with open(path, 'w') as f:
        json.dump(labels, f)
    print(f"✓ Saved label definitions to {path}")
