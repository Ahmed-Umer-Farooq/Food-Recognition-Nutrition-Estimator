# ============================================================================
# DATA PREPROCESSING & CLEANING
# ============================================================================
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from configs.config import METADATA_PATH, IMAGERY_PATH

def load_and_clean_metadata():
    print("Loading metadata...")

    # COMPREHENSIVE IGNORE LIST: Invisible ingredients & condiments
    ignore_list = {
        'salt', 'pepper', 'olive oil', 'vegetable oil', 'sugar', 'brown sugar',
        'vinegar', 'soy sauce', 'lemon juice', 'orange juice', 'grapefruit juice',
        'mayonnaise', 'butter', 'ketchup', 'mustard', 'syrup', 'white wine', 'wine',
        'cream', 'sour cream', 'buttermilk', 'milk', 'flour', 'caesar dressing',
        'vinaigrette', 'salsa', 'pesto', 'plate only', 'deprecated', 'water',
        'lemon', 'lime', 'honey', 'maple syrup', 'balsamic vinegar', 'oil',
        'spray', 'dressing'
    }

    data = []

    # Process both Cafe1 and Cafe2 files
    files = [METADATA_PATH / "dish_metadata_cafe1.csv",
             METADATA_PATH / "dish_metadata_cafe2.csv"]

    for file_path in files:
        if not file_path.exists():
            print(f"Warning: {file_path.name} not found.")
            continue

        # FIX FOR PARSER ERROR:
        # The file has variable length rows (ragged). We explicitly define column names
        # (0 to 399) to force pandas to read all columns without crashing.
        try:
            df_raw = pd.read_csv(file_path, header=None, names=[i for i in range(400)], engine='python')
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            continue

        for _, row in df_raw.iterrows():
            dish_id = str(row[0])

            # --- UPDATED NUTRITION VECTOR (NO MASS) ---
            # row[1]=Cal, row[2]=Mass(SKIP), row[3]=Fat, row[4]=Carb, row[5]=Protein
            nutrition_values = [
                float(row[1]) if pd.notna(row[1]) else 0.0, # Calories
                # Skipped row[2] (Mass)
                float(row[3]) if pd.notna(row[3]) else 0.0, # Fat
                float(row[4]) if pd.notna(row[4]) else 0.0, # Carb
                float(row[5]) if pd.notna(row[5]) else 0.0  # Protein
            ]

            # FILTER 1: Skip incomplete/testing data (0 calories)
            if nutrition_values[0] <= 1.0: continue

            # Parse ingredients
            visible_ingredients = []
            # Ingredients start at col 6, stepping by 7
            for i in range(6, len(row), 7):
                if i+1 >= len(row): break
                ing = row[i+1]

                # Check if ingredient exists and is NOT in the ignore list
                if pd.notna(ing):
                    clean_ing = str(ing).strip().lower()
                    if clean_ing not in ignore_list and len(clean_ing) > 1:
                        visible_ingredients.append(clean_ing)

            # FILTER 2: Skip dishes that have 0 visible ingredients
            if not visible_ingredients: continue

            # FILTER 3: Verify image exists
            img_path = IMAGERY_PATH / dish_id / "rgb.png"
            if not img_path.exists(): continue

            data.append({
                'dish_id': dish_id,
                'path': str(img_path),
                'nutrition': nutrition_values, # Vector of 4 values
                'ingredients': visible_ingredients
            })

    df = pd.DataFrame(data)
    print(f"✓ Data Cleaning Complete. Valid samples: {len(df)}")
    return df

def prepare_labels(df):
    # Prepare Labels (Multi-Label Binarizer)
    mlb = MultiLabelBinarizer()
    binary_labels = mlb.fit_transform(df['ingredients'])
    num_classes = len(mlb.classes_)

    print(f"Unique Visible Ingredients: {num_classes}")
    if len(df) > 0:
        print("Sample Nutrition Vector (Cal, Fat, Carb, Prot):", df.iloc[0]['nutrition'])
    return mlb, binary_labels, num_classes
