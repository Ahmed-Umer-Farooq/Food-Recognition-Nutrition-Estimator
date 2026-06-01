# Multi-Task Food & Nutrition Recognition Project

A multi-task deep learning pipeline that performs simultaneous food ingredient recognition (multi-label classification) and nutrition estimation (regression for Calories, Fat, Carbs, and Protein) from overhead food imagery.

## Project Structure

```text
food-nutrition-recognition/
│
├── configs/
│   └── config.py                  # Configuration and constants
│
├── data/
│   └── preprocessing.py           # Data cleaning & loading logic
│
├── datasets/
│   └── pipeline.py                # Image processing & augmentation pipeline
│
├── models/
│   ├── efficientnet_model.py      # Multi-task model using EfficientNetB0 backbone
│   └── resnet_model.py            # Multi-task model using ResNet50 backbone
│
├── training/
│   └── train.py                   # Model compilation and training logic
│
├── evaluation/
│   └── evaluate.py                # Model evaluation and prediction visualization
│
├── utils/
│   └── save_labels.py             # Exporting ingredient labels to JSON
│
├── notebooks/
│   └── FYP_Updated.ipynb          # Original notebook kept for reference
│
├── main_efficientnet.py           # Entry point for training EfficientNetB0 model
├── main_resnet.py                 # Entry point for training ResNet50 model
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd food-nutrition-recognition
   ```

2. **Install requirements**:
   We recommend setting up a virtual environment first:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Dataset Path**:
   The dataset paths are loaded from environment variables or default to standard Google Drive paths. Set your local path before execution:
   - **On Windows (PowerShell)**:
     ```powershell
     $env:DATASET_PATH="C:\path\to\your\nutrition5k_dataset"
     ```
   - **On Linux/macOS**:
     ```bash
     export DATASET_PATH="/path/to/your/nutrition5k_dataset"
     ```

## How to Run

- **To run training with the EfficientNetB0 backbone**:
  ```bash
  python main_efficientnet.py
  ```

- **To run training with the ResNet50 backbone**:
  ```bash
  python main_resnet.py
  ```

> [!NOTE]
> The original exploratory Jupyter notebook is preserved under `notebooks/FYP_Updated.ipynb` for reference.
