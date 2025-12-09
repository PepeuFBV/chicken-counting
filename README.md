# Chicken Counting with Pyramid Vision Transformer

Automatic chicken counting in poultry farms using deep learning. Based on the paper:

> **"Transforming Poultry Farming: A Pyramid Vision Transformer Approach for Accurate Chicken Counting"**  
> _Sensors 2024, 24, 2977_  
> DOI: [10.3390/s24092977](https://doi.org/10.3390/s24092977)

## About the Project

This project implements a complete training and inference system for counting chickens in poultry farm images using the architecture proposed in the paper:

-   **Backbone**: Pyramid Vision Transformer v2 B2 (PVT-v2-B2)
-   **PFA**: Pyramid Feature Aggregation to combine multi-scale features
-   **MDC**: Multi-Scale Dilated Convolution Head to generate density maps
-   **Loss**: Curriculum Loss combining Optimal Transport, Total Variation, and Counting Loss
-   **Output**: 2D density map where the sum = estimated number of chickens

## Model Architecture

```
  Input Image (3×256×256)
            ↓
┌────────────────────────┐
│   PVT-v2-B2 Backbone   │
│  (Pyramid Features)    │
└────────────────────────┘
            ↓
    [f1, f2, f3, f4]
            ↓
┌────────────────────────┐
│  Pyramid Feature       │
│  Aggregation (PFA)     │
│  - Lateral connections │
│  - Top-down upsampling │
└────────────────────────┘
            ↓
   Aggregated Features
            ↓
┌────────────────────────┐
│  Multi-Scale Dilated   │
│  Conv Head (MDC)       │
│  - Dilation: 1, 2, 3   │
│  - Fusion + Regression │
└────────────────────────┘
            ↓
   Density Map (1×256×256)
            ↓
        Σ = Count
```

## Quick Start

### 1. Setup Environment

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Dataset

Place your dataset in `data/dataset/` with the following structure:
- Images: `*.jpg` or `*.png`
- Annotations: `*.json` (LabelMe format with point annotations labeled as "chicken")

Example:
```
data/dataset/
├── 1.jpg
├── 1.json
├── 2.jpg
├── 2.json
└── ...
```

### 3. Train the Model

Use the quick training script:

```bash
chmod +x run_train.sh  # First time only
./run_train.sh
```

Or train with custom parameters:

```bash
./run_train.sh --epochs 100 --batch_size 4 --lr 1e-5
```

Or run training directly:

```bash
source .venv/bin/activate
python src/train.py --data_dir data/dataset --out_dir checkpoints --epochs 50 --batch_size 6 --device auto
```

**Training Parameters:**
- `--data_dir`: Path to dataset folder (default: `data/dataset`)
- `--out_dir`: Output directory for checkpoints (default: `checkpoints`)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 8)
- `--lr`: Learning rate (default: 1e-5)
- `--device`: Device to use: `auto`, `cpu`, `cuda`, or `cuda:0` (default: `auto`)
- `--train_frac`: Fraction of data for training vs validation (default: 0.8)
- `--seed`: Random seed for reproducibility (default: 42)

The training script will:
- Split dataset into train/val sets (80/20 by default)
- Generate Gaussian density maps from point annotations (sigma=4.0)
- Train with AdamW optimizer and Curriculum Loss
- Save checkpoints after each epoch to `checkpoints/model_epoch_N.pth`
- Save best model (lowest validation MAE) to `checkpoints/model_best.pth`
- Print validation MAE and RMSE after each epoch

### 4. Run Inference

Run inference on the dataset:

```bash
source .venv/bin/activate
python src/inference.py --weights checkpoints/model_best.pth --data_dir data/dataset --out_dir outputs
```

**Inference Parameters:**
- `--weights`: Path to model checkpoint (default: `checkpoints/model_best.pth`)
- `--data_dir`: Path to input images (default: `data/dataset`)
- `--out_dir`: Output directory for density maps (default: `outputs`)
- `--device`: Device to use: `auto`, `cpu`, `cuda` (default: `auto`)

Output files:
- `outputs/[image_id]_density.npy`: Predicted density map (256×256 numpy array)
- `outputs/[image_id]_density.png`: Density heatmap visualization (requires matplotlib)
- `outputs/results.json`: Summary JSON with predicted counts per image

### 5. Analyze Results

Compute validation metrics and statistics:

```bash
python scripts/compute_stats.py --pred_dir outputs --top_k 20
```

**Analysis Parameters:**
- `--pred_dir`: Directory with predicted density .npy files (default: `outputs`)
- `--gt_csv`: Optional CSV file with ground truth counts (format: `image_id,count`)
- `--gt_density_dir`: Optional directory with ground truth density .npy files
- `--save_csv`: Output CSV filename for per-image stats (default: `stats_per_image.csv`)
- `--top_k`: Number of worst predictions to display (default: 20)

The script will:
- Auto-detect ground truth from LabelMe JSON annotations in `data/dataset/`
- Compute metrics: MAE, RMSE, Relative MAE, MAPE, R²
- Display dataset statistics (mean count, median count)
- Save per-image results to CSV
- Generate scatter plot `gt_vs_pred.png` (requires matplotlib)
- Show top-K worst predictions

**Interpreting Results:**
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and true counts
- **RMSE (Root Mean Squared Error)**: Similar to MAE but penalizes large errors more
- **Relative MAE**: MAE divided by mean ground truth count (lower is better)
  - < 5%: Excellent
  - 5-10%: Good
  - 10-20%: Fair
  - \> 20%: Needs improvement
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **R²**: Coefficient of determination (1.0 = perfect, 0 = baseline)

## Typical Workflow

1. **Prepare dataset** → Place images and LabelMe annotations in `data/dataset/`
2. **Train model** → Run `./run_train.sh --epochs 50`
3. **Run inference** → Run `python src/inference.py --weights checkpoints/model_best.pth`
4. **Analyze results** → Run `python scripts/compute_stats.py`
5. **Iterate** → Adjust hyperparameters, augmentation, or architecture based on analysis

## Tips for Better Results

- **More epochs**: Try 100-200 epochs for better convergence
- **Data augmentation**: Use `src/data_treatment/main.py` to augment training data
- **Batch size**: Reduce if you run out of GPU memory (try 2-4)
- **Learning rate**: Start with 1e-5, reduce to 1e-6 if training plateaus
- **Sigma**: Adjust Gaussian kernel sigma in `src/train.py` (default: 4.0) based on chicken size
- **Pretrained backbone**: Set `pretrained=True` in model initialization for better initialization
