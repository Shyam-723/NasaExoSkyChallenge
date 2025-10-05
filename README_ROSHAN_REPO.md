# 🌟 Enhanced Multimodal Exoplanet Detection Pipeline

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+-red.svg)
![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-93.47%25-brightgreen.svg)
![NASA](https://img.shields.io/badge/Data-NASA%20Kepler-orange.svg)

**🏆 93.47% Validation Accuracy Multimodal Exoplanet Detection System**

A state-of-the-art machine learning pipeline combining tabular features, 1D CNN analysis of Kepler light curves, and 2D CNN processing of target pixel files for highly accurate exoplanet detection.

## 🚀 Quick Start

### Option 1: Use Pre-trained Model (Recommended)
```bash
# Clone repository
git clone https://github.com/RoshanKattil/exoplanet-detection-pipeline.git
cd exoplanet-detection-pipeline
git checkout enhanced-multimodal-pipeline

# Setup environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Test the model immediately
python demo_inference.py --kepid 10797460
```

### Option 2: Interactive Demo
```bash
# Interactive prediction interface
python demo_inference.py --interactive

# Batch predictions
python demo_inference.py --batch-predict --num-samples 10
```

## 🎯 Performance Highlights

- **🏆 93.47% Validation Accuracy** - Primary performance metric
- **📊 97.51% Validation AUC** - Excellent discrimination
- **⚡ 54.6s Training Time** - Fast convergence with early stopping
- **🎚️ Optimal Threshold: 0.9922** - 88.56% TPR, 4.55% FPR
- **📈 2.5% CNN Coverage** - 377 files across 9,777 targets

## 🏗️ Architecture Overview

```
📊 Tabular Features (39) → TabularNet → 
                                        → Fusion Layer → 93.47% Classification
📡 Light Curves (128) → ResidualCNN1D → ↗
🖼️ Pixel Data (32×24×24) → PixelCNN2D → ↗
```

### Model Components
- **TabularNet**: Processes 39 orbital and stellar parameters
- **ResidualCNN1D**: Analyzes detrended Kepler light curve residuals  
- **PixelCNN2D**: Processes target pixel file difference images
- **Fusion Layer**: Attention-weighted multimodal feature combination

## 📊 Performance Evolution

| Version | CNN Coverage | Files | Val Accuracy | Improvement |
|---------|-------------|-------|--------------|-------------|
| Baseline | 0.7% | 70 | 88.85% | - |
| Enhanced V1 | 1.4% | 139 | 88.85% | Stable |
| **Enhanced V2** | **2.5%** | **377** | **93.47%** | **+4.62%** |

## 🔮 Making Predictions

### Single Target Prediction
```python
from src.models import EnhancedMultiModalFusionModel
import torch

# Load pre-trained model
model = EnhancedMultiModalFusionModel(39, 128, (32, 24, 24))
model.load_state_dict(torch.load('models/enhanced_multimodal_fusion_model.pth'))
model.eval()

# Make prediction for KepID 10797460 (known confirmed planet)
kepid = "10797460"
sample_data = prepare_single_sample(kepid)

with torch.no_grad():
    outputs = model(sample_data['tabular'], sample_data['residual'], sample_data['pixel'])
    probability = torch.sigmoid(outputs).item()
    
    prediction = "CONFIRMED" if probability > 0.9922 else "FALSE POSITIVE"
    confidence = probability * 100
    
    print(f"KepID {kepid}: {prediction} ({confidence:.2f}% confidence)")
```

### Command Line Interface
```bash
# Single prediction
python demo_inference.py --kepid 10797460

# Multiple predictions
python demo_inference.py --batch-predict --num-samples 20

# Interactive mode
python demo_inference.py --interactive
```

## 🎓 Training from Scratch

### Full Pipeline (1-2 hours)
```bash
# 1. Generate CNN data (optional - pre-processed data included)
python scripts/rapid_cnn_expansion.py --max-targets 200 --max-time 270

# 2. Standardize CNN data (optional - standardized data included)
python scripts/standardize_cnn_data.py

# 3. Train enhanced model
python train_multimodal_enhanced.py
```

### Expected Training Output
```
🌟 ENHANCED MULTI-MODAL EXOPLANET DETECTION PIPELINE
=================================================================
📊 Training samples: 462
📊 Validation samples: 245
Epoch  45: Loss=0.0840, Val AUC=0.9683, Val Acc=0.9347, Time=54.6s
🛑 Early stopping at epoch 45 (patience=20)
✅ Training complete! Best validation AUC: 0.9751
🏆 ENHANCED MULTI-MODAL RESULTS
🎯 Validation Accuracy: 93.47%
💾 Enhanced model saved to models/enhanced_multimodal_fusion_model.pth
```

## 📈 Data Pipeline

### 1. Raw Data Sources
- **Primary Dataset**: `data/raw/lighkurve_KOI_dataset_enriched.csv`
- **Source**: NASA Exoplanet Archive + Kepler Object of Interest catalog
- **Size**: 9,777 exoplanet candidates with disposition labels

### 2. CNN Data Generation
```bash
# Automatic data expansion with smart targeting
python scripts/rapid_cnn_expansion.py --max-targets 200 --max-time 270

# Generates:
# - 243 residual windows (128-point light curve residuals)
# - 134 pixel differences (32×24×24 TPF difference images)
# - 2.5% total coverage across confirmed/false positive targets
```

### 3. Data Standardization
```bash
# Prepare for training
python scripts/standardize_cnn_data.py

# Output:
# - data/processed/residual_windows_std/ (243 files)
# - data/processed/pixel_diffs_std/ (134 files)
```

## 🎛️ Model Configuration

### Architecture Parameters
```python
# Model dimensions
TABULAR_DIM = 39                # Orbital + stellar features
RESIDUAL_LENGTH = 128           # Light curve residual points
PIXEL_SHAPE = (32, 24, 24)      # Phase-folded pixel differences

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 20
OPTIMAL_THRESHOLD = 0.9922      # For 88.56% TPR, 4.55% FPR
```

### Feature Engineering
- **Tabular Features (39)**:
  - 9 base KOI parameters (period, depth, duration, etc.)
  - 21 stellar parameters from NASA Exoplanet Archive
  - 9 engineered features (duty cycle, log scales, error ratios)
- **CNN Features**:
  - 1D: Detrended light curve residuals centered on transit events
  - 2D: Phase-folded target pixel file differences showing spatial signatures

## 📋 Dependencies

### Core Requirements
```txt
torch>=2.8.0              # Deep learning framework
lightkurve>=2.5.1          # Kepler data processing
scikit-learn>=1.7.2        # ML utilities and metrics
pandas>=2.3.3              # Data manipulation
numpy>=2.3.3               # Numerical computing
astropy>=7.1.0             # Astronomical calculations
scikit-image>=0.25.2       # Image processing
matplotlib>=3.10.6         # Plotting and visualization
tqdm>=4.67.1               # Progress bars
```

### Installation
```bash
pip install -r requirements.txt
```

## 🗂️ Repository Structure

```
exoplanet-detection-pipeline/
├── README.md                              # This file
├── requirements.txt                       # Dependencies
├── demo_inference.py                      # 🔮 Complete inference demo
├── train_multimodal_enhanced.py          # 🚀 Main training script
├── TRAINING_GUIDE.md                      # 📚 Step-by-step guide
├── GITHUB_RELEASE_SUMMARY.md              # 📄 Deployment guide
├── src/                                   # Core source code
│   ├── models.py                         # 🧠 Enhanced multimodal architecture
│   ├── data_loader.py                    # 📊 Data loading utilities
│   ├── cnn_data_loader.py                # 🔧 CNN data processing
│   ├── features.py                       # ⚙️ Light curve processing
│   ├── pixel_diff.py                     # 🖼️ Pixel processing
│   └── evaluate.py                       # 📈 Performance metrics
├── scripts/                              # Data generation pipeline
│   ├── rapid_cnn_expansion.py           # 🚀 CNN data expansion
│   ├── standardize_cnn_data.py          # 📏 Data standardization
│   └── enrich_koi.py                    # 🔗 NASA Archive integration
├── data/                                 # Data directory
│   ├── raw/                             # Original datasets
│   └── processed/                       # CNN training data
├── models/                              # Trained models
│   └── enhanced_multimodal_fusion_model.pth  # 🏆 93.47% accuracy model
└── notebooks/                           # Analysis notebooks
    └── demo_pipeline.ipynb              # 📓 Interactive demo
```

## 🔬 Technical Details

### Multimodal Fusion Strategy
The pipeline combines three complementary data modalities:

1. **Tabular Features**: Capture orbital mechanics and stellar properties
2. **1D CNN (Residuals)**: Detect transit signatures in time series
3. **2D CNN (Pixels)**: Identify spatial transit patterns in target pixel files

### Data Coverage Scaling
- **Confirmed Planets**: 60-80% success rate for CNN data generation
- **Short Periods (1-10d)**: 70-90% success rate
- **Deep Transits (>100ppm)**: 80-95% success rate
- **Current Coverage**: 2.5% of 9,777 targets (377 CNN samples)

### Performance Optimization
- **Early Stopping**: Prevents overfitting, stops at epoch 45
- **Threshold Optimization**: 0.9922 maximizes TPR while maintaining low FPR
- **Robust Data Loading**: Graceful fallbacks for missing CNN data
- **Efficient Training**: 54.6 seconds for 93.47% validation accuracy

## 🚀 Production Deployment

### API Integration
The model can be integrated into production systems:

```python
# Production inference function
def predict_exoplanet(kepid: str) -> dict:
    sample_data = prepare_single_sample(kepid)
    probability = model(sample_data).item()
    
    return {
        'kepid': kepid,
        'probability': probability,
        'prediction': 'CONFIRMED' if probability > 0.9922 else 'FALSE POSITIVE',
        'confidence': probability * 100
    }
```

### Batch Processing
```python
# Process multiple targets
def batch_predict(kepids: list) -> list:
    results = []
    for kepid in kepids:
        result = predict_exoplanet(kepid)
        results.append(result)
    return results
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📚 Citation

```bibtex
@software{enhanced_multimodal_exoplanet_detection,
  title={Enhanced Multimodal Exoplanet Detection Pipeline},
  author={Roshan Kattil},
  year={2025},
  url={https://github.com/RoshanKattil/exoplanet-detection-pipeline}
}
```

## 🙏 Acknowledgments

- **NASA Kepler Mission** for the light curve and target pixel file data
- **Lightkurve** for excellent astronomical data processing tools
- **PyTorch** for the deep learning framework
- **NASA Exoplanet Archive** for comprehensive stellar parameters

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🚀 Future Enhancements

- [ ] Scale CNN coverage to 5%+ (500+ samples)
- [ ] Add TESS mission support
- [ ] Implement ensemble methods
- [ ] Deploy as web API service
- [ ] Add model interpretability features
- [ ] Real-time prediction dashboard

---

**⭐ Star this repository if you find it useful!**

**🎯 Achievement: 93.47% Validation Accuracy on Real NASA Kepler Data**

For questions or support, please open an issue or contact the development team.