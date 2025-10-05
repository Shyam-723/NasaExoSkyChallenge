# 📂 Repository File Structure Guide

## 🔑 Essential Files for Inference

### 🎯 **Ready-to-Use Inference**
- `INFERENCE_GUIDE.md` - **START HERE** - Complete inference guide
- `inference_template.py` - Drop-in prediction function 
- `demo_inference.py` - Interactive inference demo
- `inference_data_requirements.py` - Detailed data requirements

### 🧠 **Trained Models**
- `models/enhanced_multimodal_fusion_model.pth` - Complete multimodal model
- `models/tabular_model.pth` - Standalone tabular model (recommended)
- `models/multimodal_fusion_model.pth` - Original multimodal model

### ⚡ **Quick Analysis Scripts**
- `analyze_performance_gap.py` - Why tabular outperforms multimodal
- `analyze_tabular_standalone.py` - Tabular model analysis  
- `check_data_overlap.py` - Validates no data leakage

## 🏗️ **Core Architecture**

### 📊 **Model Definitions**
- `src/models.py` - All model architectures (TabularNet, CNNs, Fusion)
- `train_multimodal_enhanced.py` - Enhanced training pipeline
- `train_multimodal.py` - Original multimodal training

### 📈 **Data Processing**
- `src/data_loader.py` - Tabular data loading and preprocessing
- `src/cnn_data_loader.py` - CNN data loading and batch processing
- `src/features.py` - Lightcurve feature extraction
- `src/pixel_diff.py` - Pixel-level analysis functions

### 🔧 **Utilities**
- `src/evaluate.py` - Model evaluation metrics
- `src/threshold_optimization.py` - Classification threshold tuning
- `src/train.py` - Training utilities

## 📊 **Data Structure**

### 📁 **Raw Data**
```
data/raw/
├── lighkurve_KOI_dataset_enriched.csv    # Main dataset (9,777 targets)
├── lighkurve_KOI_dataset.csv             # Original KOI data
├── all_global_synthetic.csv              # Synthetic global features
├── all_local_synthetic.csv               # Synthetic local features
└── *.csv                                 # Additional catalogs (TOI, TrES)
```

### 🔄 **Processed Data**
```
data/processed/
├── residual_windows/                     # 1D CNN lightcurve data (243 files)
├── pixel_diffs/                         # 2D CNN pixel data (134 files) 
├── residual_windows_std/                # Standardized 1D data
└── pixel_diffs_std/                     # Standardized 2D data
```

## 🚀 **Training Scripts**

### 🎯 **Main Training**
- `train_multimodal_enhanced.py` - **RECOMMENDED** - Enhanced multimodal training
- `train_multimodal.py` - Original multimodal approach

### 📊 **Testing & Validation**
- `test_fusion_final.py` - Final fusion model testing
- `test_pipeline.py` - Complete pipeline testing
- `test_components.py` - Individual component testing

## 🛠️ **Data Generation Scripts**

### 📈 **CNN Data Expansion**
```
scripts/
├── rapid_cnn_expansion.py               # Fast CNN data generation
├── reliable_cnn_booster.py             # Robust CNN expansion  
├── optimized_cnn_expansion.py          # Memory-optimized generation
├── generate_cnn_data_batch.py          # Batch processing
└── standardize_cnn_data.py             # Data standardization
```

### 🌟 **Data Enhancement**
- `scripts/enrich_koi.py` - Enhance KOI dataset with stellar parameters

## 📓 **Notebooks & Demos**

### 🎮 **Interactive Analysis**
- `notebooks/demo_pipeline.ipynb` - Complete pipeline demonstration
- `demo_multimodal.py` - Multimodal inference demo
- `demo_scaling_strategy.py` - Scaling strategy demonstration

## 📋 **Documentation**

### 📖 **Usage Guides**
- `INFERENCE_GUIDE.md` - **Main inference guide**
- `TRAINING_GUIDE.md` - Training instructions
- `README.md` - Repository overview
- `README_SUPERCHARGED.md` - Enhanced features guide

### 🔬 **Technical Analysis**
- `MULTI_MODAL_RESULTS.md` - Multimodal performance analysis
- `TECHNICAL_BREAKDOWN.md` - Detailed technical breakdown
- `SCALING_IMPLEMENTATION.md` - Scaling strategies
- `CNN_COVERAGE_STRATEGY.md` - CNN data expansion guide

### 📦 **Release Notes**
- `GITHUB_RELEASE_SUMMARY.md` - Release summary
- `README_ROSHAN_REPO.md` - Repository documentation

## ⚙️ **Configuration**
- `requirements.txt` - Python dependencies
- `prompt.txt` - Model prompt configuration

## 🎯 **Quick Start Priority**

### 🥇 **First Priority (Inference)**
1. `INFERENCE_GUIDE.md` - Start here
2. `inference_template.py` - Copy this for your code
3. `models/tabular_model.pth` - Use this model (93.6% accuracy)

### 🥈 **Second Priority (Understanding)**  
1. `analyze_performance_gap.py` - Why tabular wins
2. `src/models.py` - Model architectures
3. `check_data_overlap.py` - Data validation

### 🥉 **Third Priority (Advanced)**
1. `train_multimodal_enhanced.py` - Retraining
2. `inference_data_requirements.py` - Full data specs
3. `demo_multimodal.py` - Advanced features

---

**🎯 TL;DR**: Use `INFERENCE_GUIDE.md` + `inference_template.py` + `models/tabular_model.pth` for immediate 93.6% accuracy exoplanet detection!