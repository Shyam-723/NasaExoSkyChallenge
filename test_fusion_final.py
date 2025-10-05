"""
Simple test of the multi-modal fusion model performance
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

from src.data_loader import load_and_prepare_data, create_train_val_test_splits
from src.cnn_data_loader import load_cnn_data, create_cnn_datasets
from train_multimodal import MultiModalFusionModel

def test_fusion_model():
    """Test the trained fusion model"""
    
    print("🌟 Multi-Modal Fusion Model Performance Test")
    print("=" * 55)
    
    # Load data
    data = load_and_prepare_data()
    splits = create_train_val_test_splits(data)
    feature_names = splits['feature_names']
    
    cnn_data = load_cnn_data()
    cnn_datasets = create_cnn_datasets(cnn_data, splits, feature_names)
    
    # Align dataset sizes
    min_test_samples = min(len(splits['test']['y']), len(cnn_datasets['test']['y']))
    
    # Test data
    test_tabular = torch.FloatTensor(splits['test']['X'][:min_test_samples])
    test_cnn1d = torch.FloatTensor(cnn_datasets['test']['cnn1d'][:min_test_samples])
    test_cnn2d = torch.FloatTensor(cnn_datasets['test']['cnn2d'][:min_test_samples])
    test_labels = splits['test']['y'][:min_test_samples]
    
    print(f"📊 Test samples: {min_test_samples}")
    print(f"📊 Features: {len(feature_names)} tabular + CNN data")
    
    # Load trained fusion model
    fusion_model = MultiModalFusionModel(n_tabular_features=len(feature_names))
    fusion_model.load_state_dict(torch.load("models/multimodal_fusion_model.pth"))
    fusion_model.eval()
    
    # Get predictions
    with torch.no_grad():
        fusion_preds = fusion_model(test_tabular, test_cnn1d, test_cnn2d).numpy().flatten()
        
        # Also get individual model outputs for analysis
        tabular_features = fusion_model.tabular_model(test_tabular).numpy().flatten()
        cnn1d_features = fusion_model.cnn1d_model(test_cnn1d).numpy()
        cnn2d_features = fusion_model.cnn2d_model(test_cnn2d).numpy()
    
    # Calculate metrics
    fusion_auc = roc_auc_score(test_labels, fusion_preds)
    fusion_acc = accuracy_score(test_labels, (fusion_preds > 0.5).astype(int))
    
    print(f"\n🎯 FINAL FUSION MODEL RESULTS")
    print("=" * 35)
    print(f"🏆 Test AUC: {fusion_auc:.4f}")
    print(f"🎯 Test Accuracy: {fusion_acc:.4f}")
    
    # Detailed classification report
    print(f"\n📊 DETAILED CLASSIFICATION METRICS")
    print("=" * 40)
    y_pred_binary = (fusion_preds > 0.5).astype(int)
    print(classification_report(test_labels, y_pred_binary, 
                              target_names=['False Positive', 'Confirmed'], 
                              digits=4))
    
    # Feature contribution analysis
    print(f"\n🔍 MODEL COMPONENT ANALYSIS")
    print("=" * 35)
    print(f"📈 Tabular features range: [{tabular_features.min():.4f}, {tabular_features.max():.4f}]")
    print(f"📡 CNN1D features shape: {cnn1d_features.shape}")
    print(f"🖼️  CNN2D features shape: {cnn2d_features.shape}")
    print(f"🔗 Final predictions range: [{fusion_preds.min():.4f}, {fusion_preds.max():.4f}]")
    
    # Show some prediction examples
    print(f"\n🎲 PREDICTION EXAMPLES")
    print("=" * 25)
    for i in range(min(10, len(test_labels))):
        true_label = "CONFIRMED" if test_labels[i] == 1 else "FALSE_POS"
        pred_prob = fusion_preds[i]
        pred_label = "CONFIRMED" if pred_prob > 0.5 else "FALSE_POS"
        confidence = "✅" if (test_labels[i] == 1 and pred_prob > 0.5) or (test_labels[i] == 0 and pred_prob <= 0.5) else "❌"
        print(f"Sample {i+1:2d}: {true_label:10} → {pred_prob:.4f} ({pred_label:10}) {confidence}")
    
    print(f"\n🚀 MULTI-MODAL PIPELINE SUMMARY")
    print("=" * 40)
    print(f"✅ Successfully combined 3 data modalities:")
    print(f"   📊 Tabular features: 39 engineered features")
    print(f"   📡 1D CNN: Residual lightcurve windows") 
    print(f"   🖼️  2D CNN: Target pixel differences")
    print(f"✅ Achieved {fusion_auc:.1%} AUC on test set")
    print(f"✅ Model saved to: models/multimodal_fusion_model.pth")

if __name__ == "__main__":
    test_fusion_model()