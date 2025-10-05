"""
🌟 Multi-Modal Exoplanet Detection Demo
=======================================

This script demonstrates the complete pipeline from data loading to prediction.
"""
import torch
import numpy as np
from src.data_loader import load_and_prepare_data, create_train_val_test_splits
from src.cnn_data_loader import load_cnn_data, create_cnn_datasets
from train_multimodal import MultiModalFusionModel

def demo_pipeline():
    """Demo the complete multi-modal pipeline"""
    
    print("🌟 MULTI-MODAL EXOPLANET DETECTION DEMO")
    print("=" * 50)
    
    # Step 1: Load and prepare data
    print("\n📊 Step 1: Loading Multi-Modal Data")
    print("-" * 40)
    
    data = load_and_prepare_data()
    splits = create_train_val_test_splits(data)
    feature_names = splits['feature_names']
    
    print(f"✅ Loaded {len(feature_names)} tabular features")
    print(f"✅ Total samples: {len(splits['train']['y']) + len(splits['val']['y']) + len(splits['test']['y'])}")
    
    # Load CNN data
    cnn_data = load_cnn_data()
    cnn_datasets = create_cnn_datasets(cnn_data, splits, feature_names)
    
    print(f"✅ CNN data: {len(cnn_data['residual_windows'])} residual + {len(cnn_data['pixel_diffs'])} pixel")
    
    # Step 2: Load trained model
    print("\n🧠 Step 2: Loading Trained Fusion Model")
    print("-" * 45)
    
    model = MultiModalFusionModel(n_tabular_features=len(feature_names))
    model.load_state_dict(torch.load("models/multimodal_fusion_model.pth"))
    model.eval()
    
    print("✅ Multi-modal fusion model loaded")
    print(f"   📊 Tabular features: {len(feature_names)}")
    print(f"   📡 CNN1D input: 128 time points")
    print(f"   🖼️  CNN2D input: 32×24×24 pixels")
    
    # Step 3: Make predictions on test data
    print("\n🎯 Step 3: Making Predictions")
    print("-" * 35)
    
    # Get test data
    min_test_samples = min(len(splits['test']['y']), len(cnn_datasets['test']['y']))
    
    test_tabular = torch.FloatTensor(splits['test']['X'][:min_test_samples])
    test_cnn1d = torch.FloatTensor(cnn_datasets['test']['cnn1d'][:min_test_samples])
    test_cnn2d = torch.FloatTensor(cnn_datasets['test']['cnn2d'][:min_test_samples])
    test_labels = splits['test']['y'][:min_test_samples]
    
    with torch.no_grad():
        predictions = model(test_tabular, test_cnn1d, test_cnn2d).numpy().flatten()
    
    print(f"✅ Generated predictions for {len(predictions)} test samples")
    
    # Step 4: Show some results
    print("\n🎲 Step 4: Sample Predictions")
    print("-" * 35)
    
    confirmed_count = 0
    false_pos_count = 0
    
    for i in range(min(20, len(test_labels))):
        true_label = test_labels[i]
        pred_prob = predictions[i]
        
        if true_label == 1:
            true_class = "CONFIRMED"
            confirmed_count += 1
        else:
            true_class = "FALSE_POS"
            false_pos_count += 1
            
        pred_class = "CONFIRMED" if pred_prob > 0.5 else "FALSE_POS"
        confidence = pred_prob if pred_prob > 0.5 else (1 - pred_prob)
        
        # Correct prediction?
        correct = "✅" if (true_label == 1 and pred_prob > 0.5) or (true_label == 0 and pred_prob <= 0.5) else "❌"
        
        print(f"Sample {i+1:2d}: {true_class:10} → {pred_class:10} ({confidence:.1%} conf) {correct}")
    
    # Step 5: Summary statistics
    print(f"\n📊 Step 5: Performance Summary")
    print("-" * 40)
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    auc = roc_auc_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, (predictions > 0.5).astype(int))
    
    print(f"🏆 Test AUC: {auc:.1%}")
    print(f"🎯 Test Accuracy: {accuracy:.1%}")
    print(f"📈 Confirmed planets in test: {confirmed_count}")
    print(f"📉 False positives in test: {false_pos_count}")
    
    # Step 6: Feature importance (simplified)
    print(f"\n🔍 Step 6: Feature Types")
    print("-" * 30)
    
    print("📊 Tabular Features (top 10):")
    important_features = [
        'period', 'depth', 'duration', 'star_temp', 'star_radius',
        'duty_cycle', 'log_period', 'period_err_rel', 'impact', 'star_mass'
    ]
    for i, feat in enumerate(important_features):
        if feat in feature_names:
            print(f"   {i+1:2d}. {feat}")
    
    print("\n📡 CNN1D Features:")
    print("   • Residual lightcurve windows")
    print("   • Detrended time series patterns")
    print("   • Transit signature extraction")
    
    print("\n🖼️  CNN2D Features:")
    print("   • Target pixel differences")
    print("   • Spatial transit patterns")
    print("   • Centroid motion detection")
    
    print(f"\n🎊 DEMO COMPLETE!")
    print("=" * 20)
    print("Successfully demonstrated multi-modal exoplanet detection!")
    print(f"Final model achieves {auc:.1%} AUC with real Kepler data.")

if __name__ == "__main__":
    demo_pipeline()