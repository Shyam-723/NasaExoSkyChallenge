"""
Test individual model components vs. fusion model
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader

from src.data_loader import load_and_prepare_data, create_train_val_test_splits
from src.cnn_data_loader import load_cnn_data, create_cnn_datasets
from src.models import TabularNet, ResidualCNN1D, PixelCNN2D
from train_multimodal import MultiModalDataset, MultiModalFusionModel

def test_individual_models():
    """Test each model component individually"""
    
    print("🧪 Testing Individual Model Components")
    print("=" * 50)
    
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
    
    # 1. Test Tabular-only model
    print("\n🏁 Tabular Model Only")
    tabular_model = TabularNet(input_size=len(feature_names))
    
    # Train a simple tabular model
    from torch.optim import Adam
    from torch.nn import BCELoss
    
    optimizer = Adam(tabular_model.parameters(), lr=1e-3)
    criterion = BCELoss()
    
    # Simple training loop
    train_tabular = torch.FloatTensor(splits['train']['X'][:min_test_samples])
    train_labels_tab = torch.FloatTensor(splits['train']['y'][:min_test_samples])
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = tabular_model(train_tabular)
        loss = criterion(outputs.squeeze(), train_labels_tab)
        loss.backward()
        optimizer.step()
    
    # Test tabular model
    with torch.no_grad():
        tabular_preds = tabular_model(test_tabular).numpy().flatten()
    
    tabular_auc = roc_auc_score(test_labels, tabular_preds)
    tabular_acc = accuracy_score(test_labels, (tabular_preds > 0.5).astype(int))
    
    print(f"   AUC: {tabular_auc:.4f}")
    print(f"   Accuracy: {tabular_acc:.4f}")
    
    # 2. Test CNN1D-only model
    print("\n📡 CNN1D Model Only")
    cnn1d_model = ResidualCNN1D()
    
    # Add a classifier head for CNN1D
    cnn1d_classifier = torch.nn.Sequential(
        cnn1d_model,
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    
    optimizer = Adam(cnn1d_classifier.parameters(), lr=1e-3)
    
    # Train CNN1D
    train_cnn1d = torch.FloatTensor(cnn_datasets['train']['cnn1d'][:min_test_samples])
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = cnn1d_classifier(train_cnn1d)
        loss = criterion(outputs.squeeze(), train_labels_tab)
        loss.backward()
        optimizer.step()
    
    # Test CNN1D
    with torch.no_grad():
        cnn1d_preds = cnn1d_classifier(test_cnn1d).numpy().flatten()
    
    cnn1d_auc = roc_auc_score(test_labels, cnn1d_preds)
    cnn1d_acc = accuracy_score(test_labels, (cnn1d_preds > 0.5).astype(int))
    
    print(f"   AUC: {cnn1d_auc:.4f}")
    print(f"   Accuracy: {cnn1d_acc:.4f}")
    
    # 3. Test CNN2D-only model
    print("\n🖼️  CNN2D Model Only")
    cnn2d_model = PixelCNN2D()
    
    # Add a classifier head for CNN2D
    cnn2d_classifier = torch.nn.Sequential(
        cnn2d_model,
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )
    
    optimizer = Adam(cnn2d_classifier.parameters(), lr=1e-3)
    
    # Train CNN2D
    train_cnn2d = torch.FloatTensor(cnn_datasets['train']['cnn2d'][:min_test_samples])
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = cnn2d_classifier(train_cnn2d)
        loss = criterion(outputs.squeeze(), train_labels_tab)
        loss.backward()
        optimizer.step()
    
    # Test CNN2D
    with torch.no_grad():
        cnn2d_preds = cnn2d_classifier(test_cnn2d).numpy().flatten()
    
    cnn2d_auc = roc_auc_score(test_labels, cnn2d_preds)
    cnn2d_acc = accuracy_score(test_labels, (cnn2d_preds > 0.5).astype(int))
    
    print(f"   AUC: {cnn2d_auc:.4f}")
    print(f"   Accuracy: {cnn2d_acc:.4f}")
    
    # 4. Load and test the fusion model
    print("\n🔗 Multi-Modal Fusion Model")
    fusion_model = MultiModalFusionModel(n_tabular_features=len(feature_names))
    fusion_model.load_state_dict(torch.load("models/multimodal_fusion_model.pth"))
    fusion_model.eval()
    
    with torch.no_grad():
        fusion_preds = fusion_model(test_tabular, test_cnn1d, test_cnn2d).numpy().flatten()
    
    fusion_auc = roc_auc_score(test_labels, fusion_preds)
    fusion_acc = accuracy_score(test_labels, (fusion_preds > 0.5).astype(int))
    
    print(f"   AUC: {fusion_auc:.4f}")
    print(f"   Accuracy: {fusion_acc:.4f}")
    
    # Summary
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 40)
    print(f"Tabular Only:    AUC={tabular_auc:.4f}, Acc={tabular_acc:.4f}")
    print(f"CNN1D Only:      AUC={cnn1d_auc:.4f}, Acc={cnn1d_acc:.4f}")
    print(f"CNN2D Only:      AUC={cnn2d_auc:.4f}, Acc={cnn2d_acc:.4f}")
    print(f"Multi-Modal:     AUC={fusion_auc:.4f}, Acc={fusion_acc:.4f}")
    
    print(f"\n🚀 Multi-modal improvement over best single model:")
    best_single_auc = max(tabular_auc, cnn1d_auc, cnn2d_auc)
    improvement = ((fusion_auc - best_single_auc) / best_single_auc) * 100
    print(f"   AUC improvement: +{improvement:.2f}%")

if __name__ == "__main__":
    test_individual_models()