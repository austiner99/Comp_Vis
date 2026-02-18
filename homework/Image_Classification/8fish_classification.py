#Task 1:
'''
Task 1: 50%
• Download the 8FishSpecies dataset from Learning Suite for this assignment.
• Use the training set to train your algorithm. Augment the data as needed to get the best result.
• Use the test set to test your algorithm.
• Organize your result and use the get_conf_mat() and vis_conf_mat() functions included in the LBPSVM sample code to generate
and output a png image of the confusion matrix with proper labels.
• Examine your result and discuss what might have been the reasons for misclassification.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import Counter

def train_and_evaluate(dataset_path, output_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    #load datasets
    train_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transform)
    
    test_dataset = datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    # load model
    
    class_counts = Counter(train_dataset.targets)
    print(f"Class counts: {class_counts}")
    
    total_samples = sum(class_counts.values())
    weights = [total_samples / class_counts[i] for i in range(num_classes)]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    # for param in model.parameters():
    #     param.requires_grad = False  # Freeze all layers except the final fully connected layer
    # for param in model.layer4.parameters():
    #     param.requires_grad = True  # Unfreeze the last block (layer4)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},  # Lower learning rate for the last block
        {'params': model.fc.parameters(), 'lr': 5e-4}      # Higher learning rate for the final layer
    ], weight_decay=1e-4)
    
    #training (25 epochs or until convergence)
    epochs = 30
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        if epoch_loss < 0.2:  # Early stopping condition
            print("Early stopping as loss is below 0.25")
            break
        
    # testing
    model.eval()
    all_preds  = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, train_dataset.classes, rotation=45)
    plt.yticks(tick_marks, train_dataset.classes)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"Confusion_matrix_{output_name}.png")
    plt.show()
    print(f"Confusion matrix saved as Confusion_matrix_{output_name}.png")
    
if __name__ == "__main__":
    dataset_path = 'OysterShell'  # Update this path to your dataset location
    output_name = 'OysterShell_ResNet18_v5'
    train_and_evaluate(dataset_path, output_name)