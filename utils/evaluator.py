# utils/evaluator.py

import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, test_loader, model_name, optimizer_name, epochs):
    """
    Evaluate the model on the test set and generate visualizations.
    
    Args:
        model: Trained model to evaluate.
        test_loader: DataLoader for the test set.
        model_name: Name of the model (e.g., "ANN", "CNN").
        optimizer_name: Name of the optimizer (e.g., "Adam", "SGD").
        epochs: Number of epochs used for training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]))

    # Confusion Matrix
    # cm = confusion_matrix(y_true, y_pred)
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
    #             xticklabels=["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    #             yticklabels=["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"])
    # plt.title(f"Confusion Matrix\nModel: {model_name}, Optimizer: {optimizer_name}, Epochs: {epochs}")
    # plt.xlabel("Predicted")
    # plt.ylabel("Actual")
    # plt.savefig(f"confusion_matrix_{model_name}_{optimizer_name}_{epochs}.png")
    # plt.close()

    # Accuracy Plot (if training accuracy is available)
    if hasattr(model, 'train_accuracies'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.train_accuracies, label="Training Accuracy")
        plt.plot(model.val_accuracies, label="Validation Accuracy")
        plt.title(f"Training and Validation Accuracy\nModel: {model_name}, Optimizer: {optimizer_name}, Epochs: {epochs}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(f"accuracy_plot_{model_name}_{optimizer_name}_{epochs}.png")
        plt.close()

    # Loss Plot (if training loss is available)
    if hasattr(model, 'train_losses'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.train_losses, label="Training Loss")
        plt.plot(model.val_losses, label="Validation Loss")
        plt.title(f"Training and Validation Loss\nModel: {model_name}, Optimizer: {optimizer_name}, Epochs: {epochs}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"loss_plot_{model_name}_{optimizer_name}_{epochs}.png")
        plt.close()

    # Print Test Accuracy
    test_accuracy = 100 * np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    print(f"Test Accuracy: {test_accuracy:.2f}%")