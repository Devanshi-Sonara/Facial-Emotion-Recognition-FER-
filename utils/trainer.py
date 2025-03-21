# utils/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import *

def train_model(model, train_loader, test_loader, optimizer="adam"):
    """
    Train the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer == "adam":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # L2 regularization
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    else:
        raise ValueError("Optimizer not supported.")

    # Early stopping parameters
    patience = 5  # Number of epochs to wait before stopping
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}")

        # Early stopping logic
        if running_loss < best_loss:
            best_loss = running_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_MODEL_PATH + "best_model.pth")  # Save best model
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch+1}")
                break