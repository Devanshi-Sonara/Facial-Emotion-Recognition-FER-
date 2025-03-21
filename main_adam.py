import os
import cv2
import numpy as np
import torch
import matplotlib
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Step 1: Define Paths to Dataset
# Assuming main.py is in the DL folder
base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of main.py
train_dir = os.path.join(base_dir, 'data', 'train')    # Path to train folder
test_dir = os.path.join(base_dir, 'data', 'test')      # Path to test folder

# List of emotion labels (folder names)
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally with 50% probability
    transforms.RandomRotation(degrees=10),   # Randomly rotate by up to 10 degrees
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Random translation
    transforms.ToTensor(),                   # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values to [-1, 1]
])

# Step 2: Load Images and Labels
def load_images_from_folder(folder_path, augment=False):
    images = []
    labels = []
    for emotion_id, emotion in enumerate(emotions):
        emotion_folder = os.path.join(folder_path, emotion)
        for filename in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, (48, 48))                   # Resize to 48x48
            img = img / 255.0                                 # Normalize to [0, 1]
            
            # Apply augmentation if specified
            if augment:
                img = data_transforms(img)  # Apply augmentation
            else:
                img = transforms.ToTensor()(img)  # Convert to tensor without augmentation
            
            images.append(img)
            labels.append(emotion_id)
    return images, labels



# Load training and test data
# Load training data with augmentation
print("Loading training data...")
X_train, y_train = load_images_from_folder(train_dir, augment=True)

# Load test data without augmentation
print("Loading test data...")
X_test, y_test = load_images_from_folder(test_dir, augment=False)

# Step 3: Reshape and Convert to PyTorch Tensors
# Reshape images to include channel dimension (48x48x1)
X_train = X_train.reshape(-1, 1, 48, 48)
X_test = X_test.reshape(-1, 1, 48, 48)

# Convert to PyTorch tensors
# Stack lists into tensors
X_train = torch.stack(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.stack(X_test)
y_test = torch.tensor(y_test, dtype=torch.long)

# Step 4: Split Training Data into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Step 5: Create DataLoader for Training and Validation
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 6: Define the ANN Model
class ANN(nn.Module):
    # def __init__(self):
    #     super(ANN, self).__init__()
    #     self.flatten = nn.Flatten()
    #     self.fc1 = nn.Linear(48 * 48, 256)  # First hidden layer: 256 neurons
    #     self.fc2 = nn.Linear(256, 128)      # Second hidden layer: 128 neurons
    #     self.fc3 = nn.Linear(128, 64)       # Third hidden layer: 64 neurons
    #     self.fc4 = nn.Linear(64, 32)        # Fourth hidden layer: 32 neurons
    #     self.fc5 = nn.Linear(32, 7)                # Output size: 7 classes
    #     self.dropout1 = nn.Dropout(0.5)     # Dropout for first layer
    #     self.dropout2 = nn.Dropout(0.5)     # Dropout for second layer
    #     self.dropout3 = nn.Dropout(0.5)     # Dropout for third layer

    # def forward(self, x):
    #     x = self.flatten(x)
    #     x = torch.relu(self.fc1(x))
    #     x = self.dropout1(x)
    #     x = torch.relu(self.fc2(x))
    #     x = self.dropout2(x)
    #     x = torch.relu(self.fc3(x))
    #     x = self.dropout3(x)
    #     x = torch.relu(self.fc4(x))
    #     x = self.fc5(x)
    #     return x
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        
        # Input layer to first hidden layer
        self.fc1 = nn.Linear(48 * 48, 512)  # Increased neurons
        self.bn1 = nn.BatchNorm1d(512)      # Batch normalization
        self.dropout1 = nn.Dropout(0.5)     # Dropout
        
        # Second hidden layer
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        
        # Third hidden layer
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.5)
        
        # Fourth hidden layer
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout4 = nn.Dropout(0.5)
        
        # Fifth hidden layer
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.dropout5 = nn.Dropout(0.5)
        
        # Output layer
        self.fc6 = nn.Linear(32, 7)  # 7 classes

    def forward(self, x):
        x = self.flatten(x)
        
        # Layer 1
        x = torch.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = torch.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = torch.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # Layer 4
        x = torch.relu(self.fc4(x))
        x = self.bn4(x)
        x = self.dropout4(x)
        
        # Layer 5
        x = torch.relu(self.fc5(x))
        x = self.bn5(x)
        x = self.dropout5(x)
        
        # Output layer
        x = self.fc6(x)
        return x

# Initialize the model, loss function, and optimizer
model = ANN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Step 7: Train the Model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
# Early stopping parameters
    patience = 10
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        #learing rate scheculding
        scheduler.step(val_loss)
        
        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return train_losses, val_losses, train_accuracies, val_accuracies

print("Training the model...")
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, criterion, optimizer, epochs=50)

# Step 8: Evaluate the Model on Test Data
model.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct / total


#step visualization:-
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt

# Plot training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')  # Save the plot as an image file

# Plot training and validation accuracy
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')  # Save the plot as an image file