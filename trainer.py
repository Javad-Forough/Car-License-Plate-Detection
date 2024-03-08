import torch
import torch.nn as nn
import torch.optim as optim

class Trainer:
    def __init__(self, model, lr=0.001):
        """
        Trainer class for training the license plate detection model.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            lr (float, optional): Learning rate for the optimizer (default: 0.001).
        """
        self.model = model
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def train(self, train_loader, val_loader, epochs):
        """
        Trains the model on the training data.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
            epochs (int): Number of epochs for training.
        """
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for images, targets in train_loader:
                # Move data to the device
                images, targets = images.to(self.device), targets.to(self.device)
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.model(images)
                # Compute loss
                loss = self.criterion(outputs, targets)
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                # Print statistics
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

            # Validate the model
            self.validate(val_loader)

    def validate(self, val_loader):
        """
        Validates the model on the validation data.

        Args:
            val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    def evaluate(self, test_loader):
        """
        Evaluates the trained model on the test data.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
