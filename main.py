from data_loader import DataLoader
from model import LicensePlateDetector
from trainer import Trainer

# Set hyperparameters
batch_size = 32
epochs = 20

# Initialize data loader
data_loader = DataLoader(batch_size=batch_size)

# Load dataset
train_loader, val_loader, test_loader = data_loader.load_data()

# Initialize model
model = LicensePlateDetector()

# Initialize trainer
trainer = Trainer(model)

# Train the model
trainer.train(train_loader, val_loader, epochs)

# Evaluate the model
trainer.evaluate(test_loader)
