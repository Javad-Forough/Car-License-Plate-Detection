import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import xml.etree.ElementTree as ET

class LicensePlateDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom dataset class for loading images and corresponding annotations for license plate detection.

        Args:
            root_dir (str): Path to the root directory containing 'annotations' and 'images' subdirectories.
            transform (callable, optional): Optional transform to be applied to the image and annotations.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.annotations_dir = os.path.join(root_dir, 'annotations')
        self.images_dir = os.path.join(root_dir, 'images')
        # List of XML annotation files
        self.annotations = [ann for ann in os.listdir(self.annotations_dir) if ann.endswith('.xml')]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Retrieves the image and corresponding annotations for the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing the image and target annotations.
        """
        # Read XML annotation file
        xml_file = os.path.join(self.annotations_dir, self.annotations[idx])
        # Get corresponding image filename
        image_name = self.annotations[idx].replace('.xml', '.png')
        img_path = os.path.join(self.images_dir, image_name)
        # Open image
        image = Image.open(img_path).convert("RGB")

        # Parse XML annotation
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Extract bounding box coordinates and labels from the annotation
        boxes = []
        labels = []
        for obj in root.findall('object'):
            box = obj.find('bndbox')
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # Assuming all license plates are labeled as 1

        # Construct target dictionary containing bounding box coordinates and labels
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        # Apply transformations if specified
        if self.transform:
            image, target = self.transform(image, target)

        return image, target

class DataLoader:
    def __init__(self, batch_size):
        """
        DataLoader class for loading and splitting the dataset into train, validation, and test sets.

        Args:
            batch_size (int): Batch size for DataLoader.
        """
        self.batch_size = batch_size

    def load_data(self):
        """
        Loads the dataset, applies transformations, and splits it into train, validation, and test sets.

        Returns:
            tuple: Tuple containing train, validation, and test DataLoader objects.
        """
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Create dataset instance
        dataset = LicensePlateDataset('Dataset/', transform=transform)

        # Split dataset into train, validation, and test sets
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        # Create DataLoader objects for train, validation, and test sets
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size)
        test_loader = DataLoader(test_set, batch_size=self.batch_size)

        return train_loader, val_loader, test_loader
