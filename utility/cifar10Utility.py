
from torchvision import datasets
from torch.utils.data import DataLoader 

class CustomCIFAR10(datasets.CIFAR10):

    def apply_augmentations(self, image):
        augmented_image = self.transform(image=image)  # Pass 'image' as named argument
        return augmented_image

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def get_datasets(train_transforms_collection, test_transforms_collection, data_folder) -> tuple[datasets.CIFAR10, datasets.CIFAR10]:

    train_dataset = CustomCIFAR10( root=data_folder, train=True, download=True,
                                transform=train_transforms_collection)
    
    test_dataset = CustomCIFAR10( root=data_folder, train=False, download=True,
                                transform=test_transforms_collection)
    
    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset, batch_size = 128, shuffle=True, num_workers=4, pin_memory=True) -> tuple[DataLoader, DataLoader]:
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    for batch_data, label in test_dataloader:
    # (e.g., shape: (batch_size, 1 channel, 28, 28)). (batch_size, channels, height, width)
    # y would contain the corresponding labels for each image, indicating the actual digit represented in the image 
        print(f"Shape of test_dataloader batch_data [Batch, C, H, W]: {batch_data.shape}")
        print(f"Shape of test_dataloader label (label): {label.shape} {label.dtype}")
        # print(f"Labels for a batch of size {batch_size} are {label}")
        break

    return train_dataloader, test_dataloader


def get_mean():
    return (0.4914, 0.4822, 0.4465)

def get_std():
    return (0.2470, 0.2435, 0.2616)

def get_image_classes():
    return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')