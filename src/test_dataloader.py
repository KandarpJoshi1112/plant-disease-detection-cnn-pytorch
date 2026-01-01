from dataset import get_dataloaders

if __name__ == '__main__':
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    images, labels = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Number of classes:", len(class_names))
    print("Classes:", class_names)
