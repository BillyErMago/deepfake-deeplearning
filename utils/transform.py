from torchvision import transforms

def get_transforms(pretrained=False):
    # Normalizzazione per backbones (ImageNet) applicata sia ai custom che ai pretrained
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Trasformazioni per il training
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Trasformazioni per il test
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return transform_train, transform_test