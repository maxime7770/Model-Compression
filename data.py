from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=128):
    """CIFAR-10 train, test, and calibration data"""
    t_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
    ])
    t_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
    ])
    train = datasets.CIFAR10('data', train=True, download=True, transform=t_train)
    test  = datasets.CIFAR10('data', train=False, download=True, transform=t_test)
    # calibration: small subset of test
    calib = DataLoader(test, batch_size=batch_size, shuffle=False)
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=batch_size), calib
