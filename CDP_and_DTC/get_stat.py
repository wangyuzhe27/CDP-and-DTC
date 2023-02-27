import torch
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms


def getStat(train_set):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_set))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_set))
    std.div_(len(train_set))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':
    train_set = datasets.CIFAR10('../data', train=True, download=True)

    print(getStat(train_set))
