
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
import matplotlib.pyplot as plt
import torchsummary
import logging
import numpy as np
from tqdm import tqdm
import torchvision.datasets as datasets
import torch.optim as optim
from src.config import Config
from src.models.res_net_akamaster import *

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    config  = Config()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data/cifar10', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=512, shuffle=True,
        num_workers=5, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data/cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=5, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    print(images.shape)

    # show images
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    # imshow(torchvision.utils.make_grid(images))

    net = resnet20()
    try:
        net.load_state_dict(torch.load("net.pth"))
    except:
        print("Loading net from file failed!")

    # for parameter in net.parameters():
    #     parameter.requires_grad = False
    
    # for parameter in net.classification.parameters():
    #     parameter.requires_grad = True

    # net.reinit_classification()
    net = net.to(config.run_device)
    # print(net)
    # torchsummary.summary(net, (3, 32,32), device=str(config.run_device))
    # logger.debug(f"net on device: {config.run_device}")

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3,
                            weight_decay=1e-4)

    for epoch in range(100):  # loop over the dataset multiple times
        net.train(True)
        running_loss = 0.0
        predicted_correctly = 0
        predicted_total = 0
        pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader), leave=True)
        for i, data in pbar:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(config.run_device)
            labels = labels.to(config.run_device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            output_cat = torch.argmax(outputs.detach(), dim=1)
            correct = labels.eq(output_cat).detach()
            predicted_correctly += correct.sum().item()
            predicted_total += correct.shape[0]
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            pbar.set_description(f'[{epoch + 1}, {i + 1:03d}] loss: {running_loss/(i+1):.3f} | acc: {predicted_correctly/predicted_total:.3%}')
        torch.save(net.state_dict(), "net.pth")
    print('Finished Training')
    correct = 0
    total = 0
    net.train(False)
    with torch.no_grad():
        for data in tqdm(val_loader, total=len(val_loader)):
            images, labels = data
            images = images.to(config.run_device)
            labels = labels.to(config.run_device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Correct: {correct}/{total} or {correct/total:%}")
