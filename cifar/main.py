import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from lenet5 import Lenet5
from torch import nn, optim


def main():

    #load
    batchsz = 128
    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)
    cifar_test = datasets.CIFAR10('cifar', False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    #x, lable = iter(cifar_train).next()
    #print('x:', x.shape, 'lable:', lable.shape)



    device = torch.device('cuda:0')
    model = Lenet5().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    #train
    for epoch in range(1000):
        model.train()
        for batchsz, (x, lable) in enumerate(cifar_train):
            x, lable = x.to(device), lable.to(device)
            logits = model(x)
            #logits [b,10]
            #lable [b]
            #loss: tensor scalar
            loss = criteon(logits, lable)

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch, 'loss:', loss.item())

        model.eval()
        with torch.no_grad():
        #test
            total_correct = 0
            total_num = 0
            for x, lable in cifar_test:
                x, lable = x.to(device), lable.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                #
                total_correct += torch.eq(pred, lable).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch, 'test acc:', acc)

if __name__ == '__main__':
    main()