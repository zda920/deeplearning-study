import torch
from torch import nn
from torch.nn import functional as F

class Resblk(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Resblk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            #[b,ch_in,h,w] => [b,ch_out,h,w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=2),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        #shortcut
        #ch_in ch_out
        out = self.extra(x) + out
        out = F.relu(out)
        return out

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        #follow 4 blocks
        #[b, 64, h, w] =>[b, 128, h, w]
        self.blk1 = Resblk(64, 128)
        # [b, 128, h, w] =>[b, 256, h, w]
        self.blk2 = Resblk(128, 256)
        # [b, 256, h, w] =>[b, 512, h, w]
        self.blk3 = Resblk(256, 512)
        # [b, 512, h, w] =>[b, 1024, h, w]
        self.blk4 = Resblk(512, 512)
        self.outlayer = nn.Linear(512, 10)


    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))
       # print('1:', x.shape)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)

        x = self.outlayer(x)

        return x

def main():
    x = torch.randn(2, 3, 32, 32)
    model = Resnet18()
    out = model(x)
    print('resnet:', out.shape)
