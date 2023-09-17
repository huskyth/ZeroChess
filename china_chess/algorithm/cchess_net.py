import torch
import torch.nn as nn
import torch.nn.functional as F
from china_chess.constant import *
from utils import dotdict

args = dotdict({
    'lr': 0.001,
    'dropout': 0.5,
    'epochs': 20,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 512,
})


class CChessNNet(nn.Module):

    def _resnet(self, inp, cov1, bn1, cov2, bn2):
        return bn2(cov2(self.relu(bn1(cov1(inp))))) + inp

    def __init__(self, args):
        # game params
        self.board_x, self.board_y = 10, 9
        self.action_size = len(LABELS)
        self.args = args

        super(CChessNNet, self).__init__()

        self.conv1 = nn.Conv2d(1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = F.relu

        self.conv21 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.conv22 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn22 = nn.BatchNorm2d(32)

        self.conv31 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(32)
        self.conv32 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn32 = nn.BatchNorm2d(32)

        self.conv41 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn41 = nn.BatchNorm2d(32)
        self.conv42 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn42 = nn.BatchNorm2d(32)

        self.conv51 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn51 = nn.BatchNorm2d(32)
        self.conv52 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn52 = nn.BatchNorm2d(32)

        self.conv61 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn61 = nn.BatchNorm2d(32)
        self.conv62 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn62 = nn.BatchNorm2d(32)

        self.conv71 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn71 = nn.BatchNorm2d(32)
        self.conv72 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bn72 = nn.BatchNorm2d(32)

        self.conv8_left = nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)
        self.bn8_left = nn.BatchNorm2d(4)
        self.fc81_left = nn.Linear(360, 256)
        self.fc82_left = nn.Linear(256, 1)

        self.conv8_right = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn8_right = nn.BatchNorm2d(32)
        self.fc81_right = nn.Linear(2880, 2086)

    def forward(self, x):
        x = x.view(-1, 1, self.board_x, self.board_y)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self._resnet(x, self.conv21, self.bn21, self.conv22, self.bn22))
        # x = self.relu(self._resnet(x, self.conv31, self.bn31, self.conv32, self.bn32))
        # x = self.relu(self._resnet(x, self.conv41, self.bn41, self.conv42, self.bn42))
        # x = self.relu(self._resnet(x, self.conv51, self.bn51, self.conv52, self.bn52))
        # x = self.relu(self._resnet(x, self.conv61, self.bn61, self.conv62, self.bn62))
        # x = self.relu(self._resnet(x, self.conv71, self.bn71, self.conv72, self.bn72))

        x_left = torch.flatten(self.relu(self.bn8_left(self.conv8_left(x))))
        x_left = x_left.view(-1, 360)
        x_left = self.fc82_left(self.fc81_left(x_left))

        x_right = torch.flatten(self.relu(self.conv8_right(x)))
        x_right = x_right.view(-1, 2880)
        x_right = self.fc81_right(x_right)

        return F.log_softmax(x_right, dim=1), torch.tanh(x_left)


if __name__ == '__main__':
    cnet = CChessNNet(args)
    input = torch.rand(9, 10, 9)
    y = cnet(input)
    print(y[0].shape, y[1].shape)
