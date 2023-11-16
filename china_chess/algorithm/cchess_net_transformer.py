import torch
import torch.nn as nn
import torch.nn.functional as F

from china_chess.algorithm.common.transformer import TransformerBlock
from china_chess.constant import *


class CChessNNetWithTransformer(nn.Module):

    def _resnet(self, inp, cov1, bn1, cov2, bn2):
        return bn2(cov2(self.relu(bn1(cov1(inp))))) + inp

    def __init__(self, args):
        # game params
        self.board_x, self.board_y = 10, 9
        self.action_size = len(LABELS)
        self.args = args
        self.channel = 128

        super(CChessNNetWithTransformer, self).__init__()

        self.conv1 = nn.Conv2d(14, out_channels=self.channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.channel)
        self.relu = F.relu

        self.conv21 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn21 = nn.BatchNorm2d(self.channel)
        self.conv22 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn22 = nn.BatchNorm2d(self.channel)

        self.conv31 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn31 = nn.BatchNorm2d(self.channel)
        self.conv32 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn32 = nn.BatchNorm2d(self.channel)

        self.conv41 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn41 = nn.BatchNorm2d(self.channel)
        self.conv42 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn42 = nn.BatchNorm2d(self.channel)

        self.conv51 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn51 = nn.BatchNorm2d(self.channel)
        self.conv52 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn52 = nn.BatchNorm2d(self.channel)

        self.conv61 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn61 = nn.BatchNorm2d(self.channel)
        self.conv62 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn62 = nn.BatchNorm2d(self.channel)

        self.conv71 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn71 = nn.BatchNorm2d(self.channel)
        self.conv72 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn72 = nn.BatchNorm2d(self.channel)

        self.conv81 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn81 = nn.BatchNorm2d(self.channel)
        self.conv82 = nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1)
        self.bn82 = nn.BatchNorm2d(self.channel)

        self.conv9_left = nn.Conv2d(self.channel, 1, kernel_size=1, stride=1, padding=0)
        self.bn9_left = nn.BatchNorm2d(1)

        self.fc91_left = nn.Linear(90, 256)
        self.fc92_left = nn.Linear(256, 1)

        self.conv9_right = nn.Conv2d(self.channel, 2, kernel_size=1, stride=1, padding=0)
        self.bn9_right = nn.BatchNorm2d(2)
        self.fc91_right = nn.Linear(180, 2086)

    def forward(self, x):
        x = x.view(-1, 14, self.board_x, self.board_y)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self._resnet(x, self.conv21, self.bn21, self.conv22, self.bn22))
        x = self.relu(self._resnet(x, self.conv31, self.bn31, self.conv32, self.bn32))
        x = self.relu(self._resnet(x, self.conv41, self.bn41, self.conv42, self.bn42))
        x = self.relu(self._resnet(x, self.conv51, self.bn51, self.conv52, self.bn52))
        x = self.relu(self._resnet(x, self.conv61, self.bn61, self.conv62, self.bn62))
        x = self.relu(self._resnet(x, self.conv71, self.bn71, self.conv72, self.bn72))
        x = self.relu(self._resnet(x, self.conv81, self.bn81, self.conv82, self.bn82))

        x_left = self.relu(self.bn9_left(self.conv9_left(x)))
        x_left = torch.flatten(x_left).view(-1, 90)

        x_left = self.relu(self.fc91_left(x_left))
        x_left = self.fc92_left(x_left)

        x_right = self.relu(self.bn9_right(self.conv9_right(x)))
        x_right = torch.flatten(x_right).view(-1, 180)
        x_right = self.fc91_right(x_right)

        return F.log_softmax(x_right, dim=1), torch.tanh(x_left)
