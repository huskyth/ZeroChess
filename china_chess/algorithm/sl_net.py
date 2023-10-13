import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from NeuralNet import NeuralNet
from china_chess.constant import *
import torch
import torch.optim as optim
from china_chess.algorithm.tensor_board_tool import *
from random import shuffle
from china_chess.algorithm.cchess_net import *
from othello.pytorch.OthelloNNet import *

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 5,
    'batch_size': 128,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
})


class NNetWrapper(NeuralNet):
    def __init__(self):
        self.nnet = CChessNNet(args)
        self.board_x, self.board_y = 10, 9
        self.action_size = len(LABELS)
        if args.cuda:
            print("使用了CUDA")
            self.nnet.cuda()

    def train(self, examples, batch_iter, iter_num, lr, epoch):
        n = int(len(examples) * 0.8)
        shuffle(examples)

        training_data = examples[:n]

        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=lr, weight_decay=0.01)
        step = 0
        loss_summary = MySummary("Pi Loss {}_{}".format(batch_iter, iter_num))

        print('EPOCH ::: ' + str(epoch + 1))
        self.nnet.train()
        pi_losses = AverageMeter()
        v_losses = AverageMeter()
        ret_loss = 0
        ret_accuracy = 0
        batch_count = int(len(training_data) / args.batch_size)

        t = tqdm(range(batch_count), desc='Training Net')
        for _ in t:
            step += 1
            sample_ids = np.random.randint(len(training_data), size=args.batch_size)
            boards, pis, vs, players = list(zip(*[training_data[i] for i in sample_ids]))
            boards = torch.FloatTensor(np.array(boards).astype(np.float64))
            target_pis = torch.FloatTensor(np.array(pis))
            target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

            # predict
            if args.cuda:
                boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

            # compute output
            out_pi, out_v = self.nnet(boards)
            l_pi = self.loss_pi(target_pis, out_pi)
            l_v = self.loss_v(target_vs, out_v)
            total_loss = l_pi + l_v
            correct_prediction = torch.equal(torch.argmax(out_pi, 1), torch.argmax(self.examples, 1))
            correct_prediction = torch.cast(correct_prediction, torch.float32)
            ret_accuracy += torch.reduce_mean(correct_prediction, name='accuracy')
            ret_loss += total_loss
            # record loss
            pi_losses.update(l_pi.item(), boards.size(0))
            v_losses.update(l_v.item(), boards.size(0))
            t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
            loss_summary.add_float(step, pi_losses.avg, "Training Policy Loss")
            loss_summary.add_float(step, v_losses.avg, "Training Value Loss")
            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return ret_accuracy / batch_count, ret_loss / batch_count

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(-1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        print(filepath)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
