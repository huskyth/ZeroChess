import sys

import numpy as np

from china_chess.algorithm.cchess_net_transformer import CChessNNetWithTransformer
from china_chess.algorithm.common.Regularization import Regularization

sys.path.append('../../')
from NeuralNet import NeuralNet
from othello.pytorch.OthelloNNet import *

args = dotdict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 5,
    'batch_size': 128,
    'cuda': torch.cuda.is_available(),
    'num_channels': 128,
})
criterion = torch.nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NNetWrapper(NeuralNet):
    def __init__(self, summary_writer):
        super().__init__(None)
        self.nnet = CChessNNetWithTransformer(args)
        self.board_x, self.board_y = 10, 9
        self.action_size = len(LABELS)
        self.summary = summary_writer
        self.weight_decay = 0.0001
        self.reg_loss = Regularization(self.weight_decay, p=2)
        self.clip_grad = 100

        if args.cuda:
            print("使用了CUDA")
            self.nnet.cuda()
            self.reg_loss.to(device)

    def train(self, state_batch, mcts_probs_batch, winner_batch, step, lr):
        optimizer = optim.SGD(self.nnet.parameters(), lr=lr, momentum=0.9)

        print(f'UPDATE ITER ::: {step}')
        self.nnet.train()

        boards = torch.FloatTensor(np.array(state_batch).astype(np.float64))
        target_pis = torch.FloatTensor(np.array(mcts_probs_batch))
        target_vs = torch.FloatTensor(np.array(winner_batch).astype(np.float64))

        # predict
        if args.cuda:
            boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

        # compute output
        out_pi, out_v = self.nnet(boards)
        l_pi = self.loss_pi(target_pis, out_pi)
        l_v = self.loss_v(target_vs, out_v)
        l2_loss, grad_sum, grad_sum_norm_1 = self.reg_loss(self)
        total_loss = l_pi + l_v + l2_loss
        ret_accuracy = torch.mean((torch.argmax(out_pi, 1) == torch.argmax(target_pis, 1)).float())
        ret_loss = total_loss
        # record loss
        self.summary.add_float(step, l_pi, "Training Policy Loss", "step")
        self.summary.add_float(step, l_v, "Training Value Loss", "step")
        self.summary.add_float(step, l2_loss, "Regularization Loss", "step")
        self.summary.add_float(step, grad_sum, "Grad Norm 2", "step")
        self.summary.add_float(step, grad_sum_norm_1, "Grad Norm 1", "step")
        self.summary.add_float(step, ret_accuracy, "Training Accuracy", "step")
        self.summary.add_float(step, ret_loss, "Training ALl loss", "step")
        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=self.clip_grad)
        optimizer.step()
        return ret_accuracy, ret_loss

    def predict(self, board):
        """
        board: np array with board
        """
        if not isinstance(board, np.ndarray):
            board = np.array(board)
        if len(board.shape) == 4:
            batch = board.shape[0]
        else:
            raise Exception("dimension is not 4")
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda: board = board.contiguous().cuda()
        board = board.view(batch, -1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy(), v.data.cpu().numpy()

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs) ** 2) / targets.size()[0]

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
        filepath = os.path.join(ROOT_PATH / folder, filename)
        if not os.path.exists(filepath):
            raise ("No model in path {}".format(filepath))
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])
