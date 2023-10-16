import os
import time

from tensorboard import summary
from china_chess.constant import SUMMARY_PATH, WANDB_PATH
from torch.utils.tensorboard import SummaryWriter
import wandb

log_dir = SUMMARY_PATH


class MySummary:

    def __init__(self, log_dir_name=None):
        wandb.login(key="613f55cae781fb261b18bad5ec25aa65766e6bc8")
        ticks = str(time.time())
        self.wandb_logger = wandb.init(project="ZeroChess" + ticks, dir=WANDB_PATH)

        log_path = str(log_dir / log_dir_name)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        self.writer = SummaryWriter(log_dir=log_path)

    def add_float(self, x, y, title, x_name):
        self.writer.add_scalar(title, y, x)
        self.wandb_logger.log({x_name: x, title: y})

    def close(self):
        self.writer.close()


if __name__ == '__main__':
    test = wandb.init(project="test")
    for i in range(100):
        test.log({"a": 1, "epoch": i})
        test.log({"b": 2, "iter": i})
