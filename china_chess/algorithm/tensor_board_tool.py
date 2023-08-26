from tensorboard import summary
from china_chess.constant import SUMMARY_PATH
from torch.utils.tensorboard import SummaryWriter

log_dir = SUMMARY_PATH


class MySummary:

    def __init__(self):
        self.writer = SummaryWriter(log_dir=log_dir)

    def add_float(self, x, y, title):
        self.writer.add_scalar(title, y, x)
