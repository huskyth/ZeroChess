from torch.utils.data import dataset
from torch.utils.data import dataset
from china_chess.constant import *
from china_chess.algorithm.file_tool import *


class LoadData(dataset.Dataset):

    def __init__(self):
        super(LoadData, self).__init__()
        self.data_list = []
        self._get_examples()

    def __getitem__(self, index):
        return self.data_list[index]

    def __len__(self):
        return len(self.data_list)

    def _get_examples(self):
        files = all_files(TRAIN_DATASET_PATH)
        for f in files:
            self.data_list.append(read(f))

    def get_all_examples(self):
        temp = []
        for x in self.data_list:
            temp += x
        return temp


if __name__ == '__main__':
    l = LoadData()
    print(len(l))
    print(l[0])
