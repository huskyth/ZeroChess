from pathlib import Path

abbreviation_to_chinese = {
    "b_c": '黑车', "b_m": "黑马", "b_x": "黑相", "b_s": "黑士", "b_j": "黑将",
    "b_p": "黑炮",
    "b_z": "黑卒",
    "r_z": "红卒",
    "r_p": "红炮",
    "r_c": "红车", "r_m": "红马", "r_x": "红相", "r_s": "红士", "r_j": "红帅",
    "": "一一"
}
MAP_HEIGHT = 10
MAP_WIDTH = 9
RED_STRING = 'r'
RED_INT = 1
BLACK_STRING = 'b'
BLACK_INT = -1
NOT_END = 0
CHINESE_NUMBER_TO_INT = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
}
NOT_SEE_NUMBER_TO_INT = {
    '１': 1, '２': 2, '３': 3, '４': 4, '５': 5, '６': 6, '７': 7, '８': 8, '９': 9
}
DATASET_PATH = Path(__file__).parent / "dataset"
TRAIN_DATASET_PATH = Path(__file__).parent / "train_data"
SUMMARY_PATH = Path(__file__).parent / "summary"
MODEL_PATH = Path(__file__).parent / "algorithm" / "checkpoint"

IMAGE_PATH = Path(__file__).parent


LETTERS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
LETTERS_TO_IND = {}
for i, x in enumerate(LETTERS):
    LETTERS_TO_IND[x] = i

NUMBERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
NUMBERS_TO_IND = {}
for i, x in enumerate(NUMBERS):
    NUMBERS_TO_IND[x] = i
ABBREVIATION_TO_VALUE = {
    "b_c": -1, "b_m": -2, "b_x": -3, "b_s": -4, "b_j": -5,
    "b_p": -6,
    "b_z": -7,
    "r_z": 7,
    "r_p": 6,
    "r_c": 1, "r_m": 2, "r_x": 3, "r_s": 4, "r_j": 5,
    "": 0
}

VALUE_TO_ABBREVIATION = {}
for key in ABBREVIATION_TO_VALUE:
    VALUE_TO_ABBREVIATION[ABBREVIATION_TO_VALUE[key]] = key

NAME_TO_ENGLISH_CHAR = {
    '相': 'r_x',
    '卒': 'b_z',
    '象': 'b_x',
    '兵': 'r_z',
    '帅': 'r_j',
    '将': 'b_j',
    '仕': 'r_s',
    '士': 'b_s',

}
PAO_MA_CHE_AFTER_ENGLISH = {'炮': '_p', '马': '_m', '车': '_c'}

ALL_SELECTION = 2086


# 创建所有合法走子UCI，size 2086
def create_uci_labels():
    labels_array = []

    Advisor_labels = ['d7e8', 'e8d7', 'e8f9', 'f9e8', 'd0e1', 'e1d0', 'e1f2', 'f2e1',
                      'd2e1', 'e1d2', 'e1f0', 'f0e1', 'd9e8', 'e8d9', 'e8f7', 'f7e8']
    Bishop_labels = ['a2c4', 'c4a2', 'c0e2', 'e2c0', 'e2g4', 'g4e2', 'g0i2', 'i2g0',
                     'a7c9', 'c9a7', 'c5e7', 'e7c5', 'e7g9', 'g9e7', 'g5i7', 'i7g5',
                     'a2c0', 'c0a2', 'c4e2', 'e2c4', 'e2g0', 'g0e2', 'g4i2', 'i2g4',
                     'a7c5', 'c5a7', 'c9e7', 'e7c9', 'e7g5', 'g5e7', 'g9i7', 'i7g9']

    for l1 in range(9):
        for n1 in range(10):
            destinations = [(t, n1) for t in range(9)] + \
                           [(l1, t) for t in range(10)] + \
                           [(l1 + a, n1 + b) for (a, b) in
                            [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]]  # 马走日
            for (l2, n2) in destinations:
                if (l1, n1) != (l2, n2) and l2 in range(9) and n2 in range(10):
                    move = LETTERS[l1] + NUMBERS[n1] + LETTERS[l2] + NUMBERS[n2]
                    labels_array.append(move)

    for p in Advisor_labels:
        labels_array.append(p)

    for p in Bishop_labels:
        labels_array.append(p)

    return labels_array


LABELS = create_uci_labels()
LABELS_TO_INDEX = {}
for i, l in enumerate(LABELS):
    LABELS_TO_INDEX[l] = i


def chess_from_chinese_to_english_char(chess_chinese_name, cur_player):
    if chess_chinese_name in PAO_MA_CHE_AFTER_ENGLISH:
        return cur_player + PAO_MA_CHE_AFTER_ENGLISH[chess_chinese_name]
    return NAME_TO_ENGLISH_CHAR[chess_chinese_name]


