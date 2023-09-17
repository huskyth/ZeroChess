from matplotlib import pyplot as plt

from china_chess.algorithm.data_loader import LoadData

from china_chese_game import *


def array2str(array):
    s = ''
    for x in array:
        for y in x:
            s += str(y)
    return s


def parse(a_set, b_set):
    temp_a = {}
    temp_b = {}
    idx = 0
    s2idx = {}
    for a in a_set:
        s = array2str(a[0])
        if s not in s2idx:
            s2idx[s] = idx
            idx += 1
        else:
            pass
    for a in b_set:
        s = array2str(a[0])
        if s not in s2idx:
            s2idx[s] = idx
            idx += 1
        else:
            pass

    for a in a_set:
        s = array2str(a[0])
        if s not in temp_a:
            temp_a[s2idx[s]] = 1
        else:
            temp_a[s2idx[s]] += 1

    for a in b_set:
        s = array2str(a[0])
        if s not in temp_b:
            temp_b[s2idx[s]] = 1
        else:
            temp_b[s2idx[s]] += 1
    return temp_a.values(), temp_b.values()


if __name__ == "__main__":
    g = ChinaChessGame()
    import seaborn as sns
    from random import shuffle

    l = LoadData().get_all_examples()
    for i in range(len(l)):
        e = l[i]
        b, p = g.getSymmetries(e[0], e[1])[1]
        l.append((b, p, e[2]))
    shuffle(l)
    n = len(l)
    train_n = int(0.8 * n)

    train_set = l[:train_n]
    test_set = l[train_n:]

    # train_pi = [np.argmax(np.array(x[1])) for x in train_set]
    # test_pi = [np.argmax(np.array(x[1])) for x in test_set]
    #
    # train_v = [x[2] for x in train_set]
    # test_v = [x[2] for x in test_set]
    # print(len(train_pi), len(test_pi))

    train_board, test_board = parse(train_set, test_set)
    # 绘KDE对比分布
    # sns.kdeplot(train_pi, shade=True, color='r', label='train')
    # sns.kdeplot(test_pi, shade=True, color='b', label='test')

    # sns.kdeplot(train_v, shade=True, color='r', label='train')
    # sns.kdeplot(test_v, shade=True, color='b', label='test')

    sns.kdeplot(train_board, shade=True, color='r', label='train')
    sns.kdeplot(test_board, shade=True, color='b', label='test')

    plt.xlabel('Feature')
    plt.legend()
    plt.show()

    from scipy import stats

    # y = stats.ks_2samp(train_pi, test_pi)
    # print(y)
