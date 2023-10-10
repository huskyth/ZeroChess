from multiprocessing import Pipe, Process
from Coach import CloudpickleWrapper
from china_chess.algorithm.mcts_async import *
from china_chess.algorithm.mcts_origin import *


def handle(parent_remote, worker_remote, mcts):
    parent_remote.close()
    while True:
        cmd, data = worker_remote.recv()
        print(cmd, data)


def multiprocess_handle():
    max_parallel = 10

    parent_remote, worker_remote = zip(*[Pipe() for x in range(max_parallel)])
    mcts_list = [CloudpickleWrapper(MCTS()) for i in range(max_parallel)]

    process_list = [Process(target=handle, args=(p, w, m), daemon=True) for p, w, m in
                    zip(parent_remote, worker_remote, mcts_list)]

    for i in range(max_parallel):
        process_list[i].start()

    for i in range(max_parallel):
        worker_remote[i].close()


    for i in process_list:
        i.join()



if __name__ == '__main__':
    multiprocess_handle()
