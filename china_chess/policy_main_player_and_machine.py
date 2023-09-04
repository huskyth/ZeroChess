import sys
import pygame

from china_chess.algorithm.alg_game_adapter import PolicyAdapter
from china_chess.algorithm.china_chess_board import *
from china_chess.click_box import ClickBox


class Game(object):
    """
    游戏类
    """

    def __init__(self, screen, chessboard):
        self.screen = screen
        self.player = "r"  # 默认走棋的为红方r
        self.player_tips_r_image = pygame.image.load("images/red.png")
        self.player_tips_r_image_topleft = (550, 500)
        self.player_tips_b_image = pygame.image.load("images/black.png")
        self.player_tips_b_image_topleft = (550, 100)
        self.show_attack = False
        self.show_attack_count = 0
        self.show_attack_time = 100
        self.attack_img = pygame.image.load("images/pk.png")
        self.show_win = False
        self.win_img = pygame.image.load("images/win.png")
        self.win_player = None
        self.show_win_count = 0
        self.show_win_time = 300
        self.chessboard = chessboard

    def get_player(self):
        """
        获取当前走棋方
        """
        return self.player

    def exchange(self):
        """
        交换走棋方
        """
        self.player = "r" if self.player == "b" else "b"
        return self.get_player()

    def reset_game(self):
        """重置游戏"""
        # 所谓的重置游戏，就是将棋盘恢复到默认，走棋方默认的红方
        # 重建新的默认棋子
        self.chessboard.create_chess()
        # 设置走棋方为红方
        self.player = 'r'

    def show(self):
        # 如果一方获胜，那么显示"赢"
        # 通过计时，实现显示一会"将军"之后，就消失
        if self.show_win:
            self.show_win_count += 1
            if self.show_win_count == self.show_win_time:
                self.show_win_count = 0
                self.show_win = False
                self.reset_game()  # 游戏玩过一局之后，重置游戏

        if self.show_win:
            if self.win_player == "b":
                self.screen.blit(self.win_img, (550, 100))
            else:
                self.screen.blit(self.win_img, (550, 450))
            return

        # 通过计时，实现显示一会"将军"之后，就消失
        if self.show_attack:
            self.show_attack_count += 1
            if self.show_attack_count == self.show_attack_time:
                self.show_attack_count = 0
                self.show_attack = False

        if self.player == "r":
            self.screen.blit(self.player_tips_r_image, self.player_tips_r_image_topleft)
            # 显示"将军"效果
            if self.show_attack:
                self.screen.blit(self.attack_img, (230, 400))
        else:
            # 显示"将军"效果
            if self.show_attack:
                self.screen.blit(self.attack_img, (230, 100))
            self.screen.blit(self.player_tips_b_image, self.player_tips_b_image_topleft)

    def set_attack(self):
        """
        标记"将军"效果
        """
        self.show_attack = True

    def set_win(self, win_player):
        """
        设置获胜方
        """
        self.show_win = True
        self.win_player = win_player


class Dot(object):
    group = list()  # 这个类属性用来存储所有的“可落子对象”的引用

    def __init__(self, screen, row, col):
        """初始化"""
        self.image = pygame.image.load("images/dot2.png")
        self.rect = self.image.get_rect()
        self.rect.topleft = (60 + col * 57, 60 + row * 57)
        self.screen = screen
        self.row = row
        self.col = col

    def show(self):
        """显示一颗棋子"""
        self.screen.blit(self.image, self.rect.topleft)

    @classmethod
    def create_nums_dot(cls, screen, pos_list):
        """批量创建多个对象"""
        for temp in pos_list:
            cls.group.append(cls(screen, *temp))

    @classmethod
    def clean_last_position(cls):
        """
        清除所有可以落子对象
        """
        cls.group.clear()

    @classmethod
    def show_all(cls):
        for temp in cls.group:
            temp.show()

    @classmethod
    def click(cls):
        """
        点击棋子
        """
        for dot in cls.group:
            if pygame.mouse.get_pressed()[0] and dot.rect.collidepoint(pygame.mouse.get_pos()):
                print("被点击了「可落子」对象")
                return dot


def start_from_integer_map(integer_map):
    pass


class MainEntrance:

    def _init_pygame(self):
        # 初始化pygame
        pygame.init()
        # 创建用来显示画面的对象（理解为相框）
        self.screen = pygame.display.set_mode((750, 667))
        # 游戏背景图片
        self.background_img = pygame.image.load("images/bg.jpg")
        # 创建计时器
        self.clock = pygame.time.Clock()

    def _init_game(self):
        # 创建棋盘对象
        self.chessboard = ChinaChessBoard(self.screen)
        # 创建游戏对象（像当前走棋方、游戏是否结束等都封装到这个对象中）
        self.game = Game(self.screen, self.chessboard)
        self.clicked_row, self.clicked_col = None, None

    def map_from_outer_integer_map(self, board):
        self.chessboard.to_chess_map(board)

    def set_policy(self, policy):
        self.policy = policy

    def __init__(self):
        self.policy = PolicyAdapter()
        self._init_pygame()
        self._init_game()

    def _select_a_chess(self):
        self.clicked_row, self.clicked_col = None, None
        clicked_chess = Chess.get_clicked_chess(self.game.get_player(), self.chessboard)
        if clicked_chess:
            self.clicked_row, self.clicked_col = clicked_chess.row, clicked_chess.col
            # 创建选中棋子对象
            ClickBox(self.screen, clicked_chess.row, clicked_chess.col)
            # 清除之前的所有的可以落子对象
            Dot.clean_last_position()
            # 计算当前被点击的棋子可以落子的位置
            put_down_chess_pos = self.chessboard.get_put_down_postion(clicked_chess)
            # 根据当前被点击的棋子创建可以落子的对象
            Dot.create_nums_dot(self.screen, put_down_chess_pos)
            return True
        return False

    def _move_a_chess(self):
        clicked_dot = Dot.click()

        if clicked_dot:
            self.chessboard.move_chess_old(clicked_dot.row,
                                       clicked_dot.col, self.game.get_player(), self.clicked_row, self.clicked_col)
            # 清理「点击对象」、「可落子位置对象」
            Dot.clean_last_position()
            ClickBox.clean()

            if self.chessboard.judge_attack_general(self.game.get_player()):
                print("将军....")
                # 检测对方是否可以挽救棋局，如果能挽救，就显示"将军"，否则显示"胜利"
                if self.chessboard.judge_win(self.game.get_player()):
                    print("获胜...")
                    self.game.set_win(self.game.get_player())
                else:
                    # 如果攻击到对方，则标记显示"将军"效果
                    self.game.set_attack()
            return True
        return False

    def _step_by_user(self):
        result = self._move_a_chess()
        if result:
            return True
        else:
            self._select_a_chess()
            return False

    def _step_by_policy(self, player):
        old_row, old_col, new_row, new_col = self.policy.get_next_policy(self.chessboard.chessboard_map, player)

        if None in [old_row, old_col, new_row, new_col]:
            self.game.set_win('r')
            return True

        self.chessboard.move_chess_old(new_row, new_col, self.game.get_player(), old_row, old_col)
        # 检测落子后，是否产生了"将军"功能
        if self.chessboard.judge_attack_general(self.game.get_player()):
            print("将军....")
            # 检测对方是否可以挽救棋局，如果能挽救，就显示"将军"，否则显示"胜利"
            if self.chessboard.judge_win(self.game.get_player()):
                print("获胜...")
                self.game.set_win(self.game.get_player())
            else:
                # 如果攻击到对方，则标记显示"将军"效果
                self.game.set_attack()

    def _black_step(self):
        self._step_by_policy('b')

        # 退出for，以便不让本次的鼠标点击串联到点击棋子
        return True

    def _red_step(self):
        return self._step_by_user()

    def _game_logic(self):
        # 如果游戏没有获胜方，则游戏继续，否则一直显示"获胜"
        if self.game.show_win:
            return False
        if self.game.get_player() == 'r':
            temp = self._red_step()
        else:
            temp = self._black_step()
        if temp:
            # 落子之后，交换走棋方
            self.game.exchange()
            return temp
        else:
            return False

    def _show(self):
        # 显示游戏背景
        self.screen.blit(self.background_img, (0, 0))
        self.screen.blit(self.background_img, (0, 270))
        self.screen.blit(self.background_img, (0, 540))

        # 显示棋盘以及棋子
        self.chessboard.show_chessboard_and_chess()

        # 标记点击的棋子
        ClickBox.show()

        # 显示可以落子的位置图片
        Dot.show_all()

        # 显示游戏相关信息
        self.game.show()

        # 显示screen这个相框的内容（此时在这个相框中的内容像照片、文字等会显示出来）
        pygame.display.update()

        # FPS（每秒钟显示画面的次数）
        self.clock.tick(60)  # 通过一定的延时，实现1秒钟能够循环60次

    def start(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()  #
                if self._game_logic():
                    break
            self._show()


if __name__ == '__main__':
    policy = PolicyAdapter()
    main_entrance = MainEntrance()
    main_entrance.set_policy(policy)
    main_entrance.start()
