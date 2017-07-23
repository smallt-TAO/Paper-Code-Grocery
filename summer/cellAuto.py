# -*-coding:utf-8-*-

import random
import numpy as np
from Tkinter import *


class Solution(object):
    def __init__(self):
        self.colors = ['white', 'green', 'red', 'black']
        self.root = Tk()
        self.canvas = Canvas(self.root, width=300, height=300, bg="gray")
        """
        n = input("space?")
        percent = input("percent?")
        """
        self.n = 30
        self.percent = 60

        self.earth = np.array([[0] * self.n] * self.n)
        # states = [' ', '1', '○', '■']
        self.states = ['0', '1', '2', '3']
        self.next_turn = np.array([[0] * self.n] * self.n)

    def show_board(self):
        for i in range(self.n - 1):
            for j in range(self.n - 1):
                self.canvas.create_rectangle(10 * i + 1, 10 * j + 1,
                                             10 * i + 9, 10 * j + 9,
                                             fill=self.colors[self.earth[i][j]])
                self.canvas.pack()
        self.canvas.update()

    def tree_set(self):
        for i in range(1, self.n):
            for j in range(1, self.n):
                per = random.uniform(0, 1) * 100
                if per >= self.percent:
                    self.earth[i][j] = 0
                else:
                    self.earth[i][j] = 1

    def lighting(self):
        i = random.randint(1, self.n - 1)
        j = random.randint(1, self.n - 1)
        self.earth[i][j] = 2

    def burning1(self):
        for i in range(1, self.n - 1):
            for j in range(1, self.n - 1):
                if self.earth[i][j] == 1:
                    up = self.earth[i][j - 1]
                    down = self.earth[i][j + 1]
                    left = self.earth[i - 1][j]
                    right = self.earth[i + 1][j]
                    if up == 2 or down == 2 or left == 2 or right == 2:
                        self.next_turn[i][j] = 2
                    else:
                        self.next_turn[i][j] = 1
                elif self.earth[i][j] == 2:
                    self.next_turn[i][j] = 3
                else:
                    self.next_turn[i][j] = self.earth[i][j]
        self.earth = self.next_turn

    def step(self):
        self.tree_set()  # init the forest
        self.lighting()  # init the fire
        counter = 0
        while counter < 300:
            counter += 1
            self.burning1()
            self.show_board()
        self.root.mainloop()


if __name__ == '__main__':
    solution = Solution()
    solution.step()
