# -*-coding:utf-8-*-


class Solution(object):
    def __init__(self, n, m, x):
        self.n = n  # n个传教士、n个野人
        self.m = m  # 船能载m人
        self.x = x  # 一个解，就是船的一系列状态
        self.is_found = False  # 全局终止标志

    def get_states(self, k):  # 船准备跑第k趟
        if k % 2 == 0:
            s1, s2 = self.n - sum(s[0] for s in self.x), self.n - sum(s[1] for s in self.x)
        else:
            s1, s2 = sum(s[0] for s in self.x), sum(s[1] for s in self.x)
        for i in range(s1 + 1):
            for j in range(s2 + 1):
                if 0 < i + j <= self.m and (i * j == 0 or i >= j):
                    yield [(-i, -j), (i, j)][k % 2 == 0]  # 生成船的合法状态

    def conflict(self, k):  # 船开始跑第k趟
        if k > 0 and self.x[-1][0] == -self.x[-2][0] and self.x[-1][1] == -self.x[-2][1]:
            return True
        if 0 < self.n - sum(s[0] for s in self.x) < self.n - sum(s[1] for s in self.x):
            return True
        if 0 < sum(s[0] for s in self.x) < sum(s[1] for s in self.x):
            return True
        return False

    def backtrack(self, k):  # 船准备跑第k趟
        if self.is_found:
            return  # 终止所有递归
        if self.n - sum(s[0] for s in self.x) == 0 and self.n - sum(s[1] for s in self.x) == 0:
            print(self.x)
            self.is_found = True
        else:
            for state in self.get_states(k):
                self.x.append(state)
                if not self.conflict(k):
                    self.backtrack(k + 1)  # 深度优先
                self.x.pop()

if __name__ == '__main__':
    solution = Solution(3, 2, [])
    solution.backtrack(0)
