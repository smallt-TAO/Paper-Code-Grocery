# !/usr/bin/python
# coding = utf-8

"""
   This code trained for try except.
   __author__ = 'Smalltao'
"""

from operator import add, sub
from random import randint, choice

ops = {'+': add, '-': sub}
MAX_TRIES = 2


def do_prob():
	op = choice('+-')
	nums = [randint(1, 10) for i in range(2)]
	nums.sort(reverse=True)
	ans = ops[op](*nums)
	pr = '{0} {1} {2} = '.format(nums[0], op, nums[1])
	oop = 0
	while True:
		try:
			if int(raw_input(pr)) == ans:
				print 'correct'
				break
			if oop == MAX_TRIES:
				print 'sorry...the answer is \n {0}{1}'.format(pr, ans)
			else:
				print 'incorrect... try again'
				oop += 1
		except (KeyboardInterrupt, EOFError, ValueError):
			print 'invalid input ... try again'


def main():
	while True:
		do_prob()
		try:
			opt = raw_input('Again? [y] ').lower()
			if opt and opt[0] == 'n':
				break
		except (KeyboardInterrupt, EOFError):
			break


if __name__ == '__main__':
	main()
