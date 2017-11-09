# !/user/bin/python

"""
This code for test wrapped function.
__author__ = 'Smalltao'
"""

from time import ctime, sleep


def ts_func(func):
	def wrapped_func():
		print '[{0}] {1}() called'.format(ctime(), func.__name__)
		return func()
	return wrapped_func


@ts_func
def foo():
	pass

foo()
sleep(4)
for i in range(2):
	sleep(2)
	foo()
