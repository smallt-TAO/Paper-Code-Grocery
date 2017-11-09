# !/usr/bin/python
"""
This code for iter test.
__author__ = 'Smalltao'
"""

from random import choice


class RandSeq(object):
	def __init__(self, seq):
		self.data = seq

	def __iter__(self):
		return self

	def next(self):
		return choice(self.data)


class AnyIter(object):
	def __init__(self, data, safe=0):
		self.safe = safe
		self.iter = iter(data)

	def __iter__(self):
		return self

	def next(self, how_many=1):
		ret_val = []
		for each_Item in range(how_many):
			try:
				ret_val.append(self.iter.next())
			except StopIteration:
				if self.safe:
					break
				else:
					raise
		return ret_val


if __name__ == '__main__':
	for eachItem in RandSeq(('rock', 'hui')):
		print eachItem
