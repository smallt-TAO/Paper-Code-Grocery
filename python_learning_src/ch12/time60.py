# !/usr/bin/python
"""
This code for class test.
__author__ = 'Smalltao'
"""


class Time60(object):
	"""
	Time60 constructor - takes hours and minutes.
	"""
	def __init__(self, hr, min):
		"""
		:param hr:
		:param min:
		:return:
		"""
		self.hr = hr
		self.min = min

	def __str__(self):
		"""
		:return:
		"""
		return '%d:%d' % (self.hr, self.min)

	__repr__ = __str__

	def __add__(self, other):
		"""

		:param other:
		:return:
		"""
		return self.__class__(self.hr + other.hr, self.min + other.min)

	def __iadd__(self, other):
		"""

		:param other:
		:return:
		"""
		self.hr += other.hr
		self.min += other.min
		return self


if __name__ == "__main__":
	mon = Time60(10, 30)
	tue = Time60(11, 15)
	print mon, tue
	mon + tue
	print mon + tue
	print str(id(mon))
	print mon
	mon += tue
	print mon
	print str(id(mon))
