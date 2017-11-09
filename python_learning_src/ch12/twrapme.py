# !/usr/bin/python
"""
This code for twrap  test.
__author__ = 'Smalltao'
"""

from time import time
from time import ctime
from time import sleep


class TimeWrapMe(object):
	def __init__(self, obj):
		self.__data = obj
		self.__ctime = self.__mtime = self.__atime = time()

	def set(self, obj):
		self.__data = obj
		self.__mtime = self.__atime = time()

	def get(self):
		self.__atime = time()
		return self.__data

	def get_time_val(self, t_type):
		if not isinstance(t_type, str) or \
			t_type[0] not in 'cma':
			raise TypeError, "argument of 'c', 'm', or 'a' req 'd'"
		return eval('self._%s__%stime' % (self.__class__.__name__, t_type[0]))

	def __repr__(self):
		self.__atime = time()
		return self.__data

	def get_time_str(self, t_type):
		return ctime(self.get_time_val(t_type))

	def __str__(self):
		self.__atime = time()
		return str(self.__data)

	def __getattr__(self, attr):
		self.__atime = time()
		return getattr(self.__data, attr)


if __name__ == '__main__':
	Obj = TimeWrapMe(9324)
	print Obj.get_time_str('c')
	print Obj.get_time_str('m')
	print Obj.get_time_str('a')
	sleep(5)
	print Obj
	print Obj.get_time_str('c')
	print Obj.get_time_str('m')
	print Obj.get_time_str('a')
	sleep(5)
	Obj.set('time is up')
	print
	print Obj.get_time_str('m')
	print Obj
	print Obj.get_time_str('c')
	print Obj.get_time_str('m')
	print Obj.get_time_str('a')


