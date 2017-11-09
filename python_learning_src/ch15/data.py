#!/usr/bin/python
"""
__author__ Smalltao
"""

from os import popen
from re import split

f = [' hui   hui_hui   890', 'hui_hui 8903    34']
for eachLine in f:
	print split('\s\s+|\t', eachLine.strip())
