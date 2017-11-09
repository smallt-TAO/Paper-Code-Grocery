# !/usr/bin/python
# coding = utf-8

"""
This code for card run money
__author__ = 'Smalltao'
"""


def safe_float(object):
	"""safe version of float()"""
	try:
		return_value = float(object)
	except (TypeError, ValueError), diag:
		return_value = str(diag)
	return return_value


def main():
	"""handles all the data processing"""
	log = open('cardlog.txt', 'w')
	try:
		c_file = open('carddata.txt', 'r')
	except IOError, e:
		log.write('no txt this month \n')
		log.close()
		return

	txns = c_file.readlines()
	c_file.close()
	total = 0.00
	log.write('account log: \n')

	for eachTxn in txns:
		result = safe_float(eachTxn)
		if isinstance(result, float):
			total += result
			log.write('data... processed\n')
		else:
			log.write('ignored: {0}'.format(result))
	print '$%.2f (new balance)' % total
	log.close()


if __name__ == '__main__':
	main()
