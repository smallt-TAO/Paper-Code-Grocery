import logging
import numpy as np

log_file = "./basic_logger.log"

tensor = np.zeros((10, 1, 2, 3))
tensor = tensor.reshape((10, 1, 2, 3)).astype(np.float)

logging.basicConfig(filename=log_file, level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

logging.debug("this is a debug msg!")
logging.info("this is a info msg!")
logging.info("tensor shape is {}".format(str(34)))
logging.warn("this is a warn msg!")
logging.error("this is a error msg!")
logging.critical("this is a critical msg!")
