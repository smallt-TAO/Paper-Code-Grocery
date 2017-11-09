import numpy
import cv2
import os

file_path = 'image'
path_dir = os.listdir(file_path)

# print(os.path.join(file_path, path_dir[2]))
for i in range(len(path_dir)):
    image = cv2.imread(os.path.join(file_path, path_dir[i]))
    # cv2.imshow("Original", image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Gray", gray)

    # if don't use a floating point data type when computing
    # the gradient magnitude image, you will miss edges

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite("result_threshold\\{}_edge_by_threshold.jpg".format(str(path_dir[i])),
                numpy.hstack([gray, thresh]))
