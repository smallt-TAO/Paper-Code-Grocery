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

    # lap handle
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap = numpy.uint8(numpy.absolute(lap))
    cv2.imwrite("result\\{}_edge_by_laplacian.jpg".format(str(path_dir[i])),
                numpy.hstack([gray, lap]))

    # sobel handle
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    sobelx = numpy.uint8(numpy.absolute(sobelx))
    sobely = numpy.uint8(numpy.absolute(sobely))
    sobelcombine = cv2.bitwise_or(sobelx, sobely)
    cv2.imwrite("result\\{}_edge_by_sobel.jpg".format(str(path_dir[i])),
                numpy.hstack([gray, sobelx, sobely, sobelcombine]))

    # canny handle
    canny = cv2.Canny(gray, 30, 150)
    canny = numpy.uint8(numpy.absolute(canny))
    cv2.imwrite("result\\{}_edge_by_canny.jpg".format(str(path_dir[i])),
                numpy.hstack([gray, canny]))

