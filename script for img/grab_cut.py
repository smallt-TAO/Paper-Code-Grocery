import numpy as np
import numpy
import cv2
import os

file_path = 'image'
path_dir = os.listdir(file_path)

# print(os.path.join(file_path, path_dir[2]))
for i in range(len(path_dir)):
    img = cv2.imread(os.path.join(file_path, path_dir[i]))
    # cv2.imshow("Original", image)

    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (10, 2, img.shape[0] - 10, img.shape[1] - 2)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    mask2 = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

    output = cv2.bitwise_and(img, img, mask=mask2)
    cv2.imwrite('grabcut_output.png', output)

    cv2.imwrite("result_grabcut\\{}_edge_by_grabcut.jpg".format(str(path_dir[i])),
                numpy.hstack([img, output]))
