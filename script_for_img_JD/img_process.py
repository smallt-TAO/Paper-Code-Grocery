import os
import cv2


def img_process():
    file_path = r"D:\ssh\SCUT_FORU_DB_Release\English2k\word_annotation"
    path_dir = os.listdir(file_path)
    for i in range(len(path_dir)):
        txt_path = os.path.join(file_path, path_dir[i])
        with open(txt_path, mode='r', encoding='utf-8') as f:
            for line in f.readlines():
                line_list = line.strip().split(',')
                for j in range(int(len(line_list)/5)):
                    x0 = int(line_list[j])
                    y0 = int(line_list[j + 1])
                    x1 = x0 + int(line_list[j + 2])
                    y1 = y0 + int(line_list[j + 3])
                    label_str = line_list[j + 4]

                    # read the img
                    img_path = r"D:\ssh\SCUT_FORU_DB_Release\English2k\word_img"
                    img_name = path_dir[i].split('.')[0] + '.jpg'
                    img_path = os.path.join(img_path, img_name)
                    img = cv2.imread(img_path)

                    # crop the img
                    img = img[y0:y1, x0:x1]
                    img_label = label_str.lstrip('"').rstrip('"') + '.jpg'
                    cv2.imwrite("real_pic/" + img_label, img)


if __name__ == "__main__":
    img_process()
