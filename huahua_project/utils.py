import numpy as np
import csv
import os

def gene_label(idx, classes=6):
    res = [0] * classes
    res[idx] = 1
    return res

def target_dict():
    res = {}
    res["AAA"] = 0
    res["BBB"] = 1
    return res
    
def fake_data(batch_size, classes=6):
    image_label = []
    for i in range(batch_size):
        image = np.random.randint(225, size=[30, 3500, 1])
        label = gene_label(np.random.randint(classes), classes)
        image_label.append((image, label))
    np.random.shuffle(image_label)
    image = np.array([value[0] for value in image_label])
    label = np.array([value[1] for value in image_label])
    return image, label

def fake_data_dir(number_fake):
    for i in range(number_fake):
        with open('data/BBB/' + str(i) + '.csv', 'w') as wr:
            writer = csv.writer(wr)
            for j in range(30):
                writer.writerow(np.random.randint(255, size=[3500]))

def read_csv(data_path):
    res = []
    with open(data_path, "rb") as cs:
        reader = csv.reader(cs)
        for row in reader:
            row = [float(i) for i in row]
            res.append(row)
    return res

def read_batch_list(batch_list):
    batch_image = []
    batch_label = []
    for (im, la) in batch_list:
        batch_image.append(read_csv(im))
        batch_label.append(gene_label(la))
    return batch_image, batch_label

def file_batch(dir_path):
    res = []
    dict_label = target_dict()
    data_dir = dir_path
    path_dir = os.listdir(data_dir)
    for all_file in path_dir:
        label = dict_label[all_file]
        sub_path = os.path.join('%s/%s' % (data_dir, all_file))
        sub_file = os.listdir(sub_path)
        for all_csv in sub_file:
            image_path = '{}/{}'.format(sub_path, all_csv)
            res.append((image_path, label))
    np.random.shuffle(res)
    return res


if __name__ == "__main__":
   print(file_batch("data")) 
   read_csv("data/AAA/1.csv")

