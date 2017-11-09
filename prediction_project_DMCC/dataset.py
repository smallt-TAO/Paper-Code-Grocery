import numpy as np
import csv
import random
import os

def gen_data(sum, dim):
    data_x, data_y = [], []
    for i in range(sum):
        data_x.append([0] * dim)
        data_x.append([1] * dim)
        data_y.append([1, 0])
        data_y.append([0, 1])
    return data_x, data_y


def shuffle_data(data_x, data_y):
    training_data = np.hstack([data_x, data_y])
    np.random.shuffle(training_data)
    data_x = [i[:-2] for i in training_data]
    data_y = [i[-2:] for i in training_data]
    return np.array(data_x), np.array(data_y)


def read_data(data_path, center=51):
    result = []
    with open(data_path, 'rb') as cs:
        reader = csv.reader(cs, delimiter=',')
        rows = [row for row in reader]
        list_temp = []
        counter, res = -1, []
        for r in rows:
            list_temp.append(float(r[1]))
            counter += 1
            if r[2] != '':
                res.append(counter)
        list_max, list_min = max(list_temp), min(list_temp)
        for cou in res:
            pos = [(float(ro[1])-list_min)/(list_max-list_min) for ro in rows[cou - 25: cou + center - 25]]
            ran = random.randint(1, int(center / 4))
            neg = [(float(ro[1])-list_min)/(list_max-list_min) for ro in rows[cou - ran: cou + center - ran]]
            ran = random.randint(2, int(center / 6))
            neg_2 = [(float(ro[1])-list_min)/(list_max-list_min) for ro in rows[cou - ran: cou + center - ran]]
            if len(pos + neg + neg_2) != 3 * center: break
            result.append(pos + [0, 1])
            result.append(neg + [1, 0])
            result.append(neg_2 + [1, 0])
    return result


def load_data(dir_path):
    results = []
    # read the all data
    path_dir = os.listdir(dir_path)
    for allpath in path_dir:
        csv_path = os.path.join('%s/%s' % (dir_path, allpath))
        result = read_data(csv_path)
        results.extend(result)
    
    # print results
    # save the data set
    with open('test.csv', 'w') as wr:
        writer = csv.writer(wr)
        for i in range(len(results)):
            writer.writerow(results[i])


def load_train_data(data_path):
    data_x_y = []
    with open(data_path, "rb") as cs:
        reader = csv.reader(cs)
        for row in reader:
            row = [float(ro) for ro in row]
            data_x_y.append(row)
    # print(data_x_y[0])
    np.random.shuffle(data_x_y)
    data_x = [i[:-2] for i in data_x_y]
    data_y = [i[-2:] for i in data_x_y]
    return np.array(data_x), np.array(data_y)

def read_real_data(data_path):
    with open(data_path, "rb") as cs:
        reader = csv.reader(cs)
        row = [float(row[0]) for row in reader]
    return row

if __name__ == "__main__":
    # a, b = load_train_data("train.csv")
    # print a.size, b.size
    # load_data('all_data')
    read_real_data("data/daht_c001_04_15.csv")
