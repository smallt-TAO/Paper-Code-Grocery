import math
import csv
import numpy as np
import random

def list_normal(list_one):
    max_var, min_var = max(list_one), min(list_one)
    return [(var - min_var) / (max_var - min_var) for var in list_one]

def random_sample_function(rand_len=25):
    # random function
    fun_one = lambda x : x + 1
    fun_two = lambda x : x ** 2
    fun_three = lambda x : x ** 3
    fun_four = lambda x : random.randint(-2, 30)
    function_list = [math.sin, math.cos]
    function_list.extend([fun_one, fun_two, fun_three])
    choose_fun = random.choice(function_list)

    # random x
    rand_start, rand_gap = random.uniform(-5, 5), random.uniform(-1, 3)

    # add gauss
    pure_list = [choose_fun(rand_start + rand_gap * i) for i in range(rand_len)]
    gauss_rand_mean, gauss_rand_variance = random.uniform(0, 2), random.uniform(0, 0.06)
    gauss_list = np.random.normal(gauss_rand_mean, gauss_rand_variance, rand_len)
    res_list = [float(pure_list[i] + gauss_list[i]) for i in range(rand_len)]
    return res_list


def model_one(first_len, second_len):
    gen_list_start = [30000] * first_len
    while max(gen_list_start) > 20000 or min(gen_list_start) < -20:
        gen_list_start = random_sample_function(first_len)
    gen_list_end = [30000] * second_len
    while max(gen_list_end) > 20000 or min(gen_list_end) < -20:
        gen_list_end = random_sample_function(second_len)
    return list_normal(gen_list_start + gen_list_end)


def fake_data(fake_sum=100, fake_len=51):
    fake_list = []

    # solution one
    for i in range(fake_sum):
        left, right = 26, 25
        fake_list.append(model_one(left, right) + [0, 1])

        left0 = 26 - random.randint(1, int(fake_len/3))
        right0 = fake_len - left0
        fake_list.append(model_one(left0, right0) + [1, 0])

        left0 = 26 + random.randint(1, int(fake_len/3))
        right0 = fake_len - left0
        fake_list.append(model_one(left0, right0) + [1, 0])

    print (len(fake_list[0]))
    with open('train.csv', 'w') as wr:
        writer = csv.writer(wr)
        for i in range(len(fake_list)):
            writer.writerow(fake_list[i])


if __name__ == "__main__":
    fake_data(40000)
