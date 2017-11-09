from numpy import *


def create_data_set():
    group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    labels0 = ['A', 'A', 'B', 'B']  # four samples and two classes
    return group, labels0


# classify using kNN
def knn_classify(new_input, data_set, labels0, k):
    num_samples = data_set.shape[0]  # shape[0] stands for the num of row

    # step 1: calculate Euclidean distance
    diff0 = tile(new_input, (num_samples, 1)) - dataSet  # Subtract element-wise
    squared_diff = diff0 ** 2  # squared for the subtract
    squared_dist = sum(squared_diff, axis=1)  # sum is performed by row
    distance = squared_dist ** 0.5

    # step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sorted_dist_indices = argsort(distance)

    class_count = {}  # define a dictionary (can be append element)
    for i in range(k):
        # step 3: choose the min k distance
        vote_label = labels0[sorted_dist_indices[i]]

        # step 4: count the times labels occur
        # when the key voteLabel is not in dictionary class_count, get()
        # will return 0
        class_count[vote_label] = class_count.get(vote_label, 0) + 1

    # step 5: the max voted class will return
    max_count = 0
    max_index = 0
    for key0, value0 in class_count.items():
        if value0 > max_count:
            max_count = value0
            max_index = key0

    return max_index


if __name__ == '__main__':
    dataSet, labels = create_data_set()

    testX = array([1.2, 1.0])
    k = 3
    outputLabel = knn_classify(testX, dataSet, labels, 3)
    print("Your input is:", testX, "and classified to class: ", outputLabel)

    testX = array([0.1, 0.3])
    outputLabel = knn_classify(testX, dataSet, labels, 3)
    print("Your input is:", testX, "and classified to class: ", outputLabel)
