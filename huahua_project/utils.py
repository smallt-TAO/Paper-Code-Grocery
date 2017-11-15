import numpy as np

def fake_data(batch_size):
    image_label = []
    for i in range(batch_size):
        image = np.random.randint(225, size=[30, 3500, 1])
        label = [0, 0, 0, 1, 0, 0]
        image_label.append((image, label))
    np.random.shuffle(image_label)
    image = np.array([value[0] for value in image_label])
    label = np.array([value[1] for value in image_label])
    return image, label


if __name__ == "__main__":
    image, label = fake_data(64)
    print image, label
