import sys
import os
import numpy as np
import random
import shutil

if __name__ == '__main__':
    image_path_file = sys.argv[1]
    output_path = sys.argv[2]
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    max_num = int(sys.argv[3])
    paths = []
    with open(image_path_file) as lines:
        for line in lines:
            paths.append(line.strip().split(',')[-1])

    random.shuffle(paths)

    s = set()
    while max_num > 0:
        idx = random.randint(0, len(paths)-1)
        f = paths[idx]
        distdir = os.path.join(output_path, f.split('/')[-2])
        if distdir not in s:
            s.add(distdir)
            if not os.path.isdir    (distdir):
                os.makedirs(distdir)
            shutil.copy(f, distdir)
            max_num -= 1