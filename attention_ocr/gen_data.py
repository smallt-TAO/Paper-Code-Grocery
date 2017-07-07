import numpy as np
from PIL import Image
import os

prefix = './testdata'
def get_train_data(filepath='./testdata'):
  filelist = os.listdir(filepath)
  images = list()
  for filename in filelist:
    path = os.path.join(prefix,filename)
    image = Image.open(os.path.join(prefix,filename))
    images.append(np.array(image).resize((150,360,3)))
  images = np.array(images)
  return images

if __name__ == '__main__':
  images = get_train_data()
  print(images.shape)
