import numpy as np 
from PIL import Image
import os

def resize_data(filepath):
  image_names = os.listdir(filepath)
  images_batch = list()
  for  image_name in image_names:
    image = Image.open(os.path.join('./testdata',image_name))
    image = image.resize((150,600))
    images_batch.append(image)
  images_batch = np.array(images_batch,dtype=np.float32)
  return images_batch
