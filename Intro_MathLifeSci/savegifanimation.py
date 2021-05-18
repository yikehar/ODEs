import os
from PIL import Image

Tmax = 5000
len_T = len(str(Tmax))  #the number of digits in Tmax
path = os.getcwd() + "\\GIF\\"
files = [path + str(t).zfill(len_T) + '.png' for t in range(Tmax)]
images = list(map(lambda file: Image.open(file), files))
images.pop(0).save(path + "test.gif" ,save_all = True, append_images = images, duration = 100, optimize = False, loop = 0)