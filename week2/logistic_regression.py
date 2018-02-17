import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

print(train_set_x_orig.shape)
# (209, 64, 64, 3)
print(train_set_y.shape)
# (1, 209)

print(test_set_x_orig.shape)
# (50, 64, 64, 3)
print(test_set_y.shape)
# (1, 50)

print(classes)

index = 1 
plt.imshow(train_set_x_orig[index])
plt.show(block=False)
print('y =' + str(train_set_y[:, index]) + ', it is a' + classes[np.squeeze(train_set_y[:, index])].decode('utf-8'))

plt.close()

#Reshpae the training and test example



