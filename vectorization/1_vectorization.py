import numpy as np

a = np.array([1, 2, 3, 4])
print(a)

import time

a = np.random.random(100000)
b = np.random.random(100000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print("Vectorizes:" + str(1000 * (toc - tic)) + 'ms')

c = 0
tic = time.time()
for i in range(len(a)):
    c += a[i] * b[i]
toc = time.time()
    
print("for:" + str(1000 * (toc - tic)) + 'ms')
