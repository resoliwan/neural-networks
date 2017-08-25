import numpy as np
import matplotlib.pyplot as plt
from testCases import * 
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)

X, Y = load_planar_dataset()
#  X (2, 400)
#  Y (1, 400)

plt.scatter(X[0, :], X[1, :], c=Y, cmap=plt.cm.Spectral)
plt.show(block=False)

plt.close()

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T);
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title('Logistic regression')

plt.show(block=False)


