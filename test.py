import numpy as  np


x = np.array([[1,2],[3,4],[5,6]])

np.repeat(x, [1, 2, 1], axis=0)

a = [1,2]
b = [1,2]
c = [3,4]
for i,j, k in a,b,c:
    print(i + j+k)