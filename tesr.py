import numpy as np
import os as os
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12],
              [13, 14, 15]],dtype='float32')
print(A)
I, J = A.shape
print(I)

mu = np.mean(A,0)
print('Mean: ', repr(mu))
A_shift = np.copy(A)
for i in range(I):
    A_shift[i] -= mu
print('Cols centered:\n')   # repr-display anything string on python  
print(A_shift)

fig = plt.figure() # 畫圖
ax = fig.gca(projection='3d')
#ax = fig.add_subplot(132, projection='3d') #three axis for (1,1,1)不同程度的縮放
#mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve')
#ax.legend()

plt.show()

os.system("pause")

#######
##linear-regression
#ax+by+cz=d
#######
###
##數據點
###
#xi=zi=yi=11
#c1,c2,c3 = #random-gaussion-disturbution
#a,b,c,d = 5.0, 7.3, 9.1, 4.0
#x=np.random.randn(xi)
#y=np.random.randn(yi)
#z=np.random.randn(zi)
#linalg.lstsq(A, zi)
c1, c2,c3 = 5.0, 9.0,6.0
i = np.r_[1:11]
xi = 0.1*i
yi = 0.1*i
x2=np.r_[c1]
x2=x2.repeat(10)
zi=c2*xi+x2+c3*yi
ei = zi + 0.05 *np.max(zi)* np.random.randn(len(yi))
print(np.max(ei))
print(ei)
print(np.random.randn(len(yi)))
A = np.c_[xi[:, np.newaxis],x2[:, np.newaxis],yi[:, np.newaxis]]
print (A)
c, resid, rank, sigma = linalg.lstsq(A, ei)
print(c)
xi2 = np.r_[0.1:1.0:100j]
yi2 = c[1] + c[0]*xi2+c[2]*xi2 
###
### 
#plt.plot(xi,zi,'x',xi2,yi2)
#plt.axis([0,1.1,3.0,5.5])
#plt.xlabel('$x_i$')
#plt.title('Data fitting with linalg.lstsq')
#plt.show()


