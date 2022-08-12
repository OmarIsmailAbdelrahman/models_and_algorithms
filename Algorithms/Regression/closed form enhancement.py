# there is two methods in the in solving linear regression and logistic regression, closed-form, and gradient descent
# the problem with closed form is that the compleixty is O(n^3) and the the gradient is O(kn^2)
# using new algorithms in closed form might get the compleixty to O(n^2.4) which can be better than the gradient some cases
# but in general gradient is better, because most of the time there is no closed form solution "this is just a test"

#main goal: can we improve closed form technique and use it?

#after search there is two algorithms to speed the process, Strassen with O(2.8) and  Coppersmith Winograd Algorithm with O(2.3) "galactic algorithm"

#the next is the implementation of strassen algorithm, it is better than the naive solution in large matrcies

import numpy as np
import math
import sys
np.set_printoptions(threshold=sys.maxsize)


def splitting(x):
    row, col = x.shape  # return size
    row2, col2 = row // 2, col // 2  # integer div
    return x[:row2, :col2], x[:row2, col2:], x[row2:, :col2], x[row2:, col2:]


def mul(x, y):
    if len(x) == 1:
        return x * y
    a, b, c, d = splitting(x)
    e, f, g, h = splitting(y)
    # print (a.shape,b.shape,c.shape,d.shape)
    p1 = mul(a, f - h)
    p2 = mul(a + b, h)
    p3 = mul(c + d, e)
    p4 = mul(d, g - e)
    p5 = mul(a + d, e + h)
    p6 = mul(b - d, g + h)
    p7 = mul(a - c, e + f)

    # Computing the values of the 4 quadrants of the final matrix c
    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7
    return np.vstack((np.hstack((c11, c12)), np.hstack((c21, c22))))


z = np.random.randint(100, size=1)[0] + 1
print("size: ", z)

x = np.random.randint(10, size=(z, z))
y = np.random.randint(10, size=(z, z))

print(x.shape, np.zeros((z, 0))[:].shape)

# this is used to make the matrix n in power 2, because the function

if z > math.pow(2, math.floor(math.log(z, 2))):
    diff = (int(math.pow(2, math.floor(math.log(z, 2)) + 1) - z))
    print("diff", diff)

    for i in range(diff):
        x = np.vstack([x, np.zeros(z)])
        y = np.vstack([y, np.zeros(z)])

    x = np.hstack([x, np.zeros(((z + diff),diff))])
    y = np.hstack([y, np.zeros(((z + diff), diff))])
#print(x, x.shape)
#print(mul(x,y))
# iterate through rows of X
result = np.zeros(x.shape)
for i in range(len(x)):
   # iterate through columns of Y
   for j in range(len(y[0])):
       # iterate through rows of Y
       for k in range(len(y)):
           result[i][j] += x[i][k] * y[k][j]
res = mul(x,y)
test = result-res
for i in range(len(x)):
     for j in range(len(y)):
         if test[i][j] != 0:
             print(test[i][j])
             print("fail")
             break;
print(res)





#only from small sample we can see that the algorithm so too slow, the multiplication of one matrix with other will take a large time to process, and that is only one, more than 1000 sample will take more than 15 mins
#a modification to the algorithm to run faster is to make the size n is not an exact power of 2, the modification is to add a padding col and row recursivly whenever we get an odd number.
#but still after modifing the compleixty is higher than the gradient and the constant will be bigger, + there are invertable matrcies

#in conclution even if we used a faster algorithm in closed-form we will still get slower algorith than the gradient
#test: fail