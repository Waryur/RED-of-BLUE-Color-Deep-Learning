import numpy as np
import os
from skimage import io
from matplotlib import pyplot as plt
import pickle as pkl
'''
Input will be [12, 1]
Hidden Layers will be 3 
with 16 neurons in 1st layer
with 10 neurons in 2nd layer
with 4 neurons in 3rd layer
with 2 neurons in output
'''

DataFolder = os.listdir("C:\\Users\castrojl\Desktop\color database")
read = "C:\\Users\\castrojl\\Desktop\\color database\\"

def tanh_p(x):
    return 1 - np.tanh(x)**2

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

w1 = np.load("weights1.npy")
b1 = np.load("bias1.npy")

w2 = np.load("weights2.npy")
b2 = np.load("bias2.npy")

w3 = np.load("weights3.npy")
b3 = np.load("bias3.npy")

w4 = np.load("weights4.npy")
b4 = np.load("bias4.npy")

w5 = np.load("weights5.npy")
b5 = np.load("bias5.npy")

#InputData = np.array([#Shades of Red
#                      [222, 89, 89],
#                      [222, 13, 13],
#                      [204, 33, 33],
#                      [240, 103, 103],
#                      [186, 71, 71],
#                      [163, 3, 3],
#                      [176, 49, 49],
#                      [196, 0, 0],
#                      [158, 24, 24],
#                      [186, 11, 11],
#                      [255, 0, 0],
#                      [255, 79, 79],
#                      [250, 105, 127],
#                      [255, 148, 164],
#                      [199, 139, 148],
#                      [242, 167, 178],
#                      [217, 176, 177],
#                      [250, 200, 201],
#                      #Shades of Blue
#                      [79, 100, 255],
#                      [0, 30, 255],
#                      [84, 104, 255],
#                      [32, 50, 186],
#                      [0, 18, 158],
#                      [5, 22, 150],
#                      [72, 91, 232],
#                      [30, 47, 176],
#                      [0, 21, 179],
#                      [5, 21, 135],
#                      [34, 57, 227],
#                      [68, 82, 189],
#                      [182, 150, 255],
#                      [148, 105, 250],
#                      [167, 167, 242],
#                      [128, 128, 189],
#                      [200, 200, 247],
#                      [174, 174, 214]
#                      #Other colors
#                      #[230, 242, 97],
#                      #[153, 242, 97],
#                      #[97, 242, 131],
#                      #[0, 242, 117],
#                      #[247, 57, 212],
#                      #[8, 230, 255],
#                      #[8, 255, 237],
#                      #[27, 158, 42],
#                      #[149, 157, 27],
#                      #[158, 27, 145],
#                      #[237, 180, 231],
#                      #[180, 237, 231],
#                      #[180, 237, 193],
#                      #[212, 237, 180],
#                      #[232, 237, 180]
#                      ])

#TargetData = np.array([[0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
#                       [0, 1],#, 0],
                       
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0],# 0],
#                       [1, 0]# 0],

#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       #[0, 0, 1],
#                       ])



#w1 = np.random.randn(6, 3)
#b1 = np.random.randn(6, 1)

#w2 = np.random.randn(6, 6)
#b2 = np.random.randn(6, 1)

#w3 = np.random.randn(10, 6)
#b3 = np.random.randn(10, 1)

#w4 = np.random.randn(6, 10)
#b4 = np.random.randn(6, 1)

#w5 = np.random.randn(2, 6)
#b5 = np.random.randn(2, 1)

#iterations = 5000
#lr = 0.1
#costlist = []

#for i in range(iterations):
    
#    random = np.random.choice(len(InputData))
#    InputData1 = InputData[random].reshape(3, 1)

#    z1 = np.dot(w1, InputData1) + b1
#    a1 = np.tanh(z1)

#    z2 = np.dot(w2, a1) + b2
#    a2 = np.tanh(z2)

#    z3 = np.dot(w3, a2) + b3
#    a3 = np.tanh(z3)

#    z4 = np.dot(w4, a3) + b4
#    a4 = np.tanh(z4)

#    z5 = np.dot(w5, a4) + b5
#    a5 = ReLU(z5)

#    cost = np.sum(np.square(a5 - TargetData[random].reshape(2, 1)))

#    if i % 100 == 0:
#        c = 0
#        for x in range(len(DataFolder)):

#            InputData2 = InputData[x].reshape(3, 1)

#            z1 = np.dot(w1, InputData2) + b1
#            a1 = np.tanh(z1)

#            z2 = np.dot(w2, a1) + b2
#            a2 = np.tanh(z2)

#            z3 = np.dot(w3, a2) + b3
#            a3 = np.tanh(z3)

#            z4 = np.dot(w4, a3) + b4
#            a4 = np.tanh(z4)

#            z5 = np.dot(w5, a4) + b5
#            a5 = ReLU(z5)

#            c += np.sum(np.square(a5 - TargetData[x].reshape(2, 1)))
#        costlist.append(c)

#    #print(cost)

#    #backprop
#    dcda5 = 2 * (a5 - TargetData[random].reshape(2, 1))
#    da5dz5 = tanh_p(z5)
#    dz5dw5 = a4

#    dz5da4 = w5
#    da4dz4 = tanh_p(z4)
#    dz4dw4 = a3

#    dz4a3 = w4
#    da3dz3 = tanh_p(z3)
#    dz3dw3 = a2

#    dz3da2 = w3
#    da2dz2 = tanh_p(z2)
#    dz2dw2 = a1

#    dz2da1 = w2
#    da1dz1 = dReLU(z1)
#    dz1dw1 = InputData1

#    dw5 = dcda5 * da5dz5
#    db5 = np.sum(dw5, axis=1, keepdims=True)
#    w5 = w5 - lr * np.dot(dw5, dz5dw5.T)
#    b5 = b5 - lr * db5

#    dw4 = np.dot(dz5da4.T, dw5) * da4dz4
#    db4 = np.sum(dw4, axis=1, keepdims=True)
#    w4 = w4 - lr * np.dot(dw4, dz4dw4.T)
#    b4 = b4 - lr * db4

#    dw3 = np.dot(dz4a3.T, dw4) * da3dz3
#    db3 = np.sum(dw3, axis=1, keepdims=True)
#    w3 = w3 - lr * np.dot(dw3, dz3dw3.T)
#    b3 = b3 - lr * db3

#    dw2 = np.dot(dz3da2.T, dw3) * da2dz2
#    db2 = np.sum(dw2, axis=1, keepdims=True)
#    w2 = w2 - lr * np.dot(dw2, dz2dw2.T)
#    b2 = b2 - lr * db2

#    dw1 = np.dot(dz2da1.T, dw2) * da1dz1
#    db1 = np.sum(dw1, axis=1, keepdims=True)
#    w1 = w1 - lr * np.dot(dw1, dz1dw1.T)
#    b1 = b1 - lr * db1

#print("W1: \n{}\n".format(w1))
#print("B1: \n{}\n".format(b1))

#print("W2: \n{}\n".format(w2))
#print("B2: \n{}\n".format(b2))

#print("W3: \n{}\n".format(w3))
#print("B3: \n{}\n".format(b3))

#print("W4: \n{}\n".format(w4))
#print("B4: \n{}\n".format(b4))

#print("W4: \n{}\n".format(w5))
#print("B4: \n{}\n".format(b5))

#cost = 0

#for x in range(len(InputData)):
#    InputData1 = InputData[x].reshape(3, 1)
    
#    z1 = np.dot(w1, InputData1) + b1
#    a1 = np.tanh(z1)

#    z2 = np.dot(w2, a1) + b2
#    a2 = np.tanh(z2)

#    z3 = np.dot(w3, a2) + b3
#    a3 = np.tanh(z3)

#    z4 = np.dot(w4, a3) + b4
#    a4 = np.tanh(z4)

#    z5 = np.dot(w5, a4) + b5
#    a5 = ReLU(z5)

#    c = np.sum(np.square(a5 - TargetData[x].reshape(2, 1)))
#    print(InputData[x])
#    print("Prediction: \n{}\n".format(np.round(a5)))
#    print("Target: \n{}\n".format(TargetData[x].reshape(2, 1)))
#    print("Cost: {}\n".format(c))

#    if np.round(a4[0]) == 1:
#        print("It's BLUE\n")
#    else:
#        print("It's RED\n") 
#    cost += c
#print("Total Cost: {}\n".format(cost))

#if cost < 0.09:
#    np.save("weights1", w1)
#    np.save("weights2", w2)
#    np.save("weights3", w3)
#    np.save("weights4", w4)
#    np.save("weights5", w5)
#    np.save("bias1", b1)
#    np.save("bias2", b2)
#    np.save("bias3", b3)
#    np.save("bias4", b4)
#    np.save("bias5", b5)

#plt.plot(costlist)
#plt.show()

while True:
    print("Please enter the RGB value for Shade of RED or BLUE")
    r = input("For R: ")
    g = input("For G: ")
    b = input("For B: ")

    x = np.array([[float(r)],
                  [float(g)],
                  [float(b)]])

    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)

    z2 = np.dot(w2, a1) + b2
    a2 = np.tanh(z2)

    z3 = np.dot(w3, a2) + b3
    a3 = np.tanh(z3)

    z4 = np.dot(w4, a3) + b4
    a4 = np.tanh(z4)

    z5 = np.dot(w5, a4) + b5
    a5 = ReLU(z5)

    if np.round(a5[0]) == 1:
        print("It's BLUE\n")
    else:
        print("It's RED\n") 
