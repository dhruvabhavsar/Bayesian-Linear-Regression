#!/usr/local/bin/python3
# Submitted by: Dhruva Bhavsar, Username:dbhavsar
# Programming Project 2: 

import numpy as np
import matplotlib.pyplot as plt

def abc(phi,t,N,f):
    
    I=np.identity(f)
    pt=np.transpose(phi)
    a=np.matmul(pt,phi)
    b=np.matmul(pt,t)
    M=[]
    
    for l in range(0,130):
        c=np.linalg.inv(np.add(l*I,a))
        w=np.matmul(c,b)
#        print(w)
        MSE1=0
        for i in range(0,N):
            MSE1+=(np.subtract(np.matmul(np.transpose(phi[i]),w),t[i]))**2
        MSE=(1/N)*MSE1
#        print(MSE)
        M.append(MSE)
    return M

if __name__ == "__main__":

    phi_train=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/train-wine.csv',delimiter=',')
    t_train=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/trainR-wine.csv',delimiter=',')

    phi_test=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/test-wine.csv',delimiter=',')
    t_test=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/testR-wine.csv',delimiter=',')
#    print(phi_test.shape)
    train=abc(phi_train,t_train,phi_train.shape[0],phi_train.shape[1])
    test=abc(phi_test,t_test,phi_test.shape[0],phi_test.shape[1])
    
    R=np.arange(0,130)
    plt.xlabel("Regularization parameter lambda")
    plt.ylabel("MSE")
    plt.plot(R,train,label="Train set")
    plt.plot(R,test,label="Test set")
    plt.legend(loc='upper left')
    plt.show()


