#!/usr/local/bin/python3
# Submitted by: Dhruva Bhavsar, Username:dbhavsar
# Programming Project 2: 

import numpy as np
import matplotlib.pyplot as plt
import random
import sys


def abc(phi_train,t_train,phi_test,t_test,N,f,l):    
    I=np.identity(f)
    pt=np.transpose(phi_train)
    a=np.matmul(pt,phi_train)
    b=np.matmul(pt,t_train)

    c=np.linalg.inv(np.add(l*I,a))
    w=np.matmul(c,b)
    MSE1=0
    for i in range(0,N):
        MSE1+=(np.subtract(np.matmul(np.transpose(phi_test[i]),w),t_test[i]))**2
    MSE=(1/N)*MSE1
    return MSE


if __name__ == "__main__":

    phi_train=np.loadtxt('train-1000-100.csv',delimiter=',')
    t_train=np.loadtxt('trainR-1000-100.csv',delimiter=',')

    phi_test=np.loadtxt('test-1000-100.csv',delimiter=',')
    t_test=np.loadtxt('testR-1000-100.csv',delimiter=',')
#    l=96
    l=sys.argv[1]
    m=[]
    phi_t=[]
    for j in range(0,10):
        phi_split=[]
        q=[]
        i=10
        while i<=1000:
            index = random.randrange(0,1001-i)
            train=abc(phi_train[index:index+i],t_train[index:index+i],phi_test,t_test,1000,100,l)
            phi_split.append(train)
            q.append(i)
            i+=30         
        phi_t.append(phi_split)
#    print(phi_t)
    phit=np.transpose(phi_t)
#    print(phit)
    for i in range(0,len(q)):
        m.append(np.mean(phit[i]))
    print(m)
    
    plt.title("Lambda="+str(l))
    plt.xlabel("Sample size")
    plt.ylabel("MSE")
    plt.plot(q,m)
    plt.show()


