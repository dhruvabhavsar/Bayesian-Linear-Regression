#!/usr/local/bin/python3
# Submitted by: Dhruva Bhavsar, Username:dbhavsar
# Programming Project 2: 

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys

if __name__ == "__main__":

    phi_train=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/train-wine.csv',delimiter=',')
    t_train=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/trainR-wine.csv',delimiter=',')
    
    
    phi_test=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/test-wine.csv',delimiter=',')
    t_test=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/testR-wine.csv',delimiter=',')
    start_time = time.process_time()
    alpha=random.randint(1,10)
    beta=random.randint(1,10)
#    print(alpha,beta)
    a=0
    b=0
    N=phi_train.shape[0]
    f=phi_train.shape[1]
    while abs(alpha-a)>0.00000000000001 and abs(beta-b)>0.00000000000001:
        a=alpha
        b=beta
        I=np.identity(f)
        phi_t=np.transpose(phi_train)
        phi=np.matmul(phi_t,phi_train)
        t=np.matmul(phi_t,t_train)
        SN=np.linalg.inv(alpha*I+beta*phi)
        MN=beta*np.matmul(SN,t)
        c=beta*phi
        
        ev,v=np.linalg.eig(c)
        
#        print(ev)
        gamma=0
        for i in ev:
#            print(i)
            gamma+=i/(alpha+i)
        
        n=np.matmul(np.transpose(MN),MN)
        alpha=gamma/n
        r=0
        
        for j in range(0,N):
            r+=(np.subtract(t_train[j],np.matmul(np.transpose(MN),phi_train[j])))**2
        s=1/(N-gamma)*r
        beta=1/s
            
    print("Alpha:",alpha)
    print("Beta:",beta)
    print("lambda:",alpha/beta)   
    SN=np.linalg.inv(alpha*I+beta*phi)
    MN=beta*np.matmul(SN,t)

    N=phi_test.shape[0]

    MSE1=0
    for i in range(0,N):
        MSE1+=(np.subtract(np.matmul(np.transpose(phi_test[i]),MN),t_test[i]))**2
    MSE=(1/N)*MSE1
    print("MSE:",MSE)
    print ("Run Time:",time.process_time() - start_time, "seconds")