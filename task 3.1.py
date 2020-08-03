#!/usr/local/bin/python3
# Submitted by: Dhruva Bhavsar, Username:dbhavsar
# Programming Project 2: 

import numpy as np
import matplotlib.pyplot as plt
import random
import time

def abc(phi,t,phi_t,t_t,N,f):
    
    I=np.identity(f)
    pt=np.transpose(phi)
    a=np.matmul(pt,phi)
    b=np.matmul(pt,t)
    M=[]
    
    for l in range(0,150):
        c=np.linalg.inv(np.add(l*I,a))
        w=np.matmul(c,b)
        MSE1=0
        for i in range(0,N):
            MSE1+=(np.subtract(np.matmul(np.transpose(phi_t[i]),w),t_t[i]))**2
        MSE=(1/N)*MSE1
#        print(MSE)
        M.append([l,MSE])
#    S=np.array(M) 
    return M

def abcd(phi_train,t_train,phi_test,t_test,N,f,l):    
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


def cross_validation_split(dataset, folds):
    dataset_split = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    d_split=np.array(dataset_split)
    return d_split

if __name__ == "__main__":
    

    

    phi_train=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/train-1000-100.csv',delimiter=',')
    t_train=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/trainR-1000-100.csv',delimiter=',')
    
    phi_test=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/test-1000-100.csv',delimiter=',')
    t_test=np.loadtxt('C:/Users/dhruv/Desktop/Textbooks/Machine Learning/pp2data/testR-1000-100.csv',delimiter=',')
    
    start_time = time.process_time()
    split=cross_validation_split(phi_train,10)
    splitR=cross_validation_split(t_train,10)
 
    b=[]

    for j in range(0,10):
        testset=[]
        testsetR=[]
        trainset=[]
        trainsetR=[]
        a=[]
        for i in range(0,10):
            if(i!=j):
                trainset.extend(split[i])
                trainsetR.extend(splitR[i])
            else:
                testset.extend(split[i])
                testsetR.extend(splitR[i])  
        for l in range(0,150):
            train=abcd(trainset,trainsetR,testset,testsetR,int(phi_train.shape[0]/10),phi_train.shape[1],l)
            a.append([l,train])
        b.append(a)
#    print(b)
#    c=np.array(b)
    d=np.transpose(b)
#    print(d.shape)
    dict={}
    for k in range(0,150):
        dict[d[0][k][0]]=np.average(d[1][k])
#    print(dict)
    l=min(dict.keys(), key=(lambda k: dict[k]))
    print("Lambda:",l)
    print("MSE:",abcd(phi_train,t_train,phi_test,t_test,phi_test.shape[0],phi_test.shape[1],l))
    print ("Run Time:",time.process_time() - start_time, "seconds")


