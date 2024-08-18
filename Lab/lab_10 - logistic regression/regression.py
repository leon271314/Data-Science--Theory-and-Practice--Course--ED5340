import numpy as np
import pandas as pd

class MachineLearning:
    def __init__(self, weight, X_train, y_train, X_test=np.array([]), y_test=np.array([])):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.weight = np.array(weight)
    
    def J_Linear(self,X,Y,W):
        #print(X.shape(),Y.shape(),W.shape())
        data = (np.matmul(W,X)-Y)**2
        return 1/(2*np.shape(X)[1])*np.sum(data)
    def dJ_Linear(self,X,Y,W):
        grad=[]
        for i in range(np.shape(X)[0]):
            print(X.shape(),Y.shape(),W.shape())
            data = np.multiply(np.matmul(W,X)-Y,X[i])
            grad += [sum(data)]
        grad = np.array(grad)
        return 1/np.shape(X)[1]*grad
    
    def LinearRegression(self):
        tolerance = 0.001
        while tolerance < np.linalg.norm(np.array(self.dJ_Linear(self.X_train,self.y_train,self.weight))):
            self.weight = self.weight - 0.1*np.array(self.dJ_Linear(self.X_train,self.y_train,self.weight))
        return self.weight
    
    def J_Logistic_Sigmoid(self,X,Y,W):
        sigmoid = 1/(1+np.e**(-(np.matmul(W,X))))
        data = np.multiply(-Y,np.log(sigmoid)) + np.multiply(-(1-Y),np.log(np.ones(np.shape(Y))-sigmoid))
        return 1/(np.shape((X)[1]))*np.sum(data)
    
    def dJ_Logistic_Sigmoid(self,X,Y,W):
        #print(X.shape(),Y.shape(),W.shape())
        sigmoid = 1/(1+np.e**(-(np.matmul(W,X))))
        grad=[]
        for i in range(np.shape(X)[0]):
            data = np.multiply(sigmoid-Y,X[i])
            grad += [sum(data)]
        grad = np.array(grad)
        return 1/np.shape(X)[1]*grad
    
    def dJ_Linear(self,X,Y,W):
        grad=[]
        for i in range(np.shape(X)[0]):
            data = np.multiply(np.matmul(W,X)-Y,X[i])
            grad += [sum(data)]
        grad = np.array(grad)
        return 1/np.shape(X)[1]*grad   

    def LogisticRegression(self):
        tolerance = 0.001
        while tolerance < np.linalg.norm(np.array(self.dJ_Logistic_Sigmoid(self.X_train,self.y_train,self.weight))):
            self.weight = self.weight - 0.1*np.array(self.dJ_Logistic_Sigmoid(self.X_train,self.y_train,self.weight))
        return self.weight