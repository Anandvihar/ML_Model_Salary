# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 15:09:02 2021

@author: Anandvihar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle 


dataset  = pd.read_csv(r"C:\Users\Anandvihar\Desktop\flask\hiring.csv")
dataset.describe()

dataset['experience'].fillna(0,inplace=True)

dataset['test_score'].fillna(dataset['test_score'].mean(),inplace=True)

X = dataset.iloc[:, :3]

print(X)

# convert word to integer value 

def convert_to_int(word):
    word_dict = {'one': 1, 'two':2,'three':3, 'four':4, 'five':5, 'six':6, 'seven': 7, 
                 'eight':8 , 'nine':9,'ten':10,'eleven':11,'twelve':12, 'zero':0, 0:0}

    return word_dict[word]   

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

y = dataset.iloc[:, -1]

print(X)
    
    
# slipting the data into Training and Testing set
   
    
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(())
    

# Fitting the model with Training dataset 

regressor.fit(X,y)

#saving the model to disk

pickle.dump(regressor,open('model.pkl','wb'))


#Loading the model to compare the result 

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))
    
    
    
    