import pandas as pd 
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB

@dataclass
class Model_Trainer_Config:
    pass

@dataclass
class ModelTrainer:
    def __init__(self):
        pass
    
    def initiate_model_trainer(self,train_arr,test_arr,preprocessor_path):
        
        # Divide dataset into: X_train,X_test,y_train,y_test
        
        # Models Dictionary, containing all the candidate models
        
        # Training of models, the Model with best R2 Score on the testing data will be selected for Prediction Pipeline
        
        # Return R2 Score of the best model
        
        pass
    
    
        
