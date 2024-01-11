import os
import sys
import boto3
import dill
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from src.exception import CustomException

def export_collection_as_dataframe(collection_name,db_name):
    try:
        mongo_client = MongoClient(os.getenv('MONGO_DB_URL'))
        collection = mongo_client[db_name][collection_name]
        df = pd.DataFrame(list(collection.find()))
        if '_id' in df.columns.tolist():
            df = df.drop(columns=['_id'],axis=1)
        df.replace({"na",np.nan},inplace=True)
        return df 
       
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(dir_path,'wb') as obj_file:
            dill.dump(obj,obj_file)
        
    except Exception as e:
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as obj_file:
            return dill.load(obj_file)
        
    except Exception as e:
        raise CustomException(e,sys)
