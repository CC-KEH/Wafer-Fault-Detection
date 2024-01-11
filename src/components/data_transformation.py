import pandas as pd 
import numpy as np
from dataclasses import dataclass
import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
@dataclass
class Data_Transformation_Config:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = Data_Transformation_Config()
        
    def get_data_transformation_obj(self):
        
        # Function to replace 'na' to np.nan
        replace_na_with_nan = lambda X: np.where(X == 'na',np.nan, X)
        
        # Steps for Preprocessing pipeline
        nan_replacement = ('nan_replacement',FunctionTransformer(replace_na_with_nan))
        imputer = ('Imputer',SimpleImputer(strategy='constant',fill_value=0))
        scaler = ('Scaler',RobustScaler())
        
        preprocessor = Pipeline(steps=[nan_replacement,imputer,scaler])
        
        # Return Pipeline Object
        return preprocessor
        
                
    def initiate_transformation(self,train_path,test_path):
        try:
            # Take Training and Testing file path
            df_train = pd.read_csv(train_path)    
            df_test = pd.read_csv(test_path)
            preprocessor = self.get_data_transformation_obj()
            
            # Removing Labels from the Dataset
            target_col_name = "class"
            target_col_mapping = {'+1':0, '-1':1}
            input_features_df_train = df_train.drop(columns=[target_col_name],axis=1)
            target_feature_df_train = df_train[target_col_name].map(target_col_mapping)
            
            input_features_df_test = df_test.drop(columns=[target_col_name],axis=1)
            target_feature_df_test = df_test[target_col_name].map(target_col_mapping)

            # Apply Preprocessor Pipeline on the dataset
            
            transformed_input_train_features = preprocessor.fit_transform(input_features_df_train)
            transformed_input_test_features = preprocessor.transform(input_features_df_test)

            smt = SMOTETomek(sampling_strategy = 'minority')

            input_features_df_train_final,target_feature_df_train_final = smt.fit_resample(transformed_input_train_features,target_feature_df_train)
            input_features_df_test_final,target_feature_df_test_final = smt.fit_resample(transformed_input_test_features,target_feature_df_test)

            # Return the tranformed data in form of a list, i.e. Training List, Testing List, Preprocessor Path
            train_arr = np.c_[input_features_df_train_final,np.array(target_feature_df_train_final)]
            test_arr = np.c_[input_features_df_test_final,np.array(target_feature_df_test_final)]
            
            return(train_arr,
                   test_arr,
                   self.transformation_config.preprocessor_obj_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)
    