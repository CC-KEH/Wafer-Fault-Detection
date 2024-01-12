from flask import Flask,request,render_template,jsonify,send_file
from src.exception import CustomException
from src.logger import logging
from src.pipelines.training_pipeline import TrainPipeline
from src.pipelines.prediction_pipeline import PredictionPipeline
import os,sys

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify('home')

@app.route('/train')
def train():
    try:
        train_pipe = TrainPipeline()
        train_pipe.start_pipeline()
        return ("Training Complete")
        
    except Exception as e:
        raise CustomException(e,sys)    

@app.route('/predict',methods=['POST','GET'])
def predict():
    try:
        if request.method == "POST":        
            pred_pipe = PredictionPipeline(request)
            pred_file_detail = pred_pipe.run_pipeline()
            
            logging.info('Prediction Complete, Downloading Prediction File')
            return send_file(pred_file_detail.prediction_file_path,download_name=pred_file_detail.prediction_file_name,as_attachment=True)
        
        else:
            return render_template('predict.html')
    except Exception as e:
        raise CustomException(e,sys)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=9000,debug=True)