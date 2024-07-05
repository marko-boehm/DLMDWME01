import sys
sys.path.append('../')
import pandas as pd

from flask import Flask, request, jsonify
from lib.prediction import Predictor
from lib.gmm import GMMClassifier

data_path = '../01_Data/'
file_name_cleaned_data = '3-1_psp-data_cleaned.csv'
mdl_name_class = '4-3-1_psp-class-model.pkl'
mdl_name_regr = '4-3-1_psp-regr-model.pkl'

file_path_cleaned_data = data_path + file_name_cleaned_data
file_path_mdl_class = data_path + mdl_name_class
file_path_mdl_regr = data_path + mdl_name_regr

app = Flask(__name__)
app.json.sort_keys = False


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    pred = Predictor()
    pred.load_models(file_path_mdl_class, file_path_mdl_regr, file_path_cleaned_data)

    pred.predict_top_n_success(data['input'])
    pred.predict_top_n_fee()

    # Remove 'PSP_' from the 
    return jsonify({'psp': pred.get_predicted_psp(), 'success': round(pred.df_res_class.loc[0, 'success_max'], 2), 'fee': round(pred.df_res_regr.loc[0, 'fee'] ,2)})


if __name__ == '__main__':
    app.run(debug=True)