import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
#from .gmm import GMMClassifier

class Predictor():
    def __init__(self):
        self.df_train = None
        self.model_clf = None
        self.model_reg = None
        self.df = None
        self.df_res_class = None
        self.df_res_regr = None

    # Add one hot encoded card columns
    def __convert_request(self, request):
        input = np.array(request).reshape(1,-1)
        self.df = pd.DataFrame(input, columns=["amount", "3D_secured", "transaction_hour", "card"])


    def __one_hot_encoding(self):
        df_dummies = pd.get_dummies(self.df['card'], prefix='card', dtype=float)
        card_columns = ['card_Diners', 'card_Master', 'card_Visa']

        for col in card_columns:
            if col not in df_dummies:
                df_dummies[col] = 0.0

        self.df = pd.concat([self.df, df_dummies], axis=1)


    # Scaling of amount and transaction_hour
    def __scaling(self):
        min_max_scaler = MinMaxScaler()
        amount_train = self.df_train[["amount", "transaction_hour"]]
        min_max_scaler.fit(amount_train)
        self.df[["amount_sc", "transaction_hour_sc"]] = min_max_scaler.transform(self.df[["amount", "transaction_hour"]])


    # Duplicate rows for all 4 PSPs
    def __prepare_psp_records(self):
        # Prediction for all PSPs will be done
        row_to_duplicate = self.df.loc[0]
        new_rows = pd.DataFrame([row_to_duplicate] * 3, columns=self.df.columns) 
        self.df = pd.concat([self.df, new_rows], ignore_index=True)

        self.df['PSP_Goldcard'] = 0.0
        self.df['PSP_Moneycard'] = 0.0
        self.df['PSP_Simplecard'] = 0.0
        self.df['PSP_UK_Card'] = 0.0
 
        self.df.loc[0, 'PSP_Goldcard'] = 1.0
        self.df.loc[1, 'PSP_Moneycard'] = 1.0
        self.df.loc[2, 'PSP_Simplecard'] = 1.0
        self.df.loc[3, 'PSP_UK_Card'] = 1.0


    def __get_top_n_records(self, X, y, column, n_top, ascending=True):
        df = X.copy()
        df[column] = y
        df = df.sort_values(by=column, ascending=ascending)
        return df.head(n_top)
    
    def __load_training_data(self, file_path_df_train):
        self.df_train = pd.read_csv(file_path_df_train, index_col=0)
        self.df_train['tmsp'] = pd.to_datetime(self.df_train['tmsp'])
        self.df_train['transaction_hour'] = self.df_train['tmsp'].dt.hour


    def __prepare_request_for_prediction(self, request):
        self.__convert_request(request)
        self.__one_hot_encoding()
        self.__scaling()
        self.__prepare_psp_records()

        # 3D_secured has type object? why? Just cast it
        self.df['3D_secured'] = pd.to_numeric(self.df_train['3D_secured'])

        return self.df[['amount_sc', '3D_secured', 'transaction_hour_sc', 'card_Diners', 'card_Master', 'card_Visa', 'PSP_Goldcard', 'PSP_Moneycard', 'PSP_Simplecard', 'PSP_UK_Card']]
    

    def load_models(self, mdl_path_class, mdl_path_regr, file_path_df_train):
        self.model_clf = pickle.load(open(mdl_path_class, 'rb'))
        self.model_reg = pickle.load(open(mdl_path_regr, 'rb'))
        self.__load_training_data(file_path_df_train)


    def predict_top_n_success(self, request, n_top=2):
        X_class = self.__prepare_request_for_prediction(request)
        y_class = self.model_clf.predict_proba(X_class)[:, 1]

        self.df_result_class = self.__get_top_n_records(X_class, y_class, "success_max", n_top, False)
        return self.df_result_class
    

    def predict_top_n_fee(self, n_top = 1):
        X_reg = self.df_result_class[["PSP_UK_Card", "PSP_Goldcard", "PSP_Simplecard", "PSP_Moneycard", "success_max", "card_Visa"]]
        y_reg = self.model_reg.predict(X_reg)

        self.df_result_regr = self.__get_top_n_records(X_reg, y_reg, "fee", n_top)
        return self.df_result_regr
    
    
    def get_predicted_psp(self):
        return (self.df_result_regr == 1).idxmax(axis=1).values[0]
