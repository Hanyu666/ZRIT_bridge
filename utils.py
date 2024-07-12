import streamlit as st
import pandas as pd
import json
import plotly.express as px
import numpy as np
import pickle
import os

def upload_input_data(sheet_name='Sheet1'):
    file = st.file_uploader(f'请选择Excel文件', type=['xlsx'])
    if file is not None:
        st.write('文件名称:', file.name)
        file = pd.read_excel(file, sheet_name=sheet_name)
    else:
        st.write('上传文件后将显示文件内容')
    return file

def ML_prediction(model_select, input_data):
    with open('./predictionModel/ss_X.pkl', 'rb') as f:
        ss_X = pickle.load(f)
    with open('./predictionModel/ss_Y.pkl', 'rb') as f:
        ss_Y = pickle.load(f)
    path = os.path.join('.\predictionModel', 'model_'+model_select+'.pkl')
    with open(path, 'rb') as f_model:
        ML_model = pickle.load(f_model)

    
    tech_prediction = ML_model.predict(ss_X.transform(input_data))
    tech_prediction = ss_Y.inverse_transform(tech_prediction.reshape(-1,1))
    tech_prediction = pd.DataFrame(tech_prediction)
    tech_prediction.index = input_data['桥龄']/365
    fig3 = px.scatter(tech_prediction)
    
    return fig3
        