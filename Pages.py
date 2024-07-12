import streamlit as st
import pandas as pd
import json
import plotly.express as px
import pickle
import numpy as np
from matplotlib import pyplot as plt
import utils
import decision

def authenticate(username, password):
    '''
    定义认证函数，匹配账号和密码
    返回True / False
    '''
    with open('users.json', 'r') as f:
        users = json.load(f)
    return username in users and users[username] == password


def login_page():
    # st.title('浙江省交通运输科学研究院')
    # st.header('公路桥梁性能评估与管养科学决策系统', divider='rainbow')
    st.image('./Fig/Login_title.png')

    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    if st.button("登录"):
        # 从 JSON 文件中读取账号密码信息
        if authenticate(username, password):
            st.session_state.authenticated = True
            st.rerun()              # 当session_state 由Flase变为True了，rerun整个程序？
        else:
            st.error("用户名或密码错误")

def data_overview():
    st.header("数据总览")
    df = pd.DataFrame({
        '桥梁类型': ['拱桥', '空心板桥', 'T梁桥','悬索桥', '斜拉桥'],
        '数量': [50, 100, 12, 20, 15]
    })
    fig = px.bar(df, x='桥梁类型', y='数量', title='桥梁类型分布')
    st.plotly_chart(fig)


def tech_prediction():
    st.subheader('预测桥梁数据上传')
    input_data = utils.upload_input_data()
    if input_data is not None:
        st.subheader('数据基本概况')

        cols = st.multiselect('请选择需要展示的字段：', input_data.columns)
        if len(cols) == 0:
            st.stop()
        tabs = st.tabs(cols)
        for tab, col in zip(tabs, cols):
            with tab:
                col1, col2 = st.columns([5,1])
                with col1:
                    fig2 = px.scatter(input_data[col])
                    st.plotly_chart(fig2)
                with col2:
                    st.dataframe(input_data[col])

        st.subheader('技术评分预测结果')
        
        # 预测结果
        select_prediction_model = st.selectbox('请选择需要使用的机器学习模型',
                                               ['LR','LASSO','EN','AdaBoost','RF','MLP','GBRT','GPR','SVR'])
        if select_prediction_model is None:
            st.stop()
        with st.container():
            st.plotly_chart(utils.ML_prediction(select_prediction_model, input_data))

def maintenance_decision():
    st.subheader('养护决策系统')
    search = decision.Search()
    search.gamma = st.slider('提示：请选择风险偏好γ，-1为风险追求，1为风险规避',-1.000, 1.000, 1.000, step=0.001)
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            upper_score = st.number_input('上部结构当前评分:', min_value=0, max_value=100)
        with col2:
            down_score = st.number_input('下部结构当前评分:', min_value=0, max_value=100)
        with col3:
            deck_score = st.number_input('桥面系当前评分:', min_value=0, max_value=100)

        search.tech_origin = [upper_score, down_score, deck_score]
    with st.container():
        # col4, col5, col6 = st.columns(3)
        # with col4:
        #     ei1 = st.number_input('综合权重1:', min_value=0, max_value=1,step=100)
        # with col5:
        #     ei2 = st.number_input('综合权重2:', min_value=0, max_value=1)
        # with col6:
        #     st.write('权重3自动计算为：')

        search.ei =  np.array([[0.357615, 0.317178, 0.325207]])

    with st.container():
        st.subheader('养护方案上传Excel')
        scheme_file = st.file_uploader('请上传方案Excel文件', type=['xlsx'])
        if scheme_file is not None:
            parts = ['上部结构','下部结构','桥面系']
            with st.container():
                search.scheme_up = decision.scheme()
                search.scheme_down = decision.scheme()
                search.scheme_deck = decision.scheme()
                scheme_up_sum = pd.read_excel(scheme_file,sheet_name=parts[0])
                scheme_down_sum = pd.read_excel(scheme_file,sheet_name=parts[1])
                scheme_deck_sum = pd.read_excel(scheme_file,sheet_name=parts[2])
                search.scheme_up.tech = scheme_up_sum.iloc[:,0]
                search.scheme_up.cost = scheme_up_sum.iloc[:,1]
                search.scheme_up.evan = scheme_up_sum.iloc[:,2]
                search.scheme_down.tech = scheme_down_sum.iloc[:,0]
                search.scheme_down.cost = scheme_down_sum.iloc[:,1]
                search.scheme_down.evan = scheme_down_sum.iloc[:,2]
                search.scheme_deck.tech = scheme_deck_sum.iloc[:,0]
                search.scheme_deck.cost = scheme_deck_sum.iloc[:,1]
                search.scheme_deck.evan = scheme_deck_sum.iloc[:,2]

            with st.expander('点击可展开详细数据列表'):
                tabs = st.tabs(parts)
                for tab, part in zip(tabs, parts):
                    with tab:
                        df = pd.read_excel(scheme_file,sheet_name=part)
                        st.dataframe(df)

        else:
            st.write('请上传文件')
            st.stop()
    
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        button = st.button('计算决策结果')
    if button:
        with st.container():
            U_uti_cost, uti, index, idx_left = search.U_calculate()
            col1, col2, col3 = st.columns([1,3,1])
            with col2:
                fig = px.scatter(x=U_uti_cost[:,0],y=U_uti_cost[:,1])
                st.plotly_chart(fig)
            # U_uti_cost = pd.DataFrame(U_uti_cost, columns=['U','cost'])
            # if U_uti_cost:
            #     fig = plt.scatter(U_uti_cost['U'], U_uti_cost['cost'])
            #     st.plotly_chart(fig)


    