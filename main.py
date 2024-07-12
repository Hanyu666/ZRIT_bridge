import streamlit as st
import json
import Pages

st.set_page_config(
    page_title="浙江省交通运输科学研究院",
    layout="wide",  # 设置页面布局为宽布局
    initial_sidebar_state="expanded"  # 可选：初始侧边栏状态
)


def main():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        Pages.login_page()

    else:
        col1, col2 = st.columns([7,1])

        with col2:
            if st.button("退出登录"):
                st.session_state.authenticated = False
                st.experimental_rerun()

        with col1:
            st.sidebar.header('功能列表')
            menu = ['数据总览', '技术状况预测', '养护科学决策系统', '报告生成']
            choice = st.sidebar.radio('请选择', menu)
        
        if choice == '数据总览':
            st.title('桥梁总体数据')
            Pages.data_overview()
        if choice == '技术状况预测':
            st.title('性能预测系统')
            Pages.tech_prediction()
        if choice == '养护科学决策系统':
            st.title('养护科学决策系统')
            Pages.maintenance_decision()


if __name__ == "__main__":
    main()