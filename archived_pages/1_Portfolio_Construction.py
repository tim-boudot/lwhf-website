import streamlit as st
import plotly.express as px
import pandas as pd
import time
import requests
import stqdm

#predit_url = ('https://lwhf-edxf3vliba-ew.a.run.app/predict')
backtest_url = ('https://lwhf4-edxf3vliba-ew.a.run.app/backtest')

st.markdown('''# Let's construct a money-making guaranteed 100% becoming rich in 1 week portfolio''')

options = st.multiselect(
    "Which asset classes would you like to include in your portfolio",
    ["Equities", "Bonds", "Real Estate", "Bitcoin"],
    ["Equities", "Bonds", "Real Estate", "Bitcoin"])

st.write(" ")
st.write(" ")


if st.button("Let's create that portfolio!", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    params = {
        'as_of_date':'2024-05-27',
        'n_periods':3
    }

    req = requests.get(backtest_url, params)
    res = req.json()
    weights = res['final_weights']
    weights = {k:v for k,v in weights.items() if v!=0}

    dicto_3 = {'Stocks':weights.keys(), 'Values':weights.values()}
    fig = px.pie(dicto_3, values='Values', names='Stocks')

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)

    time.sleep(1)
    my_bar.empty()
    st.plotly_chart(fig, use_container_width=True)
