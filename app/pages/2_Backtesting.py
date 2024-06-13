import streamlit as st
import riskfolio as rp
import requests, time, datetime
import pandas as pd
import plotly.express as px

import matplotlib.pyplot as plt

st.markdown('''# Backtesting of portfolio performance''')

#predit_url = ('https://lwhf-edxf3vliba-ew.a.run.app/predict')
backtest_url = ('https://lwhf4-edxf3vliba-ew.a.run.app/backtest')


# Date Selector
default_date = datetime.datetime.strptime('2024-05-27', "%Y-%m-%d").date()
selected_date = st.date_input("Please select the end of the backtesting period", default_date)
selected_date_str = selected_date.strftime("%Y-%m-%d")
st.write("You selected:", selected_date)

# Number of Periods Selector
selected_integer = st.number_input("Please select the number of backtesting periods", min_value=0, max_value=100, value=3, step=1)
st.write("You selected:", selected_integer)

if st.button("Click here!", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):
    params = {
        'as_of_date':selected_date_str,
        'n_periods':selected_integer
    }

    req = requests.get(backtest_url, params)
    res = req.json()
    weights = res['final_weights']
    weights = {k:v for k,v in weights.items() if abs(v) > 1e-10}

    my_dict = {'Stocks':weights.keys(), 'Values':weights.values()}
    fig = px.pie(my_dict, values='Values', names='Stocks')

    st.plotly_chart(fig, use_container_width=True)
