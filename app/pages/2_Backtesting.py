import streamlit as st
import riskfolio as rp
import requests, time
import pandas as pd
import matplotlib.pyplot as plt

st.markdown('''# Backtesting of portfolio performance''')

#predit_url = ('https://lwhf-edxf3vliba-ew.a.run.app/predict')
backtest_url = ('https://lwhf4-edxf3vliba-ew.a.run.app/backtest')


if st.button("Click here!", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):

    params = {
        'as_of_date':'2024-05-27',
        'n_periods':3
    }

    req = requests.get(backtest_url, params)
    res = req.json()
    weights = res['final_weights']
    weights = {k:v for k,v in weights.items() if v!=0}

    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['weights'])

    time.sleep(1)

    fig, ax = plt.subplots()
    rp.plot_pie( w= weights_df, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                            height=6, width=10, ax=ax)
    st.pyplot(fig)
