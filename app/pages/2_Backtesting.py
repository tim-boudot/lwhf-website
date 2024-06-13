import streamlit as st
import riskfolio as rp
import requests, time, datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, json

import matplotlib.pyplot as plt

api_cache_folder = 'api_cache'

def get_total_return(weekly_returns):
    total_return = 1
    for weekly_return in weekly_returns:
        total_return *= (1 + weekly_return)
    return total_return - 1

st.markdown('''# Backtesting of portfolio performance''')

#predit_url = ('https://lwhf-edxf3vliba-ew.a.run.app/predict')
#backtest_url = ('https://lwhf4-edxf3vliba-ew.a.run.app/backtest')
#api_url = ('http://lwhf5-edxf3vliba-ew.a.run.app')
api_url = ('https://lwhf6-edxf3vliba-ew.a.run.app')


# Date Selector
default_date = datetime.datetime.strptime('2024-05-27', "%Y-%m-%d").date()
selected_date = st.date_input("Please select the end of the backtesting period", default_date)
as_of_date_str = selected_date.strftime("%Y-%m-%d")
st.write("You selected:", selected_date)

# Number of Periods Selector
num_periods = st.number_input("Please select the number of backtesting periods", min_value=0, max_value=100, value=3, step=1)
st.write("You selected:", num_periods)

if st.button("See Backtest Returns", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):

    end_point = 'backtest'
    params = {
        'as_of_date':as_of_date_str,
        'n_periods':num_periods
    }


    full_path = os.path.join(api_cache_folder, end_point)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    filename = f'{as_of_date_str}-{num_periods}.json'
    json_full_path = os.path.join(full_path, filename)
    if os.path.exists(json_full_path):
        print(f'✅ Found {filename} in the local cache.')
        # read the json file as a dictionary
        with open(json_full_path, 'r') as file:
            result = json.load(file)
    else:
        url = '/'.join([api_url, end_point])
        req = requests.get(url, params)
        result = req.json()
    with open(json_full_path, 'w') as f:
        json.dump(result, f)

    st.session_state['result'] = result


    ## Plot Portfolio returns vs Market returns
    portfolio_returns = result['portfolio_returns']
    market_returns = result['market_returns']

    cumulative_portfolio_returns = [0]
    cumulative_market_returns = [0]
    for i in range(1, len(portfolio_returns)):
        cumulative_portfolio_returns.append(get_total_return(portfolio_returns[:i]))
        cumulative_market_returns.append(get_total_return(market_returns[:i]))

    as_of = datetime.datetime.strptime(as_of_date_str, '%Y-%m-%d').date()
    dates = [as_of]
    for i in range(1,num_periods):
        dates.append(as_of - datetime.timedelta(days= 7*i))
    dates = dates[::-1]

    df = pd.DataFrame({'Date':dates, 'Portfolio Returns':cumulative_portfolio_returns, 'Market Returns':cumulative_market_returns})
    df = df.set_index('Date')

    # Plotting
    # fig, ax = plt.subplots()
    # df.plot(ax=ax)
    # plt.xlabel('Date')
    # plt.ylabel('Returns')
    # plt.title('Portfolio vs Market Returns')
    # st.pyplot(fig)
        # Plotting using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=cumulative_portfolio_returns, mode='lines', name='Portfolio Returns'))
    fig.add_trace(go.Scatter(x=dates, y=cumulative_market_returns, mode='lines', name='Market Returns'))
    fig.update_layout(
        title='Portfolio vs Market Returns',
        xaxis_title='Date',
        yaxis_title='Returns'
    )
    st.plotly_chart(fig, use_container_width=True)


if st.button("See Final Weights", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):
    # Plot weights
    result = st.session_state['result']
    weights = result['final_weights']
    weights = {k:v for k,v in weights.items() if abs(v) > 1e-7}
    if len(weights) > 5:
        sorted_weights = dict(sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True))
        top_5_weights = dict(list(sorted_weights.items())[:5])
        other_weight = sum(list(sorted_weights.values())[5:])
        top_5_weights['Other'] = other_weight
    else:
        top_5_weights = weights
    #my_dict = {'Stocks': weights.keys(), 'Values': weights.values()}
    my_dict = {'Stocks': top_5_weights.keys(), 'Values': top_5_weights.values()}
    fig = px.pie(my_dict, values='Values', names='Stocks', title='Portfolio Weights')
    st.plotly_chart(fig, use_container_width=True)

    st.session_state['top_5_weights'] = weights


# Function to handle button click
def inspect_stocks():
    st.session_state['inspect_button_pressed'] = True

# Display the button and handle click
if st.button("Inspect Stocks", on_click=inspect_stocks):
    st.session_state['inspect_button_pressed'] = True

# If the button has been pressed, display the selectbox
if st.session_state.get('inspect_button_pressed', False):
    top_5_weights = st.session_state['top_5_weights']
    option = st.selectbox(
        "Which stock would you like to inspect?",
        top_5_weights.keys(),
        index=None,
        placeholder="Select a stock...",
    )
    st.session_state['selected_stock'] = option

# If a stock has been selected, display the selected stock
if 'selected_stock' in st.session_state:
    st.write("You selected:", st.session_state['selected_stock'])

    prices = pd.read_csv('data/all_close_prices.csv')
    prices['timestamp'] = pd.to_datetime(prices['timestamp'])
    prices = prices.set_index('timestamp')
    option = st.session_state['selected_stock']

    print(prices.shape)
    print(option)

    st.session_state['prices'] = prices

    # Plot stock prices
    stock_prices = prices[option]
    dates = prices.index

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=stock_prices, mode='lines', name=option))
    fig.update_layout(
        title=f'Price of {option}',
        xaxis_title='Date',
        yaxis_title='Close Price'
    )
    st.plotly_chart(fig, use_container_width=True)



# if st.button("Inspect Stocks", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):
#     top_5_weights = st.session_state['top_5_weights']
#     option = st.selectbox(
#     "Which stock would you like to inspect?",
#     top_5_weights.keys(),
#     index=None,
#     placeholder="Select a stock...",
#     )
#     st.write("You selected:", option)

    # request stock prices
    # params = {
    #     'as_of_date':as_of_date_str
    # }
    # end_point = 'all_close_prices'
    # url = '/'.join([api_url, end_point])
    # prices = requests.get(url, params)
    # print(type(prices))

    # prices = pd.read_csv('data/all_close_prices.csv')
    # prices['timestamp'] = pd.to_datetime(prices['timestamp'])
    # prices = prices.set_index('timestamp')

    # print(prices)

    # st.session_state['prices'] = prices

    # # Plot stock prices
    # stock_prices = prices[option]
    # dates = prices.index

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=dates, y=stock_prices, mode='lines', name=option))
    # fig.update_layout(
    #     title=f'Price of {option}',
    #     xaxis_title='Date',
    #     yaxis_title='Close Price'
    # )
    # st.plotly_chart(fig, use_container_width=True)





    #st.dataframe(prices)






    # my_dict = {'Stocks':weights.keys(), 'Values':weights.values()}
    # fig = px.pie(my_dict, values='Values', names='Stocks')

    # st.plotly_chart(fig, use_container_width=True)

#if st.button("See Final Weights", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):
