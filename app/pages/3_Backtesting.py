import streamlit as st
import riskfolio as rp
import requests, time, datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import os, json

import matplotlib.pyplot as plt

MAX_WEIGHTS_TO_DISPLAY = 10
WEIGHT_TOLERANCE = 1e-7
api_cache_folder = 'api_cache_7'
DEFAULT_NUM_PERIODS = 15

#predit_url = ('https://lwhf-edxf3vliba-ew.a.run.app/predict')
#backtest_url = ('https://lwhf4-edxf3vliba-ew.a.run.app/backtest')
#api_url = ('http://lwhf5-edxf3vliba-ew.a.run.app')
#api_url = ('https://lwhf6-edxf3vliba-ew.a.run.app')
api_url = ('https://lwhf7-edxf3vliba-ew.a.run.app')

def get_total_return(weekly_returns):
    total_return = 1
    for weekly_return in weekly_returns:
        total_return *= (1 + weekly_return)
    return total_return - 1


def get_top_n_weights(weights, n=MAX_WEIGHTS_TO_DISPLAY, tolerance=WEIGHT_TOLERANCE):
    filtered_weights = {k: v for k, v in weights.items() if abs(v) > WEIGHT_TOLERANCE}

    if len(filtered_weights) > n:
        sorted_weights = dict(sorted(filtered_weights.items(), key=lambda item: abs(item[1]), reverse=True))
        top_n_weights = dict(list(sorted_weights.items())[:n])
        other_weight = sum(list(sorted_weights.values())[n:])
        top_n_weights['Other'] = other_weight
    else:
        top_n_weights = filtered_weights

    return top_n_weights

def get_aggregate_weights(weekly_weights):
    num_weeks = len(weekly_weights)
    adjusted_weekly_weights = []
    for week_weights in weekly_weights:
        adjusted_week = {k:v/num_weeks for k,v in week_weights.items()}
        adjusted_weekly_weights.append(adjusted_week)

    aggregate_weights = {}
    for week_weights in adjusted_weekly_weights:
        for k,v in week_weights.items():
            if k in aggregate_weights:
                aggregate_weights[k] += v
            else:
                aggregate_weights[k] = v

    return aggregate_weights



st.markdown('''# Backtesting of portfolio performance''')

# Date Selector
default_date = datetime.datetime.strptime('2024-05-27', "%Y-%m-%d").date()
selected_date = st.date_input("Please select the end of the backtesting period", default_date)
as_of_date_str = selected_date.strftime("%Y-%m-%d")
st.write("You selected:", selected_date)

# Number of Periods Selector
num_periods = st.number_input("Please select the number of backtesting periods", min_value=0, max_value=100, value=DEFAULT_NUM_PERIODS, step=1)
st.write("You selected:", num_periods)

# Function to handle button click
def inspect_stocks():
    st.session_state['inspect_button_pressed'] = True


if "See Backtest Returns" not in st.session_state:
    st.session_state["See Backtest Returns"] = False

if "See Final Weights" not in st.session_state:
    st.session_state["See Final Weights"] = False

if "Inspect Stocks" not in st.session_state:
    st.session_state["Inspect Stocks"] = False


if st.button("See Backtest Returns", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):

    st.session_state["See Backtest Returns"] = not st.session_state["See Backtest Returns"]

    with st.status("Building Portfolio...", expanded=True) as status:


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
            st.write("Pre-trained model found. Model predicting...")
            #progress_bar = st.progress(0)
            # for percent_complete in range(100):
            #     time.sleep(0.001)  # Adjust the sleep time to make the progress bar fill up in 3 seconds
            #     progress_bar.progress(percent_complete + 1)

            # read the json file as a dictionary
            with open(json_full_path, 'r') as file:
                result = json.load(file)
        else:
            st.warning("Model training will take approximately 3 minutes. Please wait...")
            st.write("Downloading data...")
            time.sleep(2)
            st.write("Training model...")
            url = '/'.join([api_url, end_point])
            req = requests.get(url, params)
            result = req.json()
            with open(json_full_path, 'w') as f:
                json.dump(result, f)

        time.sleep(3)
        st.write("Running Backtest...")
        time.sleep(3)
        status.update(label='Backtest done!', state='complete', expanded=False)

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


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=cumulative_portfolio_returns, mode='lines', name='Portfolio Returns'))
    fig.add_trace(go.Scatter(x=dates, y=cumulative_market_returns, mode='lines', name='Market Returns'))
    fig.update_layout(
        title='Portfolio vs Market Returns',
        xaxis_title='Date',
        yaxis_title='Returns'
    )
    st.plotly_chart(fig, use_container_width=True)

if st.session_state["See Backtest Returns"]:
    if st.button("See Final Weights", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):
        st.session_state["See Final Weights"] = not st.session_state["See Final Weights"]
        # Plot weights
        # result = st.session_state['result']
        # weights = result['final_weights']
        # weights = {k:v for k,v in weights.items() if abs(v) > 1e-7}
        # if len(weights) > 5:
        #     sorted_weights = dict(sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True))
        #     top_5_weights = dict(list(sorted_weights.items())[:5])
        #     other_weight = sum(list(sorted_weights.values())[5:])
        #     top_5_weights['Other'] = other_weight
        # else:
        #     top_5_weights = weights
        # #my_dict = {'Stocks': weights.keys(), 'Values': weights.values()}
        # my_dict = {'Stocks': top_5_weights.keys(), 'Values': top_5_weights.values()}
        # fig = px.pie(my_dict, values='Values', names='Stocks', title='Portfolio Weights')
        # st.plotly_chart(fig, use_container_width=True)

        # st.session_state['top_5_weights'] = weights
            # Retrieve weekly weights
        result = st.session_state['result']
        weekly_weights = result['weekly_weights']

        ## Aggregate weights
        aggregate_weights = get_aggregate_weights(weekly_weights)
        top_n_aggregate_weights = get_top_n_weights(aggregate_weights)
        st.session_state['top_n_weights'] = top_n_aggregate_weights

        # Plot aggregate weights
        my_dict_agg = {'Stocks': list(top_n_aggregate_weights.keys()), 'Values': list(top_n_aggregate_weights.values())}
        fig_agg = go.Figure(data=[go.Pie(labels=my_dict_agg['Stocks'], values=my_dict_agg['Values'], showlegend=True)])
        fig_agg.update_layout(title_text="Aggregate Portfolio Weights", height=600)
        st.plotly_chart(fig_agg, use_container_width=True)

        # Create a figure with subplots for weekly weights
        num_weeks = len(weekly_weights)
        num_cols = 3
        num_rows = (num_weeks + num_cols - 1) // num_cols  # Calculate the number of rows needed
        fig = sp.make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=[f'Week {i+1}' for i in range(num_weeks)],
            specs=[[{'type': 'domain'}] * num_cols for _ in range(num_rows)]
        )

        # Iterate over weekly weights and plot each week's weights
        for i, weights in enumerate(weekly_weights):
            top_n_weights = get_top_n_weights(weights)
            my_dict = {'Stocks': list(top_n_weights.keys()), 'Values': list(top_n_weights.values())}
            row = (i // num_cols) + 1
            col = (i % num_cols) + 1
            fig.add_trace(go.Pie(labels=my_dict['Stocks'], values=my_dict['Values'], showlegend=True), row=row, col=col)

        # Update layout
        fig.update_layout(title_text="Weekly Portfolio Weights", height=300 * num_rows)
        st.plotly_chart(fig, use_container_width=True)

if st.session_state["See Backtest Returns"] and st.session_state["See Final Weights"]:
# Display the button and handle click
    if st.button("Inspect Stocks", on_click=inspect_stocks):
        st.session_state["Inspect Stocks"] = not st.session_state["Inspect Stocks"]
        st.session_state['inspect_button_pressed'] = True

    # If the button has been pressed, display the selectbox
    if st.session_state.get('inspect_button_pressed', False):
        top_n_weights = st.session_state['top_n_weights']
        option = st.selectbox(
            "Which stock would you like to inspect?",
            top_n_weights.keys(),
            index=0,
            placeholder="Select a stock...",
        )
        st.session_state['selected_stock'] = option

    # If a stock has been selected, display the selected stock
    if 'selected_stock' in st.session_state:
        # st.write("You selected:", st.session_state['selected_stock'])

        # prices = pd.read_csv('data/all_close_prices.csv')
        # prices['timestamp'] = pd.to_datetime(prices['timestamp'])
        # prices = prices.set_index('timestamp')
        # option = st.session_state['selected_stock']

        # print(prices.shape)
        # print(option)

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

def rain(
    emoji: str,
    font_size: int = 64,
    falling_speed: int = 5,
    animation_length: str = "infinite",
):
    """
    Creates a CSS animation where input emoji falls from top to bottom of the screen.

    Args:
        emoji (str): Emoji
        font_size (int, optional): Font size. Defaults to 64.
        falling_speed (int, optional): Speed at which the emoji 'falls'. Defaults to 5.
        animation_length (Union[int, str], optional): Length of the animation. Defaults to "infinite".
    """

    if isinstance(animation_length, int):
        animation_length = f"{animation_length}"

    st.write(
        f"""
    <style>

    body {{
    background: gray;
    }}

    .emoji {{
    color: #777;
    font-size: {font_size}px;
    font-family: Arial;
    // text-shadow: 0 0 5px #000;
    }}

    ///*delete for no hover-effect*/
    //.emoji:hover {{
    //  font-size: 60px;
    //  text-shadow: 5px 5px 5px white;
    //}}

    @-webkit-keyframes emojis-fall {{
    0% {{
        top: -10%;
    }}
    100% {{
        top: 100%;
    }}
    }}
    @-webkit-keyframes emojis-shake {{
    0% {{
        -webkit-transform: translateX(0px);
        transform: translateX(0px);
    }}
    50% {{
        -webkit-transform: translateX(20px);
        transform: translateX(20px);
    }}
    100% {{
        -webkit-transform: translateX(0px);
        transform: translateX(0px);
    }}
    }}
    @keyframes emojis-fall {{
    0% {{
        top: -10%;
    }}
    100% {{
        top: 100%;
    }}
    }}
    @keyframes emojis-shake {{
    0% {{
        transform: translateX(0px);
    }}
    25% {{
        transform: translateX(15px);
    }}
    50% {{
        transform: translateX(-15px);
    }}
    100% {{
        transform: translateX(0px);
    }}
    }}

    .emoji {{
    position: fixed;
    top: -10%;
    z-index: 99999;
    -webkit-user-select: none;
    -moz-user-select: none;
    -ms-user-select: none;
    user-select: none;
    cursor: default;
    -webkit-animation-name: emojis-fall, emojis-shake;
    -webkit-animation-duration: 5s, 3s;
    -webkit-animation-timing-function: linear, ease-in-out;
    -webkit-animation-iteration-count: {animation_length}, {animation_length}; // overall length
    -webkit-animation-play-state: running, running;
    animation-name: emojis-fall, emojis-shake;
    animation-duration: {falling_speed}s, 3s;  // fall speed
    animation-timing-function: linear, ease-in-out;
    animation-iteration-count: {animation_length}, {animation_length}; // overall length
    animation-play-state: running, running;
    }}
    .emoji:nth-of-type(0) {{
    left: 1%;
    -webkit-animation-delay: 0s, 0s;
    animation-delay: 0s, 0s;
    }}
    .emoji:nth-of-type(1) {{
    left: 10%;
    -webkit-animation-delay: 1s, 1s;
    animation-delay: 1s, 1s;
    }}
    .emoji:nth-of-type(2) {{
    left: 20%;
    -webkit-animation-delay: 6s, 0.5s;
    animation-delay: 6s, 0.5s;
    }}
    .emoji:nth-of-type(3) {{
    left: 30%;
    -webkit-animation-delay: 4s, 2s;
    animation-delay: 4s, 2s;
    }}
    .emoji:nth-of-type(4) {{
    left: 40%;
    -webkit-animation-delay: 2s, 2s;
    animation-delay: 2s, 2s;
    }}
    .emoji:nth-of-type(5) {{
    left: 50%;
    -webkit-animation-delay: 8s, 3s;
    animation-delay: 8s, 3s;
    }}
    .emoji:nth-of-type(6) {{
    left: 60%;
    -webkit-animation-delay: 6s, 2s;
    animation-delay: 6s, 2s;
    }}
    .emoji:nth-of-type(7) {{
    left: 70%;
    -webkit-animation-delay: 2.5s, 1s;
    animation-delay: 2.5s, 1s;
    }}
    .emoji:nth-of-type(8) {{
    left: 80%;
    -webkit-animation-delay: 1s, 0s;
    animation-delay: 1s, 0s;
    }}
    .emoji:nth-of-type(9) {{
    left: 90%;
    -webkit-animation-delay: 3s, 1.5s;
    animation-delay: 3s, 1.5s;
    }}

    </style>
    """,
        unsafe_allow_html=True,
    )

    st.write(
        f"""
    <!--get emojis from https://getemoji.com-->
    <div class="emojis">
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
        <div class="emoji">
            {emoji}
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

rain(emoji='🚂',
     font_size=20,
    falling_speed=40,
    animation_length="infinite")
