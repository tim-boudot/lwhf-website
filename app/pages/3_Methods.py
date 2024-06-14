import streamlit as st
import riskfolio as rp
import requests, time
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import yfinance as yf
import plotly.express as px


api_cache_folder = 'api_cache'

def train_data(start_date = '2021-01-01', end_date = '2024-06-13'):

    backtest_charts_api_cache_folder = 'backtest_charts_api_cache'
    full_path = os.path.join(api_cache_folder, backtest_charts_api_cache_folder)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    filename = 'assets-df-wiki.csv'
    wiki_full_path = os.path.join(full_path, filename)
    if os.path.exists(wiki_full_path):
        print(f'✅ Found {filename} in the local cache.')
        # read the json file as a dictionary
        assets_df = pd.read_csv(wiki_full_path, sep=',')
    else:
        assets_df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        assets_df.to_csv(wiki_full_path, index = False)

    assets = assets_df['Symbol'].to_list()

    filename = 'data-yf.csv'
    yf_full_path = os.path.join(full_path, filename)
    if os.path.exists(yf_full_path):
        print(f'✅ Found {filename} in the local cache.')
        data = pd.read_csv(yf_full_path, sep=',')
    else:
        print('downloading data')
        data = yf.download(assets, start = start_date, end = end_date)
        data = data.loc[:, ('Adj Close', slice(None))]
        data.columns = assets
        # Ensure all datetime indices are timezone-naive
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        data.to_csv(yf_full_path, index = False)



    #RETURNS ON ONE WEEK (5)
    Y = data[assets].pct_change(5)[5:]
    Y = Y.dropna(axis=1)
    data = data.iloc[:-5]
    return assets_df, assets, data[:-5], Y




def building_portfolio(Y_b, method_mu_b='hist',
                method_cov_b='hist',
                rm_b = 'MV',
                obj_b = 'Sharpe'):

    # Building the portfolio object
    port_b = rp.Portfolio(returns=Y_b)
    #find estimates for returns and covariance or # input manually the custom parameters
    port_b.assets_stats(method_mu= method_mu_b, method_cov= method_cov_b)
    #find weights that maximizes Sharpe ratio (rm = 'MV'; Mean-Variance)
    w_b = port_b.optimization(model= 'Classic', rm= rm_b, obj= obj_b)
    return port_b, w_b

st.markdown('''# Build your portfolio using different methods!''')

selected_mu = st.selectbox(
    "Which method would you like to use to find you expected returns?",
    ['hist', 'ewma1', 'ewma2'],
    index=None,
    placeholder="Method to find mu ...",
)

formulas_mu = {
    "hist": r'''
    #####  The historical mean is calculated as:
    $$
    \mu = \frac{1}{N} \sum_{i=1}^{t} x_{i}
    $$
    ''',
    "ewma1": r'''
    #####  In general, an exponential weighted moving average of type 1 is calculated as:
    $$
    EWMA_{1} = \frac{\sum_{i=0}^{t}w_{i}x_{t-i} }{\sum_{i=0}^{t}w_{i}}\\
    $$
    ###### where the weights are given by:
    $$
    w_{i} = (1- \alpha )^{i}
    $$
    ''',
    "ewma2": r'''

    #####  In general, an exponential weighted moving average of type 2 is calculated as:
    $$
    EWMA_{1} = \frac{\sum_{i=0}^{t}w_{i}x_{t-i} }{\sum_{i=0}^{t}w_{i}}\\
    $$
    ###### where the weights are given by:
    $$
    w_{i} = \alpha (1- \alpha )^{i} \hspace{0.2cm} if \hspace{0.2cm}i<t\
        w_{i} = (1- \alpha )^{i}\hspace{0.2cm} if \hspace{0.2cm}i=t
    $$
    '''
}


# Function to display the selected formula
def display_formula(name, formulas):
    formula = formulas[name]
    st.write(formula)

# Display the selected formula outside the dropdown
if selected_mu:
    display_formula(selected_mu, formulas_mu)

selected_cov = st.selectbox(
    "Which method would you like to use to find you expected returns?",
    ['hist', 'ewma1', 'ewma2'],
    index=None,
    placeholder="Method to find cov...",
)


formulas_cov = {
    "hist": r'''
    #####  The historical cov is calculated as:
    $$
    \text{Cov}_{\mu} = \frac{1}{N} \sum_{i=1}^{t} \text{Cov}_{i}
    $$
    ''',
    "ewma1": r'''
    #####  In general, a weighted moving average is calculated as:
    $$
    EWMA_{1} = \frac{\sum_{i=0}^{t}w_{i}\text{Cov}_{t-i} }{\sum_{i=0}^{t}\text{Cov}_{i}}\\
    $$
    ###### where the weights are given by:
    $$
    w_{i} = (1- \alpha )^{i}
    $$
    ''',
    "ewma2": r'''

    #####  In general, a weighted moving average is calculated as:
    $$
    EWMA_{1} = \frac{\sum_{i=0}^{t}w_{i}\text{Cov}_{t-i} }{\sum_{i=0}^{t}w_{i}}\\
    $$
    ###### where the weights are given by:
    $$
    w_{i} = \alpha (1- \alpha )^{i} \hspace{0.2cm} if \hspace{0.2cm}i<t\
        w_{i} = (1- \alpha )^{i}\hspace{0.2cm} if \hspace{0.2cm}i=t
    $$
    '''
}


# Display the selected formula outside the dropdown
if selected_cov:
    display_formula(selected_cov, formulas_cov)


selected_rm = st.selectbox(
    "How would you like to measure you risk?",
    ['MV', 'MAD', 'GMD', 'MSV', 'FLPM', 'SLPM', 'CVaR', 'TG',
    'EVaR', 'RLVaR', 'WR', 'RG', 'CVRG', 'TGRG', 'MDD', 'ADD', 'CDaR',
    'EDaR', 'RLDaR', 'UCI'
],
    index=None,
    placeholder="Select a risk measure...",
)


formulas_rm = {
    "MV": r'''
    ##### Standard Deviation
    $$
    \sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2}
    $$
    ''',
    "KT": r'''
    ##### Square Root of Kurtosis
    $$
    \sqrt{\frac{\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^4}{\left(\frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2\right)^2}}
    $$
    ''',
    "MAD": r'''
    ##### Mean Absolute Deviation
    $$
    \text{MAD} = \frac{1}{N} \sum_{i=1}^{N} |x_i - \mu|
    $$
    ''',
    "GMD": r'''
    ##### Gini Mean Difference
    $$
    \text{GMD} = \frac{1}{N(N-1)} \sum_{i=1}^{N} \sum_{j=1}^{N} |x_i - x_j|
    $$
    ''',
    "MSV": r'''
    ##### Semi Standard Deviation
    $$
    \text{MSV} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \min(0, x_i - \mu)^2}
    $$
    ''',
    "SKT": r'''
    ##### Square Root of Semi Kurtosis
    $$
    \sqrt{\frac{\frac{1}{N} \sum_{i=1}^{N} \min(0, x_i - \mu)^4}{\left(\frac{1}{N} \sum_{i=1}^{N} \min(0, x_i - \mu)^2\right)^2}}
    $$
    ''',
    "FLPM": r'''
    ##### First Lower Partial Moment (Omega Ratio)
    $$
    \text{FLPM} = \frac{1}{N} \sum_{i=1}^{N} \max(0, \tau - x_i)
    $$
    ''',
    "SLPM": r'''
    ##### Second Lower Partial Moment (Sortino Ratio)
    $$
    \text{SLPM} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \max(0, \tau - x_i)^2}
    $$
    ''',
    "CVaR": r'''
    ##### Conditional Value at Risk
    $$
    \text{CVaR}_\alpha = \frac{1}{\alpha} \int_{0}^{\alpha} \text{VaR}_u \, du
    $$
    ''',
    "TG": r'''
    ##### Tail Gini
    $$
    \text{TG} = \frac{2}{\alpha(1-\alpha)} \int_{0}^{\alpha} \left[ \text{VaR}_u - \text{VaR}_{\alpha} \right] \, du
    $$
    ''',
    "EVaR": r'''
    ##### Entropic Value at Risk
    $$
    \text{EVaR}_\alpha = \inf_{\theta > 0} \left\{ \frac{1}{\theta} \left( \log \mathbb{E}[e^{\theta L}] - \log(1/\alpha) \right) \right\}
    $$
    ''',
    "RLVaR": r'''
    ##### Relativistic Value at Risk
    $$
    \text{RLVaR}_\alpha = \frac{1}{\theta} \log \left( \mathbb{E}[e^{\theta L}]^{\alpha} \right)
    $$
    ''',
    "WR": r'''
    ##### Worst Realization (Minimax)
    $$
    \text{WR} = \min(x_1, x_2, \ldots, x_N)
    $$
    ''',
    "RG": r'''
    ##### Range of returns
    $$
    \text{RG} = \max(x_1, x_2, \ldots, x_N) - \min(x_1, x_2, \ldots, x_N)
    $$
    ''',
    "CVRG": r'''
    ##### CVaR range of returns
    $$
    \text{CVRG} = \text{CVaR}_\alpha^{\max} - \text{CVaR}_\alpha^{\min}
    $$
    ''',
    "TGRG": r'''
    ##### Tail Gini range of returns
    $$
    \text{TGRG} = \text{TG}_\alpha^{\max} - \text{TG}_\alpha^{\min}
    $$
    ''',
    "MDD": r'''
    ##### Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio)
    $$
    \text{MDD} = \max_{\tau \in [0,T]} \left( \frac{\max_{t \in [0,\tau]} P(t) - P(\tau)}{\max_{t \in [0,\tau]} P(t)} \right)
    $$
    ''',
    "ADD": r'''
    ##### Average Drawdown of uncompounded cumulative returns
    $$
    \text{ADD} = \frac{1}{T} \sum_{\tau=1}^{T} \left( \frac{\max_{t \in [0,\tau]} P(t) - P(\tau)}{\max_{t \in [0,\tau]} P(t)} \right)
    $$
    ''',
    "CDaR": r'''
    ##### Conditional Drawdown at Risk of uncompounded cumulative returns
    $$
    \text{CDaR}_\alpha = \frac{1}{\alpha} \int_{0}^{\alpha} \text{DDaR}_u \, du
    $$
    ''',
    "EDaR": r'''
    ##### Entropic Drawdown at Risk of uncompounded cumulative returns
    $$
    \text{EDaR}_\alpha = \inf_{\theta > 0} \left\{ \frac{1}{\theta} \left( \log \mathbb{E}[e^{\theta \text{DD}}] - \log(1/\alpha) \right) \right\}
    $$
    ''',
    "RLDaR": r'''
    ##### Relativistic Drawdown at Risk of uncompounded cumulative returns
    $$
    \text{RLDaR}_\alpha = \frac{1}{\theta} \log \left( \mathbb{E}[e^{\theta \text{DD}}]^{\alpha} \right)
    $$
    ''',
    "UCI": r'''
    ##### Ulcer Index of uncompounded cumulative returns
    $$
    \text{UCI} = \sqrt{\frac{1}{T} \sum_{\tau=1}^{T} \left( \frac{\max_{t \in [0,\tau]} P(t) - P(\tau)}{\max_{t \in [0,\tau]} P(t)} \right)^2}
    $$
    '''
}


# Display the selected formula outside the dropdown
if selected_rm:
    display_formula(selected_rm, formulas_rm)



selected_objective = st.selectbox(
    "Which objective function you want to minimise / maximaze?",
    ['MinRisk', 'Sharpe', 'MaxRet'],
    index=None,
    placeholder="Select an objective function...",
)


formulas_obj = {
    "MinRisk": r'''
    #####  Minimizes the selected risk measure phi:
    $$
    w \hspace{0.3cm} s.t \hspace{0.3cm} min({\phi_{k}\hspace{0.09cm}({w}))}
    $$
    '''
    ,
    "MaxRet": r'''
    #####  Maximizes the return of our portfolio:
    $$
    w \hspace{0.3cm} s.t \hspace{0.3cm} max\hspace{0.03cm}({R\hspace{0.03cm}({w}))}
    $$
    ''',

    "Sharpe": r'''

    #####  Maximizes the risk adjusted return ratio based on the selected risk measure:
    $$
    w \hspace{0.3cm} s.t \hspace{0.3cm} max\hspace{0.03cm}(\frac{{R\hspace{0.03cm}({w})}-1}{\phi_{k}(w)} )
    $$
    '''
}


# Display the selected formula outside the dropdown
if selected_objective:
    display_formula(selected_objective, formulas_obj)

if st.button('Show Result'):

    backtest_charts_api_cache_folder = 'backtest_charts_api_cache'
    full_path = os.path.join(api_cache_folder, backtest_charts_api_cache_folder)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    filename = f'{selected_mu}-{selected_cov}-{selected_rm}-{selected_objective}.csv'
    csv_full_path = os.path.join(full_path, filename)

    if os.path.exists(csv_full_path):
        print(f'✅ Found {filename} in the local cache.')
        # read the json file as a dictionary
        weights = pd.read_csv(csv_full_path, sep = ',')
    else:
        my_assets_df, my_assets, my_data, my_Y = train_data()

        result_port, result_w = building_portfolio(Y_b= my_Y,
                                    method_mu_b= selected_mu,
                                    method_cov_b= selected_cov,
                                    rm_b= selected_rm,
                                    obj_b= selected_objective,
                            )


        keep_idx = result_w['weights'][result_w['weights'] > 1e-5].index

        weights = result_w.loc[list(keep_idx)]

        #weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['weights'])

        time.sleep(1)
        print(weights)
        print(type(weights))
        #weights_col = weights.index
        #weights_val = weights[['weights']]
        #weights.sort_values(by=['weights'], ascending = False)

        print()
        weights.reset_index(inplace=True)
        weights.rename(columns={'index': 'Asset'}, inplace=True)
        weights.to_csv(csv_full_path, index = False)


    progress_bar = st.progress(0)

    # Simulate a loading process
    for i in range(100):
        time.sleep(0.05)  # Simulate a task taking time
        progress_bar.progress(i + 1)

    # Display a message once the loading is complete
    st.success("Loading complete!")



    fig = px.pie(weights, names='Asset', values='weights', title='Portfolio Allocation',
                color_discrete_sequence=px.colors.qualitative.T10)

    fig.update_traces(textposition='inside', textfont_size=20)
    fig.update_traces(textposition='inside', textinfo='percent+label')

    fig.update_layout(
    width=800,  # width in pixels
    height=800  # height in pixels
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    st.markdown('<h2 style="text-align: center;">Want to know more about how to get rich?</h2>', unsafe_allow_html=True)

    # Placeholder variables for card details
    card_number = st.text_input("Card Number", max_chars=16, type="password")
    expiry_date = st.date_input("Expiry Date")
    cvv = st.text_input("CVV", max_chars=3, type="password")

    # Submit button to trigger showing the results
    if st.button("Submit Card Details"):
        if card_number and expiry_date and cvv:
            st.markdown('Thank you.')








    # for i in range(1, weights.shape[0]):
    #     print(i)
    #     if weights.iloc[:i,0].sum() > float(90) or i == 15:
    #         print(i)
    #         print(weights.iloc[:i,0].sum())
    #         weights_8 = weights.iloc[:i]
    #         weights_8.loc['Other'] = weights.iloc[i:,0].sum()
    #         break

    # fig, ax = plt.subplots()
    # rp.plot_pie( w= weights_8, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
    #                         height=6, width=10, ax=ax)
    # st.pyplot(fig)


#cache
