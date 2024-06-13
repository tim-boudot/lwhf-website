import streamlit as st
import riskfolio as rp
import requests, time
import pandas as pd
import matplotlib.pyplot as plt
import os
import json




class Portfolio_Optimisation:
    def __init__(self, Y, method_mu='hist',
                method_cov='hist',
                model='Classic',
                rm = 'MV',
                obj = 'Sharpe',
                hist = True,
                rf = 0,
                l = 0):


        '''
        method_mu (str, optional):
        The method used to estimate the expected returns. The default value is hist. Possible values are:
        * hist: use historical estimates.
        * ewma1: use ewma with adjust=True, see EWM for more details.
        * ewma2: use ewma with adjust=False, see EWM for more details.

        '''

        self.Y = Y
        self.method_mu = method_mu
        self.method_cov = method_cov
        self.model = model
        self.rm = rm
        self.obj = obj
        self.hist = hist
        self.rf = rf
        self.l = l


    def building_portfolio(self):

        '''
        Calculates current optimal portfolio for the future

        INPUT
        * Y: datafame containing returns as columns and dates as index
        * Methods to estimate input parameters:
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'MV' # Risk measure used, this time will be variance
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'

        OUTPUT
        * port: object for the portfolio
        * w: weight of the optimised portfolio
        '''
        # Building the portfolio object
        port = rp.Portfolio(returns=self.Y)

        #find estimates for returns and covariance or # input manually the custom parameters

        port.assets_stats(method_mu=self.method_mu, method_cov=self.method_cov, d=0.94)

        #find weights that maximizes Sharpe ratio (rm = 'MV'; Mean-Variance)
        w = port.optimization(model=self.model, rm=self.rm, obj=self.obj, rf=self.rf, l=self.l, hist=self.hist)

        self.port = port
        self.w = w

        return port, w

    # def pie_weights(self):
    #     ax = rp.plot_pie( w= self.w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
    #                     height=6, width=10, ax=None)

    # def EfficientFrontier(self, n_points = 50):
    #     points = 50 # Number of points of the frontier
    #     frontier = self.port.efficient_frontier(model=self.model, rm=self.rm, points=n_points, rf=self.rf, hist=self.hist)
    #     self.frontier = frontier
    #     return frontier

    # def plot_EF(self):
    #     label = 'Max Risk Adjusted Return Portfolio' # Title of point
    #     self.mu = self.port.mu # Expected returns
    #     self.cov = self.port.cov # Covariance matrix
    #     self.returns = self.port.returns # Returns of the assets
    #     ax = rp.plot_frontier(w_frontier=self.frontier, mu=self.mu, cov=self.cov, returns=self.returns, rm=self.rm,
    #                     rf=self.rf, alpha=0.05, cmap='viridis', w=self.w, label=label,
    #                     marker='*', s=16, c='r', height=6, width=10, ax=None)
    # def plot_EF_assetStructure(self):
    #     ax = rp.plot_frontier_area(w_frontier=self.frontier, cmap="tab20", height=6, width=10, ax=None)



def final_backtest_charts(method_mu: str, method_cov:str, model:str, rm: str, objective: str):
    #choose between:
    # method_my_mu = ['hist', 'ewma1', 'ewma2'],
    # method_my_cov = ['hist', 'ewma1', 'ewma2'],
    # model ={'Classic', 'BL', 'FM' or 'BLFM'}
    # Possible values The risk measure used to optimize the portfolio rm:
            # ’MV’: Standard Deviation.
            # ’KT’: Square Root of Kurtosis.
            # ’MAD’: Mean Absolute Deviation.
            # ’GMD’: Gini Mean Difference.
            # ’MSV’: Semi Standard Deviation.
            # ’SKT’: Square Root of Semi Kurtosis.
            # ’FLPM’: First Lower Partial Moment (Omega Ratio).
            # ’SLPM’: Second Lower Partial Moment (Sortino Ratio).
            # ’CVaR’: Conditional Value at Risk.
            # ’TG’: Tail Gini.
            # ’EVaR’: Entropic Value at Risk.
            # ’RLVaR’: Relativistic Value at Risk.
            # ’WR’: Worst Realization (Minimax).
            # ’RG’: Range of returns.
            # ’CVRG’: CVaR range of returns.
            # ’TGRG’: Tail Gini range of returns.
            # ’MDD’: Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
            # ’ADD’: Average Drawdown of uncompounded cumulative returns.
            # ’CDaR’: Conditional Drawdown at Risk of uncompounded cumulative returns.
            # ’EDaR’: Entropic Drawdown at Risk of uncompounded cumulative returns.
            # ’RLDaR’: Relativistic Drawdown at Risk of uncompounded cumulative returns.
            # ’UCI’: Ulcer Index of uncompounded cumulative returns.
    #objective function of the model  {'MinRisk', 'Utility', 'Sharpe' or 'MaxRet'}

    #same api sinc
    backtest_charts_api_cache_folder = 'backtest_charts_api_cache'
    full_path = os.path.join(QUERIED_CACHE_LOCAL, backtest_charts_api_cache_folder)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    filename = f'{method_mu}-{method_cov}-{model}-{rm}-{objective}.json'
    json_full_path = os.path.join(full_path, filename)
    if os.path.exists(json_full_path):
        print(f'✅ Found {filename} in the local cache.')
        # read the json file as a dictionary
        with open(json_full_path, 'r') as file:
            result = json.load(file)
        return result

    # time.sleep(15)
    assets_df, assets, data, Y = train_data()


    my_por = Portfolio_Optimisation(Y, method_mu,
                        method_cov,
                        model,
                        rm ,
                        objective,
                        hist = True,
                        rf = 0,
                        l = 0)
    port, w = my_por.building_portfolio()
    keep_idx = w['weights'][w['weights'] > 1e-5].index
    w = w.loc[list(keep_idx)]

    result = {
              'w': w,
              'port':port }

    # save the file as a json to full_path
    with open(json_full_path, 'w') as f:
        json.dump(result, f)

    return result










st.markdown('''# Plotting backtesting results''')

#predit_url = ('https://lwhf-edxf3vliba-ew.a.run.app/predict')
backtest_url = ('https://lwhf4-edxf3vliba-ew.a.run.app/backtest')
backtest_charts_url = ('https://lwhf4-edxf3vliba-ew.a.run.app/backtest_charts')


if st.button("Click here!", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False, use_container_width=False):

    #choose between:
    # method_my_mu = ['hist', 'ewma1', 'ewma2'],
    # method_my_cov = ['hist', 'ewma1', 'ewma2'],
    # model = {'Classic', 'BL', 'FM' or 'BLFM'}
    # Possible values The risk measure used to optimize the portfolio rm:
            # ’MV’: Standard Deviation.
            # ’KT’: Square Root of Kurtosis.
            # ’MAD’: Mean Absolute Deviation.
            # ’GMD’: Gini Mean Difference.
            # ’MSV’: Semi Standard Deviation.
            # ’SKT’: Square Root of Semi Kurtosis.
            # ’FLPM’: First Lower Partial Moment (Omega Ratio).
            # ’SLPM’: Second Lower Partial Moment (Sortino Ratio).
            # ’CVaR’: Conditional Value at Risk.
            # ’TG’: Tail Gini.
            # ’EVaR’: Entropic Value at Risk.
            # ’RLVaR’: Relativistic Value at Risk.
            # ’WR’: Worst Realization (Minimax).
            # ’RG’: Range of returns.
            # ’CVRG’: CVaR range of returns.
            # ’TGRG’: Tail Gini range of returns.
            # ’MDD’: Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
            # ’ADD’: Average Drawdown of uncompounded cumulative returns.
            # ’CDaR’: Conditional Drawdown at Risk of uncompounded cumulative returns.
            # ’EDaR’: Entropic Drawdown at Risk of uncompounded cumulative returns.
            # ’RLDaR’: Relativistic Drawdown at Risk of uncompounded cumulative returns.
            # ’UCI’: Ulcer Index of uncompounded cumulative returns.
    #objective function of the model  {'MinRisk', 'Utility', 'Sharpe' or 'MaxRet'}

    params = {
        'method_mu': str,
        'method_cov':str,
        'model':str,
        'rm': str,
        'objective': str
    }


    req = requests.get(backtest_charts_url, params)




    res = req.json()
    weights = res['w']

    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['weights'])

    time.sleep(1)

    fig, ax = plt.subplots()
    rp.plot_pie( w= weights_df, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                            height=6, width=10, ax=ax)
    st.pyplot(fig)
