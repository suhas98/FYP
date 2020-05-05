import quantopian.optimize as opti
import quantopian.pipeline.factors as Factors
from quantopian.pipeline import Pipeline, CustomFactor 
 
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.data import Fundamentals
 
from quantopian.pipeline.data.builtin import USEquityPricing  
from quantopian.pipeline.experimental import risk_loading_pipeline
from quantopian.pipeline.factors import SimpleBeta
 
from quantopian.pipeline.factors import SimpleMovingAverage
import numpy as np
import pandas as pd
 
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from quantopian.pipeline.factors import Latest, DailyReturns, Returns
 
import cvxopt as opt
import cvxpy as cvx
import datetime
from cvxopt import blas, solvers
 
import quantopian.algorithm as algo
 
 
def initialize(context):    
     #Schedule a function'rebalance', to run once a month
    schedule_function(schedule,
                      date_rules.month_start(),
                      time_rules.market_open(hours=0, minutes=3))
    attach_pipeline(make_pipeline(), 'my_pipeline')
    attach_pipeline(risk_loading_pipeline(), 'risk_factors')
        
def make_pipeline():
    
    # beta = SimpleBeta(
    #                 target=sid(8554),
    #                 regression_length=252,
    #                 )
        
    universe = QTradableStocksUS()
    
    
    # 1 billion Mkt Cap to include small, mid and large cap companies 
    # universe &= (Fundamentals.market_cap.latest > 1e9) 
    
    #Companies with Positive Earnings which indicate profitability before taxes
    universe &= (Fundamentals.ebit.latest > 0)                        
    
    # Companies with Positive Enterprise Value = Market Cap + Total Debt - Cash (Liquid Assets)
    universe &= (Fundamentals.enterprise_value.latest > 0)            
    
    # Companies with Positive EV/EBIT Ratio 
    universe &= (Fundamentals.ev_to_ebitda.latest > 0)               
    
    #Filter stocks with low volatility 
    low_volatility_screen = Factors.AnnualizedVolatility(mask=universe).percentile_between(85,95)
    universe &= low_volatility_screen                                 
 
    # universe &= beta.notnull()
 
    #Piotroski Scoring Criterias
    
    #Profitability Criteria
    profit = ROA() + ROAChange() + CashFlow() + CashFlowFromOps()  
    
    #Leverage and Liquidity Criteria
    leverage = LongTermDebtRatioChange() + CurrentDebtRatioChange() + SharesOutstandingChange() 
    
    #Operating Efficiency Criteria
    operating = GrossMarginChange() + AssetsTurnoverChange()  
    
    #Piotroski F-Score
    piotroski_f_score = profit + leverage + operating
    
    return Pipeline(columns={'piotroski': piotroski_f_score}, screen=universe)
 
 
#Scheduler Function called at the beginning of every month
def schedule(context, data):
    stock_analysis(context, data)
    allocate(context, data)
    create_model(context, data)
    trade(context, data)
 
def stock_analysis(context, data):
    
    #Get data from pipeline and select the top 50 stocks with highest Piotroski Score
    context.output = pipeline_output('my_pipeline')
    
    context.stocks = context.output.index
       
    context.best_equities = context.output.sort_values('piotroski', ascending=False).head(50).index
    
    #Calculate the historic prices for the shortlisted prices
    context.prices = data.history(assets=context.best_equities, fields='price', bar_count=60, frequency='1d').dropna()
    
def allocate(context, data):
    
    #Create the ML model 
    context.vol_model = GradientBoostingRegressor()
    context.price_model = GradientBoostingRegressor()
 
    #Lookback period and history range for collecting data
    context.lookback = 1
    context.history_range = 150
    
    #Get pricing and returns data for stock universe shortlisted 
    prices = context.prices[context.best_equities]
    returns = prices.pct_change().dropna().as_matrix(context.best_equities)
    
    #Calculate the optimal weights for the stocks using Mean-Variance Optimization technique
    allocation = optimal_portfolio(context, returns.T)
    
def create_model(context, data):
    
    #Train the two ML models for each stock data
    for idx, security in enumerate(context.best_equities):
        create_vol_model(context, data, idx)
        create_price_model(context, data, idx)
 
def create_vol_model(context, data, idx):
    
    # Retrieve the the daily share volume data for the stock
    recent_volumes = data.history(context.best_equities[idx], 'volume', context.history_range, '1d').values
    
    # Calculate the change in volume
    volume_changes = np.diff(recent_volumes).tolist()
 
    #Input and Output variables for the ML model
    X = []
    Y = []
    
    # Each day in the historic range save the previous volume to X and current volume to Y
    for i in range(context.history_range-context.lookback-1):
        X.append(volume_changes[i:i+context.lookback])
        Y.append(volume_changes[i+context.lookback])
 
    #Fit the model with the input and output variable
    context.vol_model.fit(X, Y)
 
def create_price_model(context, data, idx):
    
    # Retrieve the the daily share price data for the stock
    recent_prices = data.history(context.best_equities[idx], 'price', context.history_range, '1d').values
    
    # Calculate the change in prices
    price_changes = np.diff(recent_prices).tolist()
 
    #Input and Output variables for the ML model
    X = [] 
    Y = [] 
    
   # Each day in the historic range save the previous volume to X and current volume to Y
    for i in range(context.history_range-context.lookback-1):
        X.append(price_changes[i:i+context.lookback])
        Y.append(price_changes[i+context.lookback]) 
 
    #Fit the model with the input and output variable
    context.price_model.fit(X, Y)  
 
#Mean Variance Optimization technique to calculate the optimal weights
def optimal_portfolio(context,returns):
    
    n = len(returns)
    returns = np.asmatrix(returns)
 
    # Convert to cvxopt matrices
    # minimize: w * mu*S * w
    S = opt.matrix(np.cov(returns))
    
    # Minimize betas
    pbar = opt.matrix(np.zeros(n))
    
    # Create constraint matrices
    # Gx < h: Every item is positive
    G = opt.matrix(-np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(np.zeros(n))
    
    # Ax = b sum of all items = 1
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(S, pbar, G, h, A, b)['x']
    context.weights_list = pd.Series(np.asarray(wt).ravel(), index=context.best_equities)
    
    return np.asarray(wt).ravel() 
 
#Portfolio Construction and Rebalancing 
def trade(context, data):  
      
    #If both models are created then execute trades
    if context.vol_model and context.price_model:
        for idx, security in enumerate(context.best_equities):
            
            # Get the latest prices to test on the models
            latest_volumes = data.history(security, 'volume', context.lookback+1, '1d').values
            latest_prices = data.history(security, 'price', context.lookback+1, '1d').values
 
            # Calculate the change in price and volume
            volume_difference = np.diff(latest_volumes).tolist()
            price_difference = np.diff(latest_prices).tolist()
 
            # Predict using our models and the recent prices and volumes
            vol_prediction = context.vol_model.predict(volume_difference)
            record(vol_prediction = vol_prediction)
            
            price_prediction = context.price_model.predict(price_difference)
            record(price_prediction = price_prediction)
             
            weight = context.weights_list[idx]
            
            #If both volume and price prediction is greater than 0 then go long on the security
            if vol_prediction > 0 and price_prediction > 0:
                order_target_percent(security, abs(weight)*1)
                # context.weights_list[idx] = abs(context.weights_list[idx])*1
                
            #If both volume and price prediction is lesser than 0 then go short on the security
            elif vol_prediction < 0 and price_prediction < 0:
                order_target_percent(security, abs(weight)*(-1))
                # context.weights_list[idx] = abs(context.weights_list[idx])*(-1)
            
            #Else don't trade the security
            else:    
                order_target_percent(security, 0)
                #context.weights_list[idx] = abs(context.weights_list[idx])*0
            
        
class ROA(CustomFactor):  
    window_length = 1  
    inputs = [Fundamentals.roa]  
    def compute(self, today, assets, out, roa):  
        out[:] = (roa[-1] > 0).astype(int)
        
class ROAChange(CustomFactor):  
    window_length = 22  
    inputs = [Fundamentals.roa]  
    def compute(self, today, assets, out, roa):  
        out[:] = (roa[-1] > roa[0]).astype(int)
        
class CashFlow(CustomFactor):  
    window_length = 1  
    inputs = [Fundamentals.operating_cash_flow]  
    def compute(self, today, assets, out, cash_flow):  
        out[:] = (cash_flow[-1] > 0).astype(int)
        
class CashFlowFromOps(CustomFactor):  
    window_length = 1  
    inputs = [Fundamentals.cash_flow_from_continuing_operating_activities, Fundamentals.roa]  
    def compute(self, today, assets, out, cash_flow_from_ops, roa):  
        out[:] = (cash_flow_from_ops[-1] > roa[-1]).astype(int)
        
class LongTermDebtRatioChange(CustomFactor):  
    window_length = 22  
    inputs = [Fundamentals.long_term_debt_equity_ratio]  
    def compute(self, today, assets, out, long_term_debt_ratio):  
        out[:] = (long_term_debt_ratio[-1] < long_term_debt_ratio[0]).astype(int)
        
class CurrentDebtRatioChange(CustomFactor):  
    window_length = 22  
    inputs = [Fundamentals.current_ratio]  
    def compute(self, today, assets, out, current_ratio):  
        out[:] = (current_ratio[-1] > current_ratio[0]).astype(int)
        
class SharesOutstandingChange(CustomFactor):  
    window_length = 22  
    inputs = [Fundamentals.shares_outstanding]  
    def compute(self, today, assets, out, shares_outstanding):  
        out[:] = (shares_outstanding[-1] <= shares_outstanding[0]).astype(int)
        
class GrossMarginChange(CustomFactor):  
    window_length = 22  
    inputs = [Fundamentals.gross_margin]  
    def compute(self, today, assets, out, gross_margin):  
        out[:] = (gross_margin[-1] > gross_margin[0]).astype(int)
        
class AssetsTurnoverChange(CustomFactor):  
    window_length = 22  
    inputs = [Fundamentals.assets_turnover]  
    def compute(self, today, assets, out, assets_turnover):  
        out[:] = (assets_turnover[-1] > assets_turnover[0]).astype(int)