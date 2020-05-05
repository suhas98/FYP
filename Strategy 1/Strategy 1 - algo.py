import pandas as pd
import numpy as np
import cvxpy as cvx
import math
from quantopian.pipeline.data import morningstar
from quantopian.pipeline.data import Fundamentals 
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from sklearn.cluster import KMeans
import quantopian.algorithm as algo
import quantopian.optimize as opti
from quantopian.pipeline.data.builtin import USEquityPricing  
import cvxopt as opt
import datetime
from cvxopt import blas, solvers
 
 
def initialize(context):
    
    
    # Attach pipeline
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'fundamental_line')
    
    schedule_function(schedule,
                      date_rules.month_start(),
                      time_rules.market_open(hours=0, minutes=3))
 
    
def before_trading_start(context, data):
    
    # Pipeline_output returns the constructed dataframe.
    context.output = pipeline_output('fundamental_line')
       
def make_pipeline():
 
    # Get fundamental data from liquid universe for easier trades
    assets = morningstar.balance_sheet.total_assets.latest
    factor_revenues = morningstar.income_statement.total_revenue.latest
    factor_income = morningstar.cash_flow_statement.net_income.latest
    factor_margin = Fundamentals.net_margin.latest
    factor_growth = Fundamentals.growth_score.latest
                                                       
    pipe = Pipeline(
              columns={
                'total_assets': assets,
                'total_revenues': factor_revenues,
                'net_income': factor_income,
                'net_margin': factor_margin,
                'growth_score': factor_growth,
              }, screen = Q1500US()
          )
    
    return pipe
 
def schedule(context, data):
    cluster_analysis(context, data)
    allocate(context, data)
    trade(context, data)
 
def allocate(context, data):
    # prices = context.prices[context.best_equities]
    # returns = prices.pct_change().dropna().as_matrix(context.best_equities)
    returns = data.history(context.best_equities, 'price', 60, '1d').pct_change().dropna()
    try:
        #allocation = optimal_portfolio(context, returns.T)
        allocation = get_minimax_weights(context, returns, max_position=0.05)
    except ValueError as e:
        print(e)
    
def get_minimax_weights(context, returns, max_position=0.05):
    num_stocks = len(returns.columns)
    mean=returns.mean(axis=0).values
    A = returns.as_matrix()
    x = cvx.Variable(num_stocks)
    objective = cvx.Maximize(mean.T*x)
 
    constraints = [  
        A*x >= 0,
        sum(x) <= 1,
        x <= max_position,
        x >= 0  
    ]  
    prob = cvx.Problem(objective, constraints)  
    prob.solve(verbose=True)  
    print (prob.value) 
    w=pd.Series(data=np.asarray(x.value).flatten(),index= context.best_equities)
    w=w/w.abs().sum()
    context.weights_list = w
    return w    
 
# def optimal_portfolio(context,returns):
    
#     n = len(returns)
#     returns = np.asmatrix(returns)
 
#     # Convert to cvxopt matrices
#     # minimize: w * mu*S * w
#     S = opt.matrix(np.cov(returns))
    
#     # Minimize betas
#     pbar = opt.matrix(np.zeros(n))
    
#     # Create constraint matrices
#     # Gx < h: Every item is positive
#     G = opt.matrix(-np.eye(n))   # negative n x n identity matrix
#     h = opt.matrix(np.zeros(n))
    
#     # Ax = b sum of all items = 1
#     A = opt.matrix(1.0, (1, n))
#     b = opt.matrix(1.0)
    
#     # CALCULATE THE OPTIMAL PORTFOLIO
#     wt = solvers.qp(S, pbar, G, h, A, b)['x']
#     context.weights_list = pd.Series(np.asarray(wt).ravel(), index=context.best_equities)
#     return np.asarray(wt).ravel()
 
def cluster_analysis(context, data):
    
    # Get list of equities that made it through the pipeline
    context.output['return_on_asset'] = context.output.net_income / context.output.total_assets
    context.output['asset_turnover'] = context.output.total_revenues / context.output.total_assets
    context.output['asset_margin'] = context.output.net_margin 
    context.output['asset_growth'] = context.output.growth_score
    
    context.output['feature'] = 0.25 * context.output.return_on_asset + 0.25 * context.output.asset_turnover + 0.25 * context.output.asset_margin + 0.25 * context.output.asset_growth
    
    #context.output['feature'] = context.output.alpha    
    
    #Replacing the positive infinity with NaN
    context.output = context.output.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    
    #Get list of all the equities
    equity_list = context.output.index
    
    # Run k-means on feature matrix
    n_clust = 15
    alg = KMeans(n_clusters=n_clust, random_state=10, n_jobs=1)
    cluster_results = alg.fit_predict(context.output[['feature']])
    context.output['cluster'] = pd.Series(cluster_results, index=equity_list)
    
    # Remove single stock cluster
    cluster_count = context.output.cluster.groupby(context.output.cluster).count()
    context.output = context.output[context.output.cluster.isin(cluster_count[cluster_count > 1].index)]
    
    print(('cluster result', cluster_count))
    
    # Get rolling window of past prices and compute returns
    context.prices = data.history(assets=equity_list, fields='price', bar_count=100, frequency='1d').dropna()
    
    log_return = context.prices.apply(np.log).diff()
    
    # Calculate expected returns and Sharpes
    mean_return = log_return.mean().to_frame(name='mean_return')
    return_std = log_return.std().to_frame(name='return_std')
    context.output = context.output.join(mean_return).join(return_std)
    context.output['sharpes'] = context.output.mean_return / context.output.return_std
    
    top_n_sharpes = 1
    best_n_sharpes = context.output.groupby('cluster')['sharpes'].nlargest(top_n_sharpes)
    
    context.best_equities = best_n_sharpes.index.get_level_values(1)
    
    
def trade(context, data):
 
    weights = context.weights_list
    objectives = opti.TargetWeights(weights)
    
    constraints = [
        opti.MaxGrossExposure(1.0),
    ]
    algo.order_optimal_portfolio(objectives, constraints)
 