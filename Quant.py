import numpy as np
from scipy import optimize
import scipy.stats as st
import pandas as pd

def compounded_returns(values):
    if type(values).__module__ != np.__name__:
        raise Exception(f'Needs a numpy array. Given: {type(values)}')
    prev_values = values[:-1]
    curr_values = values[1:]
    returns = [np.log(curr / prev) * 100 for curr, prev in zip(curr_values, prev_values)]
    returns.insert(0, np.nan)
    return returns

def simple_returns(values):
    if type(values).__module__ != np.__name__:
            raise Exception(f'Needs a numpy array. Given: {type(values)}')
    prev_values = values[:-1]
    curr_values = values[1:]
    returns = [((curr - prev) / prev) * 100 for curr, prev in zip(curr_values, prev_values)]
    returns.insert(0, np.nan)
    return returns

def skewness(values):
        return st.skew(values, bias=False)

def variance(values):
    return np.std(values, ddof=1) ** 2

def standard_deviation(values):
    return np.std(values, ddof=1)

def kurtosis(values):
    return st.kurtosis(values, bias=False)

def simple_stats(values, index):
        return pd.DataFrame(data={'mean': np.mean(values), 
                                'std': standard_deviation(values), 
                                'skew': skewness(values),
                                'kurt': kurtosis(values)}, index=[index]) 
    
def covariance(X, Y):
    xmean = np.mean(X)
    ymean = np.mean(Y)
    summation = np.sum([(x - xmean)*(y - ymean) for x, y in zip(X, Y)])
    return summation / (float(len(X)) - 1.0)

def correlation(X, Y):
    xstd = standard_deviation(X)
    ystd = standard_deviation(Y)
    return covariance(X, Y) / (xstd * ystd)

def spearman_rank_correlation(X, Y):
    return st.spearmanr(X, Y).correlation
    
def spearman_rank_correlation(X, Y):
    xrank = len(X) - st.rankdata(X) 
    yrank = len(Y) - st.rankdata(Y)
    return correlation(xrank, yrank)

def portfolio_returns(returns, weights):
        return returns.mul(weights).sum(1, skipna=False)

def covariance_matrix(returns):
        returns = returns.dropna()
        mean = list(returns.mean())
        deviation = returns.sub(mean)
        tensor = deviation.to_numpy()
        return np.dot(np.transpose(tensor), tensor) / (len(returns) - 1)

def portfolio_variance_covcalc(returns, weights):
    covariance = covariance_matrix(returns)
    return np.dot(np.dot(np.transpose(weights), covariance), weights)

def portfolio_variance(weights, covariance):
    return np.dot(np.dot(np.transpose(weights), covariance), weights)

def portfolio_mean_return(returns, weights):
    means = returns.mean().to_numpy()
    return np.dot(np.asanyarray(weights).T, means)

def portfolio_individual_mean_returns(returns):
        return returns.mean()

def portfolio_standard_dev(returns, weights):
    return portfolio_variance_covcalc(returns, weights) ** 0.5

from numpy.linalg import inv
def minimum_variance_portfolio_weight(returns):
    covariance = covariance_matrix(returns)
    column_vector = np.ones(len(covariance[0])).T
    return np.dot(column_vector, inv(covariance)) / np.dot(np.dot(column_vector.T, inv(covariance)), column_vector)

from scipy.optimize import minimize

def maximize_operation(weights, returns):
            ret_mean = portfolio_individual_mean_returns(returns)
            return -sum(ret_mean * weights)

def minimize_operation(weights, returns):
            ret_mean = portfolio_individual_mean_returns(returns)
            return sum(ret_mean * weights)

def optimize_mean_weight(returns, std, operation):
        def portfolio_variance_fixed_std(w, r, fixed_std):
            return (portfolio_variance_covcalc(r, w) ** 0.5) - fixed_std

        def equal_to_one(weights):
            return sum(weights) - 1

        equality_constraint = {'type': 'eq', 'fun': portfolio_variance_fixed_std, 'args': (returns, std,)}
        add_to_one_contraint = {'type': 'eq', 'fun': equal_to_one}
        
        initial_weight  = np.random.random(len(returns.columns))
        initial_weight /= sum(initial_weight)

        bou = ((0, None), (0, None), (0, None))
        con = (equality_constraint, add_to_one_contraint)

        result = minimize(operation, initial_weight, args=(returns,), constraints=con, tol=1e-6, bounds=bou)
        if result['success']:
            return result['x']
        else:
            return None

def minimize_mean_weights_fixed_std(returns):
    minimized_weights = minimum_variance_portfolio_weight(returns)
    minimum_std = portfolio_standard_dev(returns, minimized_weights)
    starting_std = int((minimum_std // 0.5) + 1)
    ending_std = int((minimum_std // 0.5) * 1.5) - 1
    weight_df = pd.DataFrame()
    for std in range(starting_std, ending_std):
        weight = optimize_mean_weight(returns, std/2, minimize_operation)
        if weight is not None:
            row = {}
            row['Mean'] = portfolio_mean_return(returns, weight)
            for col, weight in zip(returns.columns, weight):
                row[col] = weight
            weight_df = weight_df.append(pd.DataFrame(data=row, index=[std/2]))
    return weight_df


def maximize_mean_weights_fixed_std(returns):
    minimized_weights = minimum_variance_portfolio_weight(returns)
    minimum_std = portfolio_standard_dev(returns, minimized_weights)
    starting_std = int((minimum_std // 0.5) + 1)
    ending_std = int((minimum_std // 0.5) * 3)
    weight_df = pd.DataFrame()
    for std in range(starting_std, ending_std):
        weight = optimize_mean_weight(returns, std/2, maximize_operation)
        if weight is not None:
            row = {}
            row['Mean'] = portfolio_mean_return(returns, weight)
            for col, weight in zip(returns.columns, weight):
                row[col] = weight
            weight_df = weight_df.append(pd.DataFrame(data=row, index=[std/2]))
    return weight_df

def sharpe_operation(weights, returns, risk_free_returns, maxcalc=False):
            risk_free = risk_free_returns.mean()/12 # fixed
            risk_premium = portfolio_mean_return(returns, weights) - risk_free
            if maxcalc:
                return -risk_premium / portfolio_standard_dev(returns, weights)
            else:
                return risk_premium / portfolio_standard_dev(returns, weights)

def maximize_sharpe_ratio_weights(returns, risk_free_returns):

    def equal_to_one(weights):
            return sum(weights) - 1
    
    add_to_one_contraint = {'type': 'eq', 'fun': equal_to_one}
    initial_weight  = np.random.random(len(returns.columns))
    initial_weight /= sum(initial_weight)
    bou = ((0, None), (0, None), (0, None))

    result = minimize(sharpe_operation, initial_weight, args=(returns, risk_free_returns, True), constraints=add_to_one_contraint, tol=1e-6, bounds=bou)
    return result['x']

def capital_market_line(returns, risk_free_returns):
    minimized_weights = minimum_variance_portfolio_weight(returns)
    minimum_std = portfolio_standard_dev(returns, minimized_weights)
    maximum_std = minimum_std * 3
    risk_free = risk_free_returns.mean()/12
    sharpe_weights = maximize_sharpe_ratio_weights(returns, risk_free_returns)
    sharpe_ratio = sharpe_operation(sharpe_weights, returns, risk_free_returns)\

    df = pd.DataFrame()
    for i in range(20):
        std = i
        df = df.append(pd.DataFrame(data={'return': risk_free + sharpe_ratio * std}, index=[std]))
    return df
    
import matplotlib.pyplot as plt

def display_cml_efficient_frontier(returns, risk_free_return):
    minimized_mean_weights = minimize_mean_weights_fixed_std(returns)
    maximized_mean_weights = maximize_mean_weights_fixed_std(returns)
    plt.plot(minimized_mean_weights.index, minimized_mean_weights['Mean'])
    plt.plot(maximized_mean_weights.index, maximized_mean_weights['Mean'])
    cml = capital_market_line(returns, risk_free_return)
    plt.plot(cml.index, cml['return'])
    plt.show()

def portfolio_summary_statistics(returns):
        mvp_weights = minimum_variance_portfolio_weight(returns)
        individual_returns = portfolio_individual_mean_returns(returns)
        covariance = covariance_matrix(returns)
        portfolio_mean_return = portfolio_mean_return(returns, mvp_weights)
        portfolio_variance = portfolio_variance_covcalc(returns, mvp_weights)
        
        portfolio_standard_deviation = portfolio_standard_dev(returns, mvp_weights)

        print(f'Mean: {portfolio_mean_return}\nVariance: {portfolio_variance}\nSTD: {portfolio_standard_deviation}\n')
        print(individual_returns, '\n\nCovariance Matrix:')
        print(covariance)
        print(f'\nWeights: {mvp_weights}')