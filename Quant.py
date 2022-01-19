import numpy as np
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

def portfolio_variance(returns, weights):
    covariance = covariance_matrix(returns)
    return np.dot(np.dot(np.transpose(weights), covariance), weights)
    