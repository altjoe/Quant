from os import stat
from sshtunnel import SSHTunnelForwarder
from db_manager import db_manager
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import Quant as quant
import scipy.stats as st
import plotly.express as px

REMOTE_HOST = 'mi3-ss64.a2hosting.com'
REMOTE_SSH_PORT = 7822
SSH_USERNAME = 'altjoeah'
SSH_PASSWORD = 'f3N03rll:)'
REMOTE_BIND_ADDR = ('localhost', 5432)
LOCAL_BIND_ADDR = REMOTE_BIND_ADDR
pd.set_option('display.max_columns', None)
def main():
    df = pd.read_csv('2-8-1-Excersize.csv')
    df['FORDRET'] = quant.simple_returns(df['FORD'].values)
    df['GERET'] = quant.simple_returns(df['GE'].values)
    df['MSOFTRET'] = quant.simple_returns(df['MICROSOFT'].values)
    
    return_columns = ['FORDRET', 'GERET', 'MSOFTRET']
    weights = np.asanyarray([0.33, 0.33, 0.34])
    average_returns = np.asanyarray(df[return_columns].mean())
    df['PORTRET'] = quant.portfolio_returns(df[return_columns], weights)

    
    covariance = quant.covariance_matrix(df[return_columns])
    print(covariance)
    print(f'Average Returns: {average_returns}')

    def portfolio_mean(weights, returns):
        return np.sum(np.asanyarray(returns.mean()) * weights)

    print(portfolio_mean(weights, df[return_columns]))
    variance = np.dot(np.dot(np.transpose(weights), covariance), weights)
    print(variance)
    #pg 137
    # print(df)

    

# with SSHTunnelForwarder((REMOTE_HOST, REMOTE_SSH_PORT), ssh_username=SSH_USERNAME, ssh_password=SSH_PASSWORD,
#                             remote_bind_address=REMOTE_BIND_ADDR, local_bind_address=LOCAL_BIND_ADDR):
main() 