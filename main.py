from os import stat
from sshtunnel import SSHTunnelForwarder
from db_manager import db_manager
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import Quant as quant
import scipy.stats as st


REMOTE_HOST = 'mi3-ss64.a2hosting.com'
REMOTE_SSH_PORT = 7822
SSH_USERNAME = 'altjoeah'
SSH_PASSWORD = 'f3N03rll:)'
REMOTE_BIND_ADDR = ('localhost', 5432)
LOCAL_BIND_ADDR = REMOTE_BIND_ADDR

def main():
    df = pd.read_csv('RickySteve.csv')

    ricky = list(df['Ricky'].to_numpy(dtype='float64'))
    steve = df['Steve'].to_numpy(dtype='float64')
    
    print(quant.covariance(ricky, steve))
    print(quant.correlation(ricky, steve))

    # stat_df = pd.DataFrame()
    # stat_df = stat_df.append(simple_stats(ricky, 'ricky'))
    # stat_df = stat_df.append(simple_stats(steve, 'steve'))
    # print(stat_df) 


# with SSHTunnelForwarder((REMOTE_HOST, REMOTE_SSH_PORT), ssh_username=SSH_USERNAME, ssh_password=SSH_PASSWORD,
#                             remote_bind_address=REMOTE_BIND_ADDR, local_bind_address=LOCAL_BIND_ADDR):
main()