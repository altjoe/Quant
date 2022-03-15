from os import stat
from random import sample
from sshtunnel import SSHTunnelForwarder
from db_manager import db_manager
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import Quant as quant
import plotly.express as px
from db_manager import db_manager

REMOTE_HOST = 'mi3-ss64.a2hosting.com'
REMOTE_SSH_PORT = 7822
SSH_USERNAME = 'altjoeah'
SSH_PASSWORD = 'f3N03rll:)'
REMOTE_BIND_ADDR = ('localhost', 5432)
LOCAL_BIND_ADDR = REMOTE_BIND_ADDR
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def main():
    manager = db_manager()

    df = manager.select_df('select * from (select * from ethusd_15_model1 order by d_sec desc limit 5000 offset 0) as top order by d_sec asc')
    df['returns'] = quant.compounded_returns(df['c'].values)
    df['ret_predict'] = df['returns'].shift(-1)
    df['3ret_predict'] = df['returns'].shift(-5).rolling(5, min_periods=1).sum()
    
    print(df.head())
    # print(df.head(5))

    import matplotlib.pyplot as plt
    df = df.dropna()
    # print(quant.covariance(df['rsi3'], df['ret_predict']))

    sample_size = 48 * 4

    highest_corr = 0
    best_df = pd.DataFrame()
    index_arr = []
    corr_arr = []
    for i in range(4000):
        mask = (df.index >= i) & (df.index < i + sample_size)
        masked_df = df[mask]
        columns = ['rsi3', 'rsi4', 'rsi5', 'rsi6', 'rsi7', 'rsi8', 'rsi9']#, 'macd_26_12_9', 'bb_20']

        # avg_corr = 0
        # for col in columns:
        #     avg_corr += quant.correlation(masked_df[col], masked_df['3ret_predict'])
        # avg_corr /= len(columns)

        corr_arr.append(quant.correlation(masked_df['rsi9'], masked_df['3ret_predict']))
        index_arr.append(i)

    # print(highest_corr)

    # columns = ['rsi3', 'rsi4', 'rsi5', 'rsi6', 'rsi7', 'rsi8', 'rsi9']#, 'macd_26_12_9', 'bb_20']
    # for col in columns:
    #     corr = abs(quant.correlation(best_df[col], best_df['3ret_predict']))
    #     print(f'{col}: {corr}')


    plt.plot(index_arr, corr_arr)
    # plt.scatter(best_df['rsi9'], best_df['3ret_predict'])
    plt.show()
    # print(df)


    manager.close()


    # fig = px.scatter(x=index, y=stock)
    # fig.show()
    


with SSHTunnelForwarder((REMOTE_HOST, REMOTE_SSH_PORT), ssh_username=SSH_USERNAME, ssh_password=SSH_PASSWORD,
                            remote_bind_address=REMOTE_BIND_ADDR, local_bind_address=LOCAL_BIND_ADDR):
    main() 