from os import stat
from numpy import NaN, random
from numpy.lib.function_base import cov
from scipy.optimize.moduleTNC import minimize
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
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def main():
    stock = np.asanyarray([17.8, 39.0, 12.8, 24.2, 17.2])
    index = np.asanyarray([13.7, 23.2, 6.9, 16.8, 12.3])

    from sklearn.linear_model import LinearRegression
    import plotly.express as px

    def get_a_and_b_linear_regression(X, y):
        reg = LinearRegression().fit(index.reshape(-1, 1), stock)
        print(reg.score(index.reshape(-1, 1), stock))
        return {'a': reg.intercept_, 'b': reg.coef_}

    print(get_a_and_b_linear_regression(index, stock))

    # fig = px.scatter(x=index, y=stock)
    # fig.show()
    


# with SSHTunnelForwarder((REMOTE_HOST, REMOTE_SSH_PORT), ssh_username=SSH_USERNAME, ssh_password=SSH_PASSWORD,
#                             remote_bind_address=REMOTE_BIND_ADDR, local_bind_address=LOCAL_BIND_ADDR):
main() 