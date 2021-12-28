from sshtunnel import SSHTunnelForwarder
from db_manager import db_manager
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import Quant

REMOTE_HOST = 'mi3-ss64.a2hosting.com'
REMOTE_SSH_PORT = 7822
SSH_USERNAME = 'altjoeah'
SSH_PASSWORD = 'f3N03rll:)'
REMOTE_BIND_ADDR = ('localhost', 5432)
LOCAL_BIND_ADDR = REMOTE_BIND_ADDR

def main():
    manager = db_manager()

    df = pd.read_csv('SandPhedge.csv')
    spot = df['Spot']
    

    df['RSpot'] = Quant.compounded_returns(spot)

    print(df)

    manager.close()




with SSHTunnelForwarder((REMOTE_HOST, REMOTE_SSH_PORT), ssh_username=SSH_USERNAME, ssh_password=SSH_PASSWORD,
                            remote_bind_address=REMOTE_BIND_ADDR, local_bind_address=LOCAL_BIND_ADDR):
    main()