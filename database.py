import csv
import pandas as pd
import psycopg2

class database:
    con = None
    cur = None

    def __init__(self):
        self.con = psycopg2.connect(host='localhost', database='altjoeah_cryptodb', port=5432,
                                    user='altjoeah', password='f3N03rll:)')
        self.cur = self.con.cursor()

    def close(self):
        self.cur.close()
        self.con.close()

    def select_df(self, command, index=None):
        return pd.read_sql_query(command, self.con, index_col=index)

    def query(self, command):
        self.cur.execute(command)
        self.con.commit()

    def query_without_commit(self, command):
        self.cur.execute(command)
    
    def commit(self):
        self.con.commit()

    def fetch(self):
        return self.cur.fetchall()

    def fetch_one(self):
        return self.cur.fetchone()[0]

    def run_file(self, filename):
        self.cur.execute(open(filename, 'r').read())
        self.con.commit()
