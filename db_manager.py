from database import database
import ta
import pandas as pd 
import psycopg2
import time

class db_manager(database):
    # basically its taking the base day, taking the hour and extracting it and dividing by for getting how many of those intervals 
    # have already happened, rounding it down then multiplying it by the interval to get the candle in which that particular time lays

    ############# creating trained data
    # manager.create_ohlc(15, 'minute', 'ethusd')
    # manager.create_new_model('ethusd_15', rsi_windows=[3,4,5,6,7,8,9], macd_windows=[(26, 12, 9)], bollinger_windows=[20])
    # manager.create_slice_pointers('ethusd_15_model1', 0.75, 1440//15) # a day length
    def create_new_model(self, model_num, table_name, rsi=True, rsi_windows=[3, 4, 5, 6, 7, 8], bollinger=True, bollinger_windows=[15, 20, 25], macd=True, macd_windows=[(15, 6, 9), (26, 12 ,9), (39, 19, 9)], convolve_window=50):
        table = super().select_df(f'select d, d_sec, c from {table_name} order by d asc', 'd_sec')
        close = table['c']
        model = pd.DataFrame()
        model['c'] = table['c']
        if rsi:
            for rsi_window in rsi_windows:
                model[f'rsi{rsi_window}'] = self.add_single_rsi(close, rsi_window)
        if macd:
            for macd_window in macd_windows:
                model[f'macd_{macd_window[0]}_{macd_window[1]}_{macd_window[2]}'] = self.add_single_macd_diff(close, macd_window)
        if bollinger:
            for bb_win in bollinger_windows:
                model[f'bb_{bb_win}'] = self.add_single_bb(close, bb_win)
        
        # convolved = pp.smooth_data(close, k=convolve_window) 
        # buy_sell = pp.optimal_buy_sell(convolved, close.values)
        # model[f'trend_conv{convolve_window}'] = pp.create_trend_target_data(buy_sell.buy_locals, buy_sell.sell_locals, model.index)
        # model[f'trigger_conv{convolve_window}'] = pp.create_buy_sell_triggers(buy_sell.buy_locals, buy_sell.sell_locals, model.index)

        model_table_name = f'{table_name}_model{model_num}'
        self.add_model_table_from_df(model, model_table_name)
        super().query(f'alter table {model_table_name} add column id serial primary key')

    def create_ohlc(self, interval_len, timeframe, ticker):
        if timeframe == 'minute':
            interval_cmd = f'date_trunc(\'hour\', d) + floor(extract({timeframe} from d)/ {interval_len}) * interval \'{interval_len} {timeframe}\''
            command = f'select {interval_cmd} as d, extract(epoch from {interval_cmd}) as d_sec, (array_agg(trade.p order by trade.d asc))[1] o, max(trade.p) h, min(trade.p) l, (array_agg(trade.p order by trade.d desc))[1] c, sum(v) v, count(*) t from {ticker}_trade as trade group by {interval_cmd} order by d desc'
            super().query(f'create table if not exists {ticker}_{interval_len} as {command}')
            self.order_table(f'{ticker}_{interval_len}', 'd')
        elif timeframe == 'hour':
            interval_cmd = f'date_trunc(\'day\', d) + floor(extract({timeframe} from d)/ {interval_len}) * interval \'{interval_len} {timeframe}\''
            command = f'select {interval_cmd} as d, (array_agg(trade.p order by trade.d asc))[1] o, max(trade.p) h, min(trade.p) l, (array_agg(trade.p order by trade.d desc))[1] c, sum(v) v, count(*) t from {ticker}_trade as trade group by {interval_cmd} order by d desc'
            super().query(f'create table if not exists {ticker}_{interval_len*60} as {command}')
            self.order_table(f'{ticker}_{interval_len*60}', 'd')

    def add_model_table_from_df(self, df, table_name):
        create_table = f'create table if not exists {table_name} ({df.index.name} bigint, '
        insert_into = f'insert into {table_name} ({df.index.name}, '
        for column in df.columns:
            create_table += f'{column} double precision, '
            insert_into += f'{column}, '
        create_table = create_table[:-2] + ')'
        insert_into = insert_into[:-2] + ') values '
        super().query(create_table)
        for index, row in df.iterrows():
            line = str(list(row.values))
            line = f'({index}, {line[1: -1].replace("nan", "null")}), '
            insert_into += line
        insert_into = insert_into[:-2]
        super().query(insert_into)

    def add_column_quickly_w_index(self, table_name, values, indexes, index_col_name, index_type, column_name, column_type):
        super().query(f'create temp table {column_name}_temp ({index_col_name} {index_type}, new {column_type})')
        command = f'insert into {column_name}_temp ({index_col_name}, new) values'
        for index, value in zip(indexes, values):
            if str(value) != 'nan':
                command += f'(\'{index}\', {value}), '
            else:
                command += f'(\'{index}\', null), '
        command = command[:-2]
        super().query(command)
        super().query(f'alter table {table_name} add column if not exists {column_name} {column_type}')
        super().query(f'update {table_name} set {column_name} = new from {column_name}_temp where {table_name}.{index_col_name} = {column_name}_temp.{index_col_name}')
        super().query(f'drop table {column_name}_temp')
    
    def add_single_rsi(self, close_df, window):
        return ta.momentum.rsi(close_df, window)

    def add_single_macd_diff(self, close_df, window):
        return ta.trend.MACD(close_df, window[0], window[1], window[2]).macd_diff()
    
    def add_single_bb(self, close_df, window):
        bb = ta.volatility.BollingerBands(close_df, window)
        return bb.bollinger_pband()


    def order_table(self, table_name, order_by_column):
        super().query(f'create table if not exists copy_table as select * from {table_name} order by {order_by_column} asc')
        super().query(f'drop table {table_name}')
        super().query(f'create table {table_name} as select * from copy_table order by {order_by_column} asc')
        super().query(f'drop table copy_table')
        # super().query(f'copy temp({column_name}) from \'temp.csv\'')


