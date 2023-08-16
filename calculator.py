import pandas as pd
# import os
import pandas_market_calendars as mcal
import numpy as np

def get_trading_dates(algo_num):
    '''gets valid trading dates and adds trading day index to clean data
        returns: df, list
            df of clean dates with day index, list of valid trading days for interval'''

    clean_data_dir = "trading_data/clean_data"
    filename = f"{clean_data_dir}/{algo_num}_clean.csv"
    if filename.endswith('_matches.csv'):
        algo_number = filename.split('_')[0]
    df = pd.read_csv(filename)

    # Getting date range for cleaned pdq-output
    start = df['first_entry_time'].min()
    end = df['last_exit_time'].max()

    # Getting market calendar for date range from above, then creating dict of valid market days and day number
    nyse = mcal.get_calendar('NYSE')
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_dates = pd.Index(mcal.date_range(schedule, frequency='1D').date)

    try:
        entry_day_num = df['first_entry_time'].apply(lambda x: trading_dates.get_loc(pd.Timestamp(x).date()))
        exit_day_num = df['last_exit_time'].apply(lambda x: trading_dates.get_loc(pd.Timestamp(x).date()))
        df['entry_day_num'] = entry_day_num
        df['exit_day_num'] = exit_day_num
    except KeyError as e:
        print(f"this trading data set has trades on {e.args[0]}, which is not registered as a trading day")

    return df, trading_dates


def aggregate(df, start, end, interval, stats_df, trading_dates):
    '''Aggregates trailing stats

    Aggregates stats for interval starting from latest date to earliest date. 
    Average pnl (pnl per trade), sum of trades, sum of profits, pl ratio
    
    Parameters:
        df: Dataframe
            Cleaned trades df with dates
        start: int
            latest day of interval
        end: int
            earliest day of interval
        interval: int
            number of days in interval 

    Returns:
        stats_df: dataframe
            day | total profit | trailing stats for each interval
    '''

    total_pnl = 0
    num_trades = start - end + 1 # each row is one trade & going backwards, so end - start is number of trades (zero based index)
    winners = 0 # number of winning trades
    losers = 0 # number of losing trades
    gains = 0
    losses = 0
    i = start
    while i >= end:
        total_pnl += df['pnl'][i]
        if df['pnl'][i] > 0:
            winners += 1
            gains += df['pnl'][i]
        elif df['pnl'][i] < 0:
            losers += 1
            losses += df['pnl'][i]
        i -= 1
        if i < 0:
            print(i)
            raise IndexError
    
    if winners == 0 and losers == 0: # if all trades for interval break even
        pl_ratio = 0
    else:
        pl_ratio = (gains/winners)/(losses/losers)
    avg_pnl = total_pnl/num_trades

    # index is the day trailing stats are aggregated for, and the current day is index + 1
    index = df['exit_day_num'][start]

    if interval == 1:
        stats_df['pnl'][index - 1] = total_pnl # writing this to previous row because need curr day pnl
        try:
            # adding current date to stats_df (bc indexing df at start + 1)
            stats_df['day'][index] = trading_dates[df['exit_day_num'][start + 1]].strftime("%Y-%m-%d")
        except KeyError:
            stats_df['day'][index] = np.nan
        


    stats_df[f'avg_pnl_last_{interval}D'][index] = avg_pnl
    stats_df[f'num_trades_last_{interval}D'][index] = num_trades
    stats_df[f'pnl_last_{interval}D'][index] = total_pnl
    stats_df[f'pl_ratio_last_{interval}D'][index] = pl_ratio

    return stats_df

# Creating stats dataframe

# TODO: Aggregate standard deviation for trailing intervals

def build_features(intervals, df, trading_dates):
    '''Builds stats_df that has {day | pnl | trailing stats for various intervals}'''

    # Initializing stats dataframe
    stats = ['day', 'pnl']
    for i in intervals:
        stats.append(f'avg_pnl_last_{i}D')
        stats.append(f'num_trades_last_{i}D')
        stats.append(f'pnl_last_{i}D')
        stats.append(f'pl_ratio_last_{i}D')
    stats_df = pd.DataFrame(index=range(len(trading_dates)), columns=stats)


    # Loop allows us to calculate moving averages/sums of the statistics for any interval (smallest interval is 1D)
    for interval in intervals: 
        granularity = interval

        i = len(df['exit_day_num']) - 1
        while i > 0:
            days = {}
            start = i
            while len(days) <= granularity and i >= 0:
                if df['exit_day_num'][i] not in days:
                    days[df['exit_day_num'][i]] = i
                i -= 1 
            if len(days) == granularity + 1 or len(days) == granularity:
                try: 
                    stats_df = aggregate(df, start, i + 2, granularity, stats_df, trading_dates)
                    i = days[list(days)[1]] 
                except IndexError:
                    # if I want to here I can aggregate all of the statistics at the beginning
                    break
            elif len(days) < granularity:
                i = -1
    return stats_df

df, trading_dates = get_trading_dates(46)
intervals = [1, 3, 5, 10, 21] # [1, 3, 5, 10, 21, 200] these are the real intervals, but we don't have data for 200 days yet

output = build_features(intervals, df, trading_dates)
output.to_csv('training_data/sample_output3.csv', index=False)