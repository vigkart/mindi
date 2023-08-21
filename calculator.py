import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import os
import math
import time


def get_trading_dates(algo_num):
    '''gets valid trading dates and adds trading day index to clean data
        returns: df, list
            df of clean dates with day index, list of valid trading days for interval'''

    clean_data_dir = r'trading_data/clean_data'
    file = str(algo_num) + '_clean.csv'
    filename = os.path.join(clean_data_dir, file)
    # clean_data_dir = "trading_data/clean_data"
    # filename = f"{clean_data_dir}/{algo_num}_clean.csv"
    df = pd.read_csv(filename, usecols=['entry_time', 'entry_time', 'pnl'])

    # Getting date range for cleaned pdq-output
    start = df['entry_time'].min()
    end = df['entry_time'].max()

    # Getting market calendar for date range from above, then creating dict of valid market days and day number
    nyse = mcal.get_calendar('NYSE')
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end)
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_dates = pd.Index(mcal.date_range(schedule, frequency='1D').date)

    try:
        entry_day_num = df['entry_time'].apply(lambda x: trading_dates.get_loc(pd.Timestamp(x).date()))
        exit_day_num = df['entry_time'].apply(lambda x: trading_dates.get_loc(pd.Timestamp(x).date()))
        df['entry_day_num'] = entry_day_num
        df['exit_day_num'] = exit_day_num
    except KeyError as e:
        print(f"this trading data set has trades on {e.args[0]}, which is not registered as a trading day")

    return df, trading_dates


def aggregate(df, start, end, interval, stats_df, trading_dates, algo_num):
    '''Aggregates trailing stats

    Aggregates stats for interval starting from latest date to earliest date. 
    Average pnl (pnl per trade), sum of trades, sum of profits, pl ratio, standard deviation (std)
    
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
    
    pl_ratio = 0
    if (gains == 0 and losses == 0) or (winners == 0 and losers == 0): # if all trades for interval break even
        pl_ratio = 0
    elif (gains != 0 and losses == 0) or (winners != 0 and losers == 0):
        pl_ratio == gains/winners
    elif (gains == 0 and losses != 0) or (winners == 0 and losers != 0):
        pl_ratio == losses/losers
    else:
        pl_ratio = (gains/winners)/(losses/losers)
    
    avg_pnl = total_pnl/num_trades
    
    # Calculate trailing standard deviation (std)
    i = start
    sum = 0
    while i >= end:
        sum += (df['pnl'][i] - avg_pnl) ** 2
        i -= 1

    std = math.sqrt(sum/num_trades)

    # index is the day trailing stats are aggregated for, and the current day is index + 1
    try:
        index = df['exit_day_num'][start + 1]
    except: # This means is the last day with data
        return stats_df

    # Writing current day PnL (not trailing), and Writing Date to Dataframe
    if interval == 1:
        stats_df[f'{algo_num}_pnl'][index - 1] = total_pnl # writing this to previous row because need curr day pnl
        try:
            # adding current date to stats_df (bc indexing df at start + 1)
            stats_df[f'{algo_num}_day'][index] = trading_dates[df['exit_day_num'][start + 1]].strftime("%Y-%m-%d")
        except KeyError:
            stats_df[f'{algo_num}_day'][index] = np.nan
        

    # Writing features to features dataframe
    stats_df[f'{algo_num}_avg_pnl_last_{interval}D'][index] = avg_pnl
    stats_df[f'{algo_num}_num_trades_last_{interval}D'][index] = num_trades
    stats_df[f'{algo_num}_pnl_last_{interval}D'][index] = total_pnl
    stats_df[f'{algo_num}_pl_ratio_last_{interval}D'][index] = pl_ratio
    stats_df[f'{algo_num}_std_last_{interval}D'][index] = std

    return stats_df

# Creating stats dataframe
def build_features(intervals, df, trading_dates, algo_num):
    '''Builds stats_df that has {day | pnl | trailing stats for various intervals}'''

    # Initializing stats dataframe
    stats = [f'{algo_num}_day', f'{algo_num}_pnl']
    for i in intervals:
        stats.append(f'{algo_num}_avg_pnl_last_{i}D')
        stats.append(f'{algo_num}_num_trades_last_{i}D')
        stats.append(f'{algo_num}_pnl_last_{i}D')
        stats.append(f'{algo_num}_pl_ratio_last_{i}D')
        stats.append(f'{algo_num}_std_last_{i}D')
    stats_df = pd.DataFrame(index=trading_dates, columns=stats)


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
                    stats_df = aggregate(df, start, i + 2, granularity, stats_df, trading_dates, algo_num)
                    i = days[list(days)[1]] 
                except IndexError:
                    # if I want to here I can aggregate all of the statistics at the beginning
                    break
            elif len(days) < granularity:
                i = -1
    return stats_df

start = time.time()
algo_num = 2
training_data_dir = r'training_data'
output_name = str(algo_num) + '_features.csv'
output_name = os.path.join(training_data_dir, output_name)

try:
    df, trading_dates = get_trading_dates(algo_num)
    intervals = [1, 3, 5, 10, 21] # [1, 3, 5, 10, 21, 200] these are the real intervals, but we don't have data for 200 days yet

    output = build_features(intervals, df, trading_dates, algo_num)
    output.to_csv(output_name, index=True)

except Exception as e:
    print(f'Error Occured:\n{e}')
    raise e

finally:
    end = time.time()
    duration = end - start
    print(f'\nStart: {start}, End: {end}, Duration: {duration}\n')