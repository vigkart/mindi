import pandas as pd
import os
import pandas_market_calendars as mcal
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

clean_data_dir = "trading_data/clean_data"
filename = f"{clean_data_dir}/46_clean.csv"
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

# Creating stats dataframe
stats = ['interval_start', 'interval_end', 'avg_pnl', 'num_trades', 'total_pnl', 'pl_ratio']
stats_df = pd.DataFrame(index=range(trading_dates.get_loc(pd.Timestamp(end_date).date())))

# method to aggregate the stats for a given interval
def aggregate(df, start, end, interval):
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

    # prev_day is the day trailing stats are aggregated for, and the current day is prev_day + 1
    prev_day = df['exit_day_num'][start]
    try:
        stats_df['day'][prev_day] = trading_dates[df['exit_day_num'][start + 1]].strftime("%Y-%m-%d")
    except KeyError:
        stats_df['day'][prev_day] = np.nan
    # stats_df['interval_start'][prev_day] = trading_dates[df['exit_day_num'][start]].strftime("%Y-%m-%d")
    # stats_df['interval_end'][prev_day] = trading_dates[df['exit_day_num'][end]].strftime("%Y-%m-%d")
    stats_df[f'avg_pnl_last{interval}D'][prev_day] = avg_pnl
    stats_df[f'num_trades_last_{interval}D'][prev_day] = num_trades
    stats_df[f'total_pnl_last_{interval}D'][prev_day] = total_pnl
    stats_df[f'pl_ratio_last{interval}D'][prev_day] = pl_ratio


# Loop allows us to calculate moving averages/sums of the statistics for any interval (smallest interval is 1D)
granularity = 3

i = len(df['exit_day_num']) - 1
while i > 0:
    days = {}
    start = i
    while len(days) <= granularity and i >= 0:
        if df['exit_day_num'][i] not in days:
            days[df['exit_day_num'][i]] = i
        i -= 1 

    try: 
        aggregate(df, start, i + 2, granularity)
        i = days[list(days)[1]] 
    except IndexError:
        # if I want to here I can aggregate all of the statistics at the beginning
        break




# Plotting stuff:

# stats_df['interval_start'] = pd.to_datetime(stats_df['interval_start'])
print(stats_df)
# plt.plot(stats_df['interval_start'], stats_df['avg_pnl'])

# corr_df = stats_df.drop(['interval_start', 'interval_end'], axis=1)

# corr = corr_df.astype('float').corr()
# heatmap = sns.heatmap(corr)

# plt.show()