import pandas as pd
import os
import pandas_market_calendars as mcal

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


def aggregate(df, start, end):
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
    
    print(f"winners: {winners}, losers: {losers}, gains: {gains}, losses: {losses}")
    pl_ratio = (gains/winners)/(losses/losers)
    avg_pnl = total_pnl/num_trades
    print(f"your pl ratio is {pl_ratio} and your avg pnl is {avg_pnl}")
    print(f"start: {start}, end {end}")


# Aggregating statistics from the latest trade to the earliest trade
granularity = 1

i = len(df['exit_day_num']) - 1
while i > 0:
    days = {}
    start = i
    while len(days) <= granularity and i >= 0:
        if df['exit_day_num'][i] not in days:
            days[df['exit_day_num'][i]] = i
        i -= 1 

    aggregate(df, start, i + 2)

    try: 
        i = days[list(days)[granularity]] 
    except IndexError:
        # if I want to here I can aggregate all of the statistics at the beginning
        break