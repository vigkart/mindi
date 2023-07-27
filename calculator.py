import pandas as pd
import os
import pandas_market_calendars as mcal

clean_data_dir = "trading_data/clean_data"
filename = f"{clean_data_dir}/47_clean.csv"
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

print(df)
