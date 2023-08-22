import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import os
import math
import time


def get_trading_dates(algo_num):
    clean_data_dir = r"trading_data/clean_data"
    file = str(algo_num) + "_clean.csv"
    filename = os.path.join(clean_data_dir, file)
    df = pd.read_csv(filename, usecols=["entry_time", "exit_time", "pnl"])
    df["entry_time"] = pd.to_datetime(df["entry_time"]).dt.date
    df["exit_time"] = pd.to_datetime(df["exit_time"]).dt.date

    # Getting market calendar for date range from above, then creating Index of valid market days
    nyse = mcal.get_calendar("NYSE")
    start_date = df["exit_time"].min()
    end_date = df["exit_time"].max()
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    trading_dates = pd.Index(mcal.date_range(schedule, frequency="1D").date)

    return df, trading_dates


def aggregate(algo_num, stats_df, subframe, interval, trading_dates, index):
    try:
        next_day = trading_dates[index + 1]
    except IndexError:
        return stats_df

    if index < interval:
        return stats_df

    if subframe.empty:  # if no trades made in last interval of days
        num_trades = 0
        total_pnl = 0
        avg_pnl = 0
        win_avg = 0
        loss_avg = 0
        std = 0

    else:
        num_trades = subframe.shape[0]  # Each row represents one trade
        total_pnl = subframe["pnl"].sum()
        avg_pnl = total_pnl / num_trades
        std = subframe["pnl"].std()

        # Find win and loss avgs
        winners = subframe[subframe["pnl"] > 0]
        losers = subframe[subframe["pnl"] < 0]
        num_winners = winners.shape[0]
        num_losers = losers.shape[0]
        gains = winners["pnl"].sum()
        losses = losers["pnl"].sum()

        if num_winners != 0:
            win_avg = gains / num_winners
        else:
            win_avg = 0

        if num_losers != 0:
            loss_avg = losses / num_losers
        else:
            loss_avg = 0

    # Writing PnL for current day
    if interval == 1:
        stats_df[f"{algo_num}_pnl"][trading_dates[index]] = total_pnl

    # Writing trailing stats
    stats_df[f"{algo_num}_avg_pnl_last_{interval}D"][next_day] = avg_pnl
    stats_df[f"{algo_num}_num_trades_last_{interval}D"][next_day] = num_trades
    stats_df[f"{algo_num}_pnl_last_{interval}D"][next_day] = total_pnl
    stats_df[f"{algo_num}_win_avg_last_{interval}D"][next_day] = win_avg
    stats_df[f"{algo_num}_loss_avg_last_{interval}D"][next_day] = loss_avg
    stats_df[f"{algo_num}_std_last_{interval}D"][next_day] = std

    return stats_df


def build_features(df, trading_dates, intervals):
    # Initializing stats dataframe
    stats = [f"{algo_num}_pnl"]
    for i in intervals:
        stats.append(f"{algo_num}_avg_pnl_last_{i}D")
        stats.append(f"{algo_num}_num_trades_last_{i}D")
        stats.append(f"{algo_num}_pnl_last_{i}D")
        stats.append(f"{algo_num}_win_avg_last_{i}D")
        stats.append(f"{algo_num}_loss_avg_last_{i}D")
        stats.append(f"{algo_num}_std_last_{i}D")
    stats_df = pd.DataFrame(index=trading_dates, columns=stats)

    for interval in intervals:
        i = len(trading_dates)
        while i > 0:
            granularity = interval
            days = trading_dates[i - granularity : i]
            subframe = df[df["exit_time"].isin(days)]
            stats_df = aggregate(
                algo_num, stats_df, subframe, interval, trading_dates, i
            )
            i -= 1

    return stats_df


# Grabbing the desired file to be processed
start = time.time()
algo_num = 46
features_data_dir = r"features_data"
output_name = str(algo_num) + "_features.csv"
output_name = os.path.join(features_data_dir, output_name)

# Driver for the processing
try:
    df, trading_dates = get_trading_dates(algo_num)
    intervals = [
        1,
        3,
        5,
        10,
        21,
    ]  # [1, 3, 5, 10, 21, 200] these are the real intervals, but we don't have data for 200 days yet
    output = build_features(df, trading_dates, intervals)
    output.to_csv(output_name, index=True)

except Exception as e:
    print(f"Error Occured:\n{e}")
    raise e

finally:
    end = time.time()
    duration = end - start
    print(f"\nFinished processing algo number {algo_num}, Duration: {duration}\n")
