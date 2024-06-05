# this is moddified from my own code project 6: maketsimcode.py

import pandas as pd
from util import get_data
import datetime as dt
import yfinance as yf

def compute_portvals(
        stock,
        orders,
        start_val=1000000,
        commission=9.95,
        impact=0.005,

):


    # ticker = yf.Ticker(stock)
    # data = ticker.history(period="500d")

    # data = data["Close"]
    # print(data)
    # print(data.index.values)
    # data["Close"]

    # data = data.to_frame()
    # data.rename(columns={data.columns[0]: stock}, inplace=True)
    # print(data)
    # prices = pd.DataFrame(index=data.index.values, dtype=float)
    # prices = data.copy()
    # prices[stock] = data["Close"]
    stocks = [stock]
    # print(data["Close"])
    # prices[stock] = data["Close"]
    # print(prices)
    prices = get_data(stocks, pd.date_range(orders.index[0], orders.index[-1]), addSPY=True)
    prices.fillna(method="ffill", inplace=True)
    prices.fillna(method="bfill", inplace=True)


    records = pd.DataFrame(index=prices.index.values, columns=prices.columns.values, dtype=float)
    # records = pd.DataFrame(index=prices.index.values, dtype=float)


    # print(records)
    records["CASH"] = 0.0
    records = records.fillna(0.0)

    total_orders = orders.shape[0]
    for i in range(total_orders):
        date = orders.index[i]
        shares = orders.iloc[i, 0]

        records.loc[date, stock] += shares
        records.loc[date, "CASH"] -= prices.loc[date, stock] * shares

        if commission > 0:
            records.loc[date, "CASH"] -= commission
        if impact > 0:
            records.loc[date, "CASH"] -= prices.loc[date, stock] * shares * impact





    portfolio = records.copy()
    portfolio.iloc[0, -1] += start_val
    portfolio = portfolio.cumsum()
    portfolio["Total"] = 0.0
    total_portfolio = portfolio.shape[0]
    for i in range(total_portfolio):
        for j in range(portfolio.shape[1] - 2):
            portfolio.iloc[i, -1] += portfolio.iloc[i, j] * prices.iloc[i, j]
        portfolio.iloc[i, -1] += portfolio.iloc[i, -2]




    return portfolio.iloc[:, -1]



def author():
    return "gwang383"
