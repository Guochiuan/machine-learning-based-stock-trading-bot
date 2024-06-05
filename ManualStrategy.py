import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf

import indicators as ind
from marketsimcode import compute_portvals
from util import get_data


def calculateIndicators(symbol='AAPL', target =33, sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31)):
    dates = pd.date_range(sd, ed)[0:-1]

    # print(dates)

    # print(dates[-1])
    prices = get_data([symbol], dates, addSPY=False)
    prices.fillna(method="ffill", inplace=True)
    prices.fillna(method="bfill", inplace=True)


    data = pd.DataFrame(index=prices.index.values, dtype=float)
    data['price'] = prices

    new_datetime = data.index[-1:]  # current end of datetime index
    increment = '1 days'  # string for increment - eventually will be in a for loop to add add'l days
    new_datetime = new_datetime + pd.Timedelta(increment)




    # print(data)


    today_df = pd.DataFrame({'price': target}, index=new_datetime)
    data = pd.concat([data, today_df])
    # print(data)

    # print(data)


    # data.append(pd.DataFrame(index='2012-1-1'))
    # print(data)






    rsi = ind.createRSI(data)
    bb = ind.createBollinger(data)
    so = ind.createStochasticOscillator(data)
    macd = ind.createMACD(data)
    gdc = ind.createGoldenDeathCross(data)
    data['bbp'] = bb['PERCENT']
    data['rsi'] = rsi['RSI']
    data['so'] = so['D']
    data['macd'] = macd['MACD']
    data['macd_diff'] = data['macd'].diff(1)
    data['macd_signal'] = macd['SIGNAL']
    data['sma20'] = gdc['SMA20']
    data['sma20_diff'] = data['sma20'].diff(1)
    data['sma50'] = gdc['SMA50']

    # print(data)

    data_without_nan = data.copy()
    data_without_nan = data_without_nan[~data_without_nan.isnull().any(axis=1)]

    # print(data_without_nan)
    # print(data_without_nan.iloc[-1])
    # print(data.shape)

    buy_count = 0
    sell_count = 0
    if (data_without_nan.iloc[-1, 1] < 0.05):
        buy_count += 1
    if (data_without_nan.iloc[-1, 1] > 0.95):
        sell_count += 1
    if (data_without_nan.iloc[-1, 2] < 33):
        buy_count += 1
    if (data_without_nan.iloc[-1, 2] > 67):
        sell_count += 1
    if (data_without_nan.iloc[-1, 3] < 20):
        buy_count += 1
    if (data_without_nan.iloc[-1, 3] > 80):
        sell_count += 1

    # print('4  ', data_without_nan.iloc[-1, 4])
    # print('5  ', data_without_nan.iloc[-1, 5])
    # print('6  ', data_without_nan.iloc[-1, 6])
    if (data_without_nan.iloc[-1, 5] > 0) and (data_without_nan.iloc[-1, 4] > data_without_nan.iloc[-1, 6]):
        buy_count += 1
        print('MACD BUY')
    if (data_without_nan.iloc[-1, 5] < 0) and (data_without_nan.iloc[-1, 4] < data_without_nan.iloc[-1, 6]):
        sell_count += 1
        print('MACD SELL')
    if (data_without_nan.iloc[-1, 8] > 0) and (data_without_nan.iloc[-1, 7] > data_without_nan.iloc[-1, 9]):
        buy_count += 1
        print('Golden Death Cross BUY')
    if (data_without_nan.iloc[-1, 8] < 0) and (data_without_nan.iloc[-1, 7] < data_without_nan.iloc[-1, 9]):
        sell_count += 1
        print('Golden Death Cross SELL')


    if (data_without_nan.iloc[-1, 1] < 0.05) and (data_without_nan.iloc[-1, 2] < 33) and (data_without_nan.iloc[-1, 3] < 20):
        print('BUY signal is given')

    elif (data_without_nan.iloc[-1, 1] > 0.95) and (data_without_nan.iloc[-1, 2] > 67) and (data_without_nan.iloc[-1, 3] > 80):
        print('SELL signal is given')

    # print("bbp range: 0.05 - 0.95")
    # print("rsi range: 33 - 67")
    # print("so range: 20 - 80")
    # o1 = "macd range: macd going up and cross signl line from below is buy, macd going down and cross signal line from above is sell"
    #
    # print("macd range: macd going up and cross signl line from below is buy, macd going down and cross signal line from above is sell")
    # print("gold death cross : SMA-20 going up and cross SMA-50 from below is buy, SMA-20 going down and cross SMA-50 from above is sell")


    # red
    # print("\033[91m {}\033[00m".format(o1))


    # green
    # print("\033[92m {}\033[00m".format(o1))

    print("buy count: " + str(buy_count))
    print("sell count: " + str(sell_count))




def testPolicy(days = "500d",symbol='AAPL', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):

    # ticker = yf.Ticker(symbol)
    # data = ticker.history(period=days)
    #
    # data = data["Close"]
    #
    # prices = data.to_frame()
    # prices.rename(columns={prices.columns[0]: symbol}, inplace=True)


    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=False)
    prices.fillna(method="ffill", inplace=True)
    prices.fillna(method="bfill", inplace=True)
    # prices = prices.drop(columns=["SPY"])

    data = pd.DataFrame(index=prices.index.values, dtype=float)
    data['price'] = prices






########## ADDDD  NEW DATA HERE


    new_datetime = data.index[-1:]  # current end of datetime index
    increment = '1 days'  # string for increment - eventually will be in a for loop to add add'l days
    new_datetime = new_datetime + pd.Timedelta(increment)
    today_df = pd.DataFrame({'price': 33}, index=new_datetime)
    # data = pd.concat([data, today_df])
    # print(data)




    # data.append(pd.DataFrame(index='2012-1-1'))
    # print(data)






    rsi = ind.createRSI(prices)
    bb = ind.createBollinger(prices)
    so = ind.createStochasticOscillator(prices)
    data['bbp'] = bb['PERCENT']
    data['rsi'] = rsi['RSI']
    data['so'] = so['D']


    data_without_nan = data.copy()
    data_without_nan = data_without_nan[~data_without_nan.isnull().any(axis=1)]

    # print(data_without_nan)
    # print(data_without_nan.iloc[-1])
    ########## ADDDD  NEW ABOVE


    holding = 0

    trades = pd.DataFrame(index=data_without_nan.index.values, columns=['trade'], dtype=float)



    for i in range(0, trades.shape[0]):
        sign = 0
        if (data_without_nan.iloc[i, 1] < 0.05) and (data_without_nan.iloc[i, 2] < 33) and (data_without_nan.iloc[i, 3] < 20):
            sign = 1

        elif (data_without_nan.iloc[i, 1] > 0.95) and (data_without_nan.iloc[i, 2] > 67) and (data_without_nan.iloc[i, 3] > 80):
            sign = -1



        trades.iloc[i, 0] = 0
        if holding == 1000:
            if sign == 1:
                trades.iloc[i, 0] = 0
                holding = 1000

            elif sign == -1:
                trades.iloc[i, 0] = -2000
                holding = -1000

        elif holding == -1000:
            if sign == 1:
                trades.iloc[i, 0] = 2000
                holding = 1000

            elif sign == -1:
                trades.iloc[i, 0] = 0
                holding = -1000


        elif holding == 0:
            if sign == 1:
                trades.iloc[i, 0] = 1000
                holding = 1000

            elif sign == -1:
                trades.iloc[i, 0] = -1000
                holding = -1000




    return trades


def createChartAndStatistics():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    start_val = 100000
    commission = 9.95
    impact = 0.005
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=True)
    trades = testPolicy(symbol=symbol, sd=sd, ed=ed, sv=start_val)
    df = compute_portvals(symbol, trades, start_val=start_val, commission=commission, impact=impact)
    output = pd.DataFrame(index=prices.index.values, dtype=float)
    output['total'] = df
    output.fillna(method="bfill", inplace=True)


    normalized_total = output / output.iloc[0]
    normalized_total = normalized_total['total']
    benchmark = pd.DataFrame(index=prices.index.values, dtype=float)

    benchmark['total'] = 0
    benchmark.iloc[0,0] = 1000
    benchmark_total = compute_portvals(symbol, benchmark, start_val=start_val, commission=commission, impact=impact)


    normalized_benchmark_total = benchmark_total / benchmark_total.iloc[0]
    plt.plot(normalized_benchmark_total, "purple")
    plt.plot(normalized_total, "red")



    for i in range(0, trades.shape[0]):
        if trades.iloc[i,0] > 0:
            plt.axvline( x=trades.iloc[i].name, color='blue')
        elif trades.iloc[i,0] < 0:
            plt.axvline(x=trades.iloc[i].name, color='black')


    plt.legend(labels=['Benchmark', 'Manual','Sell', 'Buy'])
    plt.ylabel("Normalized Portfolio Value")
    plt.xlabel("Date")
    plt.gcf().autofmt_xdate()
    plt.title("Manual Strategy vs Benchmark on JPM for in-sample data")
    plt.savefig("manual_strategy_in_sample.png")
    plt.close()



    cumulative_manual_strategy_return_1 = round((normalized_total.iloc[-1] - normalized_total.iloc[0]) / normalized_total.iloc[0], 6)
    cumulative_benchmark_return_1 = round((normalized_benchmark_total.iloc[-1] - normalized_benchmark_total.iloc[0]) / normalized_benchmark_total.iloc[0],6)

    daily_manual_strategy_rets_1 = normalized_total[1:] / normalized_total[:-1].values - 1
    daily_benchmark_rets_1 = normalized_benchmark_total[1:] / normalized_benchmark_total[:-1].values - 1

    std_benchmark_1 = round(daily_benchmark_rets_1.std(), 6)
    std_manual_strategy_1 = round(daily_manual_strategy_rets_1.std(), 6)

    mean_benchmark_1 = round(daily_benchmark_rets_1.mean(), 6)
    mean_manual_strategy_1 = round(daily_manual_strategy_rets_1.mean(), 6)


# out-of-sample

    symbol = 'JPM'
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    start_val = 100000
    commission = 9.95
    impact = 0.005
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=True)
    trades = testPolicy(symbol=symbol, sd=sd, ed=ed, sv=start_val)
    df = compute_portvals(symbol, trades, start_val=start_val, commission=commission, impact=impact)
    output = pd.DataFrame(index=prices.index.values, dtype=float)
    output['total'] = df
    output.fillna(method="bfill", inplace=True)

    normalized_total = output / output.iloc[0]
    normalized_total = normalized_total['total']
    benchmark = pd.DataFrame(index=prices.index.values, dtype=float)

    benchmark['total'] = 0
    benchmark.iloc[0,0] = 1000
    benchmark_total = compute_portvals(symbol, benchmark, start_val=start_val, commission=commission, impact=impact)

    normalized_benchmark_total = benchmark_total / benchmark_total.iloc[0]

    # print("final value: ")
    # print(normalized_total[-1])

    plt.plot(normalized_benchmark_total, "purple")
    plt.plot(normalized_total, "red")



    for i in range(0, trades.shape[0]):
        if trades.iloc[i,0] > 0:
            plt.axvline( x=trades.iloc[i].name, color='blue')
        elif trades.iloc[i,0] < 0:
            plt.axvline(x=trades.iloc[i].name, color='black')


    plt.legend(labels=['Benchmark', 'Manual','Sell', 'Buy'])
    plt.ylabel("Normalized Portfolio Value")
    plt.xlabel("Date")
    plt.gcf().autofmt_xdate()
    plt.title("Manual Strategy vs Benchmark on JPM for out-of-sample data")
    plt.savefig("manual_strategy_out_of_sample.png")
    plt.close()



    cumulative_manual_strategy_return_2 = round((normalized_total.iloc[-1] - normalized_total.iloc[0]) / normalized_total.iloc[0], 6)
    cumulative_benchmark_return_2 = round((normalized_benchmark_total.iloc[-1] - normalized_benchmark_total.iloc[0]) / normalized_benchmark_total.iloc[0],6)

    daily_manual_strategy_rets_2 = normalized_total[1:] / normalized_total[:-1].values - 1
    daily_benchmark_rets_2 = normalized_benchmark_total[1:] / normalized_benchmark_total[:-1].values - 1

    std_benchmark_2 = round(daily_benchmark_rets_2.std(), 6)
    std_manual_strategy_2 = round(daily_manual_strategy_rets_2.std(), 6)

    mean_benchmark_2 = round(daily_benchmark_rets_2.mean(), 6)
    mean_manual_strategy_2 = round(daily_manual_strategy_rets_2.mean(), 6)





















    output = pd.DataFrame({"Cumulative Return": [cumulative_benchmark_return_1, cumulative_manual_strategy_return_1, cumulative_benchmark_return_2, cumulative_manual_strategy_return_2],
                           "STD of Daily Returns": [std_benchmark_1, std_manual_strategy_1, std_benchmark_2, std_manual_strategy_2],
                           "Mean of Daily Returns": [mean_benchmark_1, mean_manual_strategy_1, mean_benchmark_2, mean_manual_strategy_2]},
                          index=["Benchmark_in_sample", "Manual_Strategy_in_sample", "Benchmark_out_of_sample", "Manual_Strategy_out_of_sample"])



    # print(output)
    file = open("manual_strategy_statistics.txt", "w")
    file.write(str(pd.DataFrame(output)))
    file.close()




if __name__ == "__main__":
    stock_list = ['TSLA', 'PLTR', 'NIO', 'IONQ', 'F', 'AMD', 'NVDA', 'AAPL', 'KVUE', 'AMZN', 'PLUG', 'MPW', 'BAC',
                  'RIVN', 'SOFI', 'BABA', 'LCID', 'MARA', 'JNJ', 'DNA', 'CCL', 'DIS','SPY', 'QQQ', 'PCG', 'MSFT','GOOG']
    today = dt.datetime.today()
    # print(today)
    for tickerSymbol in stock_list:
    # tickerSymbol = 'AAPL'

    # Get data on this ticker
    # tickerData = yf.Ticker(tickerSymbol)
    # price = tickerData.info['regularMarketPrice']
    # print(price)
    # Get the current price
    # current_price = tickerData.info['regularMarketPrice']
    # print('Current Price: ' + str(current_price))

    # yca = yf.Ticker("AAPL").history(interval="1m", period="1d")
    # ss = yca['Close'][-1]
    # print(ss)

    # tickerSymbol = 'CCL'


        tickerData = yf.Ticker(tickerSymbol)
        todayData = tickerData.history(period='1d')
        lastPrice = todayData['Close'][0]

        # tickerData = yf.Ticker(tickerSymbol)
        # todayData = tickerData.history( interval="1m",period='1d')
        # # todayData = tickerData.history(period='1d')
        # # print(todayData)
        # ss = todayData['Close'][-1]
        # print(ss)

    ###### test with and without interval = 1m   ['Close'][-1] and ['Close'][0]  ######

        # print(tickerSymbol)
        print("\033[92m {}\033[00m".format(tickerSymbol))
        calculateIndicators(symbol=tickerSymbol, target=lastPrice, sd=dt.datetime(2023, 1, 1), ed=today)

    # calculateIndicators(symbol='AAPL', target=181.99, sd=dt.datetime(2023, 1, 1), ed=dt.datetime(2023, 8, 5))
    # testPolicy()
    # trades = testPolicy(symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # createChartAndStatistics()
    # testPolicy(symbol='JPM', sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)
    # print(output)






