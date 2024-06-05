import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from marketsimcode import compute_portvals
from ManualStrategy import testPolicy
import StrategyLearner as sl
import Q_StrategyLearner as sl2
from util import get_data
import numpy as np
import yfinance as yf
def author():
    return "gwang383"


def createChartAndStatisticsForExperiment1( days= "500d", stock = 'JPM', sd1= dt.datetime(2008, 1, 1), ed1= dt.datetime(2009, 12, 31),
                                           sd2=dt.datetime(2010, 1, 1), ed2=dt.datetime(2011, 12, 31),
                                           commission=0, impact = 0.000):
    symbol = stock
    sd = sd1
    ed = ed1
    start_val = 100000



    # ticker = yf.Ticker(stock)
    # data = ticker.history(period=days)




    # prices = data["Close"]


    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=True)
    trades = testPolicy(days="500d",symbol=symbol, sd=sd, ed=ed, sv=start_val)


    # print(trades)
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


    learner = sl.StrategyLearner()
    learner.add_evidence(symbol=stock, sd=sd, ed=ed, sv=100000)
    learner_output = learner.testPolicy(symbol=stock, sd=sd1, ed=ed1, sv=100000)
    df2 = compute_portvals(symbol, learner_output, start_val=start_val, commission=commission, impact=impact)
    port_output = pd.DataFrame(index=prices.index.values, dtype=float)
    port_output['total'] = df2
    port_output.fillna(method="bfill", inplace=True)

    port_normalized_total = port_output / port_output.iloc[0]
    port_normalized_total = port_normalized_total['total']




    learner = sl2.StrategyLearner()
    learner.add_evidence(symbol=stock, sd=sd, ed=ed, sv=100000)
    learner_output = learner.testPolicy(symbol=stock, sd=sd, ed=ed, sv=100000)
    df3 = compute_portvals(symbol, learner_output, start_val=start_val, commission=commission, impact=impact)
    port_output2 = pd.DataFrame(index=prices.index.values, dtype=float)
    port_output2['total'] = df3
    port_output2.fillna(method="bfill", inplace=True)



    port_normalized_total2 = port_output2 / port_output2.iloc[0]
    port_normalized_total2 = port_normalized_total2['total']








    plt.plot(normalized_benchmark_total, "purple")
    plt.plot(normalized_total, "red")
    plt.plot(port_normalized_total, "green")
    plt.plot(port_normalized_total2, "blue")



    plt.legend(labels=['Benchmark', 'Manual Strategy','RT Learner', 'Q-Learner'], loc='best')




    #
    plt.ylabel("Normalized Portfolio Value")
    plt.xlabel("Date")
    plt.gcf().autofmt_xdate()

    plt.title("Learner Comparison on " + stock + " in-sample")
    plt.savefig("learner_in_sample.png")
    plt.close()
    #
    #
    #




    start_val = 100000

    prices = get_data([symbol], pd.date_range(sd2, ed2), addSPY=True)
    trades = testPolicy(symbol=symbol, sd=sd2, ed=ed2, sv=start_val)
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
    learner = sl.StrategyLearner()
    learner.add_evidence(symbol=stock, sd=sd, ed=ed, sv=100000)
    learner_output = learner.testPolicy(symbol=stock, sd=sd2, ed=ed2, sv=100000)
    df2 = compute_portvals(symbol, learner_output, start_val=start_val, commission=commission, impact=impact)
    port_output = pd.DataFrame(index=prices.index.values, dtype=float)
    port_output['total'] = df2
    port_output.fillna(method="bfill", inplace=True)

    port_normalized_total = port_output / port_output.iloc[0]
    port_normalized_total = port_normalized_total['total']


    # print('total ', port_normalized_total[-1])




    learner2 = sl2.StrategyLearner()
    learner2.add_evidence(symbol=stock, sd=sd1, ed=ed1, sv=100000)
    learner_output2 = learner2.testPolicy(symbol=stock, sd=sd2, ed=ed2, sv=100000)
    df3 = compute_portvals(symbol, learner_output2, start_val=start_val, commission=commission, impact=impact)
    port_output2 = pd.DataFrame(index=prices.index.values, dtype=float)
    port_output2['total'] = df3
    port_output2.fillna(method="bfill", inplace=True)
    # print(output)

    port_normalized_total2 = port_output2 / port_output2.iloc[0]
    port_normalized_total2 = port_normalized_total2['total']
    fig, ax = plt.subplots()

    plt.plot(normalized_benchmark_total, "purple", label="Benchmark")
    plt.plot(normalized_total, "red", label="Manual Strategy")
    plt.plot(port_normalized_total, "green", label="RT Learner")
    plt.plot(port_normalized_total2, "blue", label="Q-Learner")


    plt.legend()

    ax.text(0.5, 0.5, 'out-of-sample', transform=ax.transAxes,
            fontsize=40, color='gray', alpha=0.5,
            ha='center', va='center', rotation=30)
    plt.ylabel("Normalized Portfolio Value")
    plt.xlabel("Date")
    plt.gcf().autofmt_xdate()
    plt.title("Learner Comparison on " + stock + " out-of-sample")
    plt.savefig("learner_out_of_sample.png")
    plt.close()

if __name__ == "__main__":
    np.random.seed(903760738)
    createChartAndStatisticsForExperiment1(stock = 'META', sd1= dt.datetime(2020, 1, 1), ed1= dt.datetime(2022, 9, 30),
                                           sd2=dt.datetime(2022, 10, 1), ed2=dt.datetime(2023, 8, 5),
                                           commission=0, impact = 0.005)