import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


from marketsimcode import compute_portvals
import StrategyLearner as sl
from util import get_data


def author():
    return "gwang383"


def createChartAndStatisticsForExperiment2():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    start_val = 100000
    commission = 9.95
    impact = 0.005
    prices = get_data([symbol], pd.date_range(sd, ed), addSPY=True)




    learner = sl.StrategyLearner(verbose = False, impact = 0.001, commission = 0.0)
    learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    learner_output = learner.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df2 = compute_portvals(symbol, learner_output, start_val=start_val, commission=commission, impact=impact)
    port_output = pd.DataFrame(index=prices.index.values, dtype=float)
    port_output['total'] = df2
    port_output.fillna(method="bfill", inplace=True)


    port_normalized_total_1 = port_output / port_output.iloc[0]
    port_normalized_total_1 = port_normalized_total_1['total']
    cr_1 = round((port_normalized_total_1.iloc[-1] - port_normalized_total_1.iloc[0]) / port_normalized_total_1.iloc[0],6)


    daily_rets_1 = port_normalized_total_1[1:] / port_normalized_total_1[:-1].values - 1

    std_1 = round(daily_rets_1.std(), 6)
    mean_1 = round(daily_rets_1.mean(), 6)




    learner = sl.StrategyLearner(verbose = False, impact = 0.005, commission = 0.0)
    learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    learner_output = learner.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df2 = compute_portvals(symbol, learner_output, start_val=start_val, commission=commission, impact=impact)
    port_output = pd.DataFrame(index=prices.index.values, dtype=float)
    port_output['total'] = df2
    port_output.fillna(method="bfill", inplace=True)


    port_normalized_total_2 = port_output / port_output.iloc[0]
    port_normalized_total_2 = port_normalized_total_2['total']
    cr_2 = round((port_normalized_total_2.iloc[-1] - port_normalized_total_2.iloc[0]) / port_normalized_total_2.iloc[0],6)

    daily_rets_2 = port_normalized_total_2[1:] / port_normalized_total_2[:-1].values - 1

    std_2 = round(daily_rets_2.std(), 6)
    mean_2 = round(daily_rets_2.mean(), 6)



    learner = sl.StrategyLearner(verbose = False, impact = 0.025, commission = 0.0)
    learner.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    learner_output = learner.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    df2 = compute_portvals(symbol, learner_output, start_val=start_val, commission=commission, impact=impact)
    port_output = pd.DataFrame(index=prices.index.values, dtype=float)
    port_output['total'] = df2
    port_output.fillna(method="bfill", inplace=True)


    port_normalized_total_3 = port_output / port_output.iloc[0]
    port_normalized_total_3 = port_normalized_total_3['total']


    cr_3 = round((port_normalized_total_3.iloc[-1] - port_normalized_total_3.iloc[0]) / port_normalized_total_3.iloc[0], 6)


    daily_rets_3 = port_normalized_total_3[1:] / port_normalized_total_3[:-1].values - 1

    std_3 = round(daily_rets_3.std(), 6)
    mean_3 = round(daily_rets_3.mean(), 6)




    plt.plot(port_normalized_total_3, "purple", label="Impact = 0.025")
    plt.plot(port_normalized_total_2, "red", label="Impact = 0.005")
    plt.plot(port_normalized_total_1, "green", label="Impact = 0.001")




    plt.legend()

    plt.ylabel("Normalized Portfolio Value")
    plt.xlabel("Date")
    plt.gcf().autofmt_xdate()
    plt.title("Experiment 2: different impact values on Strategy Learner in-sample")
    plt.savefig("experiment_2.png")
    plt.close()


    statistics = pd.DataFrame({"Cumulative Return": [cr_1, cr_2, cr_3],
                           "STD of Daily Returns": [std_1, std_2, std_3]},
                          index=["impact = 0.001", "impact = 0.005", "impact = 0.025"])


    file = open("experiment_2_statistics.txt", "w")
    file.write(str(pd.DataFrame(statistics)))
    file.close()





if __name__ == "__main__":
    createChartAndStatisticsForExperiment2()