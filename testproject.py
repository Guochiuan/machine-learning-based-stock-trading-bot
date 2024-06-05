from ManualStrategy import createChartAndStatistics
from experiment1 import createChartAndStatisticsForExperiment1
from experiment2 import createChartAndStatisticsForExperiment2
import numpy as np

def author():
    return "gwang383"


if __name__ == "__main__":
    np.random.seed(134534535)
    createChartAndStatistics()
    # createChartAndStatisticsForExperiment1()
    # createChartAndStatisticsForExperiment2()


