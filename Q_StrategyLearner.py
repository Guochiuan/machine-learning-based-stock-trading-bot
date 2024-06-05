
import datetime as dt
import random
import numpy as np
import pandas as pd
import util as ut
import indicators as ind
import BagLearner as bl
from marketsimcode import compute_portvals
import QLearner as qlt

class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """

    bin1 = []
    bin2 = []
    bin3 = []
    def create_bins(self, var1, var2, var3):
        bins = 10
        step_size = int(len(var1)/bins)

        # sort data
        s1 = np.sort(var1)
        s2 = np.sort(var2)
        s3 = np.sort(var3)

        # stack
        v1 = np.vstack((s1, s2))
        v2 = np.vstack((v1, s3))

        # Create bins
        for i in range(0, len(var1), step_size):
            if i > 0:
                # Lower and upper bound
                self.bin1.append([s1[i - step_size], s1[i]])
                self.bin2.append([s2[i - step_size], s2[i]])
                self.bin3.append([s3[i - step_size], s3[i]])
        pass

    def get_state_from_bins(self, var1, var2, var3):
        ind1 = 0
        ind2 = 0
        ind3 = 0
        for i in range(len(self.bin1), 0, -1):
            if var1 < self.bin1[i-1][1]:
                ind1 = i - 1
            if var2 < self.bin2[i - 1][1]:
                ind2 = i - 1
            if var3 < self.bin3[i - 1][1]:
                ind3 = i - 1
        state = int(str(ind1) + str(ind2) + str(ind3))
        return state

    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        # self.learner = bl.BagLearner(kwargs={"leaf_size": 5}, bags=15, boost=False, verbose=False)
        self.num_bins = 10

    # this method should create a QLearner, and train it for trading
    def add_evidence(
            self,
            symbol="IBM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=10000,
    ):

        # data = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=False,
        #                    colname="Adj Close").drop(columns=["SPY"])  # column 0

        data = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=False,
                           colname="Adj Close")  # column 0



        rsi = ind.createRSI(data)
        bb = ind.createBollinger(data)
        so = ind.createStochasticOscillator(data)
        data['bbp'] = bb['PERCENT']
        data['rsi'] = rsi['RSI']
        data['so'] = so['D']
        # RSI Strategy
        # data["rsi"] = ind.rsi_calc(data[symbol], time_horizon=14, trade_signal=False)  # column 1
        #
        # # MACD Strategy
        # data["MACD"] = ind.MACD(data[symbol], short_term=12, long_term=26, trade_signal=False) # column 2
        #
        # # Golden Cross
        # data["Golden"] = ind.golden_death_cross(data[symbol], trade_signal=False)  #column 3

        # Setup columns for iteration:
        data["trade_signal"] = 0  # column 4
        data["state"] = 0  # column 5

        # ------- Calculate returns for reward function ------- #
        data["lvl_returns"] = data[symbol] - data[symbol].shift(1)  # column 6d, aily rets, for terminal condition
        data["pct_returns"] = data[symbol].pct_change()  # column 7, daily percentage returns

        # Drop data not in the specified date range
        # data = data.iloc[50:, :].to_numpy()

        temp = data.copy()
        data = temp[~temp.isnull().any(axis=1)]

        # print(data)
        data = data.to_numpy()
        # ------- Discretize indicators ------- #
        self.create_bins(data[:, 1], data[:, 2], data[:, 3])
        for i in range(len(data)):
            data[i, 5] = int(self.get_state_from_bins(data[i, 1], data[i, 2], data[i, 3]))

        # ------- Train QLearner ------- #
        # Initialize Learner Class
        self.learner = qlt.QLearner(
            num_states=1000,
            num_actions=3,
            alpha=0.15,  # learning rate, 0.2
            gamma=0.99,  # discount rate
            rar=0.9,  # random action rate, probs of selecting action
            radr=0.99,  # random action decay rate rar = rar * radr
            dyna=200,
            verbose=False
        )

        epochs = 1000
        counter = 0
        scores = np.zeros((epochs, 1))

        while counter < epochs:
            # Iterate over number of epochs and keep score
            total_reward = 0
            state = int(data[0, 5])  # first state of time series
            action = self.learner.querysetstate(state)  # set the state and get first action

            data[0, 4] = action  # initialize action at zero

            # Learner starts at sd and ends at ed. Iterates over rows or days
            for i in range(1, len(data) - 1):
                # -- Reward function -- #
                # Argmax prefers index 0 if information or reward is unknown, hold if unknown
                if action == 0:
                    # hold
                    factor = 0
                elif action == 1:
                    # long
                    factor = 1
                elif action == 2:
                    # short
                    factor = -1

                reward = data[i, 7] * factor - self.impact  # pct returns, impact assumed as % of prior "price * shares"
                total_reward += data[i, 6] * factor

                # Record action history for reward function
                data[i, 4] = factor

                # Update for new state based on time series
                new_state = int(data[i, 5])

                # Provide Q table with new_state and reward for prior action
                # Update action for iteration
                action = self.learner.query(int(new_state), reward)

            # Update scores for epoch
            scores[counter, 0] = total_reward

            # Terminal Conditions
            # end epochs
            if counter > 1:
                if scores[counter, 0] - scores[counter - 1, 0] < 0:
                    break

            # Increment
            counter += 1













        # prices = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=True)
        # prices.fillna(method="ffill", inplace=True)
        # prices.fillna(method="bfill", inplace=True)
        # prices = prices.drop(columns=["SPY"])
        # # print(prices)
        # data = pd.DataFrame(index=prices.index.values, dtype=float)
        # data['price'] = prices
        #
        # rsi = ind.createRSI(prices)
        # bb = ind.createBollinger(prices)
        # so = ind.createStochasticOscillator(prices)
        # data['bbp'] = bb['PERCENT']
        # data['rsi'] = rsi['RSI']
        # data['so'] = so['D']
        # diff_period = 7
        # rate = data['price'].diff(diff_period )/ data['price']
        # rate = rate.shift(-diff_period)
        # data['rate'] = rate
        # # data['y'] = np.zeros(data.shape[0])
        #
        # temp = data.copy()
        # temp = temp[~temp.isnull().any(axis=1)]
        #
        # sign = np.zeros(temp.shape[0])
        # for i in range(temp.shape[0]):
        #     if temp.iloc[i ,-1] < -0.02:
        #         sign[i] = -1
        #     elif temp.iloc[i ,-1] > 0.02:
        #         sign[i] = 1
        # # temp['y'] = sign
        #
        # # print(sign)
        #
        # bbp_mean = temp['bbp'].mean()
        # bbp_std = temp['bbp'].std()
        # rsi_mean = temp['rsi'].mean()
        # rsi_std = temp['rsi'].std()
        # so_mean = temp['so'].mean()
        # so_std = temp['so'].std()
        #
        # temp['bbp'] = (temp['bbp'] - bbp_mean) / bbp_std
        # temp['rsi'] = (temp['rsi'] - rsi_mean) / rsi_std
        # temp['so'] = (temp['so'] - so_mean) / so_std
        #
        # temp.drop(columns=['price', 'rate'], inplace=True)
        # features = temp.copy()
        # # print(features.values)
        # self.learner.add_evidence(features, sign)
        #




    def testPolicy(
            self,
            symbol="IBM",
            sd=dt.datetime(2009, 1, 1),
            ed=dt.datetime(2010, 1, 1),
            sv=10000,
    ):
        # test_data = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=True,
        #                    colname="Adj Close").drop(columns=["SPY"])

        test_data = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=False,
                           colname="Adj Close")


        rsi = ind.createRSI(test_data)
        bb = ind.createBollinger(test_data)
        so = ind.createStochasticOscillator(test_data)
        test_data['bbp'] = bb['PERCENT']
        test_data['rsi'] = rsi['RSI']
        test_data['so'] = so['D']



        test_data["signal"] = 0
        test_data["state"] = 0
        # Drop data with nans/zeros - limited by 50 day moving average
        # convert to numpy for speed
        # df = test_data.iloc[50:, :]
        # test_data = df.to_numpy()

        temp = test_data.copy()
        test_data2 = temp[~temp.isnull().any(axis=1)]
        test_data = test_data2.to_numpy()


        # ------- Create Discretized Spaces and Query Q table ------- #
        for i in range(len(test_data)):
            # Code state column
            # Get current state
            test_data[i, 5] = self.get_state_from_bins(test_data[i, 1], test_data[i, 2], test_data[i, 3])

            action = self.learner.test_query(int(test_data[i, 5]))

            if action == 0:
                # hold
                factor = 0
            elif action == 1:
                # Long
                factor = 1
            elif action == 2:
                # short
                factor = -1

            test_data[i, 4] = factor

        sign = pd.DataFrame(data=test_data[:, 4], index=test_data2.index.values)

        # print(trade_signals)



        holding = 0

        trades = pd.DataFrame(index=sign.index.values, columns=['trade'], dtype=float)
        sign = sign.to_numpy()
        count = 0

        for i in range(0, trades.shape[0]):


            trades.iloc[i, 0] = 0
            if holding == 1000:
                if sign[i] == 1:
                    trades.iloc[i, 0] = 0
                    holding = 1000
                    count += 1
                elif sign[i] == -1:
                    trades.iloc[i, 0] = -2000
                    holding = -1000
                    count += 1

            elif holding == -1000:
                if sign[i] == 1:
                    trades.iloc[i, 0] = 2000
                    holding = 1000
                    count += 1

                elif sign[i] == -1:
                    trades.iloc[i, 0] = 0
                    holding = -1000
                    count += 1

            elif holding == 0:
                if sign[i] == 1:
                    trades.iloc[i, 0] = 1000
                    holding = 1000
                    count += 1

                elif sign[i] == -1:
                    trades.iloc[i, 0] = -1000
                    holding = -1000
                    count += 1
        return trades

        # df = compute_portvals(symbol, trades, start_val=sv, commission=0, impact=0.0)
        # output = pd.DataFrame(index=df.index.values, dtype=float)
        # output['total'] = df
        # output.fillna(method="bfill", inplace=True)
        # print(output)
        # return output['total'].iloc[-1]
        # print(output)




if __name__ == "__main__":
    ex = StrategyLearner()
    ex.add_evidence(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    cr = 0
    # for i in range(0, 50):
    # output = ex.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)
    # print(output)
    output = ex.testPolicy(symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000)

    # cr += output
    # print(output)
    # print("cr = " + str(cr/50))