""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	

Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	

Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	

We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	

-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	

Student Name: Guochiuan Wang (replace with your name)  		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: gwang383 (replace with your User ID)  		  	   		  		 			  		 			 	 	 		 		 	
GT ID: 903760738 (replace with your GT ID)  		  	   		  		 			  		 			 	 	 		 		 	
"""

import datetime as dt
import numpy as np
import pandas as pd  		  	   		  		 			  		 			 	 	 		 		 	
import util as ut
import indicators as ind
import BagLearner as bl
from marketsimcode import compute_portvals
  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
class StrategyLearner(object):

    def author(self):
        return "gwang383"

    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        self.verbose = verbose  		  	   		  		 			  		 			 	 	 		 		 	
        self.impact = impact  		  	   		  		 			  		 			 	 	 		 		 	
        self.commission = commission
        self.learner = bl.BagLearner(kwargs={"leaf_size": 5}, bags=50, boost=False, verbose=False)
		  	   		  		 			  		 			 	 	 		 		 	
    def add_evidence(  		  	   		  		 			  		 			 	 	 		 		 	
        self,  		  	   		  		 			  		 			 	 	 		 		 	
        symbol="IBM",  		  	   		  		 			  		 			 	 	 		 		 	
        sd=dt.datetime(2008, 1, 1),  		  	   		  		 			  		 			 	 	 		 		 	
        ed=dt.datetime(2009, 1, 1),  		  	   		  		 			  		 			 	 	 		 		 	
        sv=10000,  		  	   		  		 			  		 			 	 	 		 		 	
    ):
        prices = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=False)
        prices.fillna(method="ffill", inplace=True)
        prices.fillna(method="bfill", inplace=True)
        # prices = prices.drop(columns=["SPY"])
        data = pd.DataFrame(index=prices.index.values, dtype=float)
        data['price'] = prices

        rsi = ind.createRSI(prices)
        bb = ind.createBollinger(prices)
        so = ind.createStochasticOscillator(prices)
        data['bbp'] = bb['PERCENT']
        data['rsi'] = rsi['RSI']
        data['so'] = so['D']
        rate = data['price'].diff(7) / data['price']
        rate = rate.shift(-7)
        data['rate'] = rate

        data_without_nan = data.copy()
        data_without_nan = data_without_nan[~data_without_nan.isnull().any(axis=1)]
        sign = np.zeros(data_without_nan.shape[0])
        for i in range(data_without_nan.shape[0]):
            if data_without_nan.iloc[i,-1] < -0.02 - self.impact:
                sign[i] = -1
            elif data_without_nan.iloc[i,-1] > 0.02 + self.impact:
                sign[i] = 1


        bbp_mean = data_without_nan['bbp'].mean()
        bbp_std = data_without_nan['bbp'].std()
        rsi_mean = data_without_nan['rsi'].mean()
        rsi_std = data_without_nan['rsi'].std()
        so_mean = data_without_nan['so'].mean()
        so_std = data_without_nan['so'].std()

        data_without_nan['bbp'] = (data_without_nan['bbp'] - bbp_mean) / bbp_std
        data_without_nan['rsi'] = (data_without_nan['rsi'] - rsi_mean) / rsi_std
        data_without_nan['so'] = (data_without_nan['so'] - so_mean) / so_std

        data_without_nan.drop(columns=['price', 'rate'], inplace=True)
        features = data_without_nan.copy()
        self.learner.add_evidence(features, sign)





    def testPolicy(
        self,  		  	   		  		 			  		 			 	 	 		 		 	
        symbol="IBM",  		  	   		  		 			  		 			 	 	 		 		 	
        sd=dt.datetime(2009, 1, 1),  		  	   		  		 			  		 			 	 	 		 		 	
        ed=dt.datetime(2010, 1, 1),  		  	   		  		 			  		 			 	 	 		 		 	
        sv=10000,  		  	   		  		 			  		 			 	 	 		 		 	
    ):
        prices = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=False)
        prices.fillna(method="ffill", inplace=True)
        prices.fillna(method="bfill", inplace=True)
        # prices = prices.drop(columns=["SPY"])
        data = pd.DataFrame(index=prices.index.values, dtype=float)
        data['price'] = prices

        rsi = ind.createRSI(prices)
        bb = ind.createBollinger(prices)
        so = ind.createStochasticOscillator(prices)
        macd = ind.createMACD(prices)
        gdc = ind.createGoldenDeathCross(prices)

        macd['macd'] = macd['MACD']
        macd['macd_diff'] = macd['macd'].diff(1)
        macd['macd_signal'] = macd['SIGNAL']
        gdc['sma20'] = gdc['SMA20']
        gdc['sma20_diff'] = gdc['sma20'].diff(1)
        gdc['sma50'] = gdc['SMA50']


        data['signal_1'] = 0
        for i in range(0, data.shape[0]):
            if macd.iloc[i, -2] > 0 and macd.iloc[i, -3] > macd.iloc[i, -1]:
                data.iloc[i, -1] = 1
            elif macd.iloc[i, -2] < 0 and macd.iloc[i, -3] < macd.iloc[i, -1]:
                data.iloc[i, -1] = -1

        data['signal_2'] = 0
        for i in range(0, data.shape[0]):
            if gdc.iloc[i, -2] > 0 and gdc.iloc[i, -3] > gdc.iloc[i, -1]:
                data.iloc[i, -1] = 1
            elif gdc.iloc[i, -2] < 0 and gdc.iloc[i, -3] < gdc.iloc[i, -1]:
                data.iloc[i, -1] = -1






        data['bbp'] = bb['PERCENT']
        data['rsi'] = rsi['RSI']
        data['so'] = so['D']
        #
        data_without_nan = data.copy()
        data_without_nan = data_without_nan[~data_without_nan.isnull().any(axis=1)]
        #
        #
        #
        # bbp_mean = data_without_nan['bbp'].mean()
        # bbp_std = data_without_nan['bbp'].std()
        # rsi_mean = data_without_nan['rsi'].mean()
        # rsi_std = data_without_nan['rsi'].std()
        # so_mean = data_without_nan['so'].mean()
        # so_std = data_without_nan['so'].std()
        # signal_1_mean = data_without_nan['signal_1'].mean()
        # signal_1_std = data_without_nan['signal_1'].std()
        # signal_2_mean = data_without_nan['signal_2'].mean()
        # signal_2_std = data_without_nan['signal_2'].std()
        #
        #
        #
        # data_without_nan['bbp'] = (data_without_nan['bbp'] - bbp_mean) / bbp_std
        # data_without_nan['rsi'] = (data_without_nan['rsi'] - rsi_mean) / rsi_std
        # data_without_nan['so'] = (data_without_nan['so'] - so_mean) / so_std
        # data_without_nan['signal_1'] = (data_without_nan['signal_1'] - signal_1_mean) / signal_1_std
        # data_without_nan['signal_2'] = (data_without_nan['signal_2'] - signal_2_mean) / signal_2_std

        data_without_nan.drop(columns=['price'], inplace=True)
        feature_data = data_without_nan.copy()


        # print(feature_data.values)

        sign = self.learner.query(feature_data.values)

        # print("herer")
        # print(sign)

        holding = 0


        trades = pd.DataFrame(index=data_without_nan.index.values, columns=['trade'], dtype=float)


        for i in range(0, trades.shape[0]):

            trades.iloc[i, 0] = 0
            if holding == 1000:
                if sign[i] == 1:
                    trades.iloc[i, 0] = 0
                    holding = 1000

                elif sign[i] == -1:
                    trades.iloc[i, 0] = -2000
                    holding = -1000

            elif holding == -1000:
                if sign[i] == 1:
                    trades.iloc[i, 0] = 2000
                    holding = 1000

                elif sign[i] == -1:
                    trades.iloc[i, 0] = 0
                    holding = -1000

            elif holding == 0:
                if sign[i] == 1:
                    trades.iloc[i, 0] = 1000
                    holding = 1000

                elif sign[i] == -1:
                    trades.iloc[i, 0] = -1000
                    holding = -1000

        df = compute_portvals(symbol, trades, start_val=sv, commission=9.95, impact=0.005)
        output = pd.DataFrame(index=prices.index.values, dtype=float)
        output['total'] = df
        output.fillna(method="bfill", inplace=True)
 

        return trades

