# this is moddified from my own code project 6: indicators.py



def createStochasticOscillator(df):
    high = df.rolling(14).max()
    low = df.rolling(14).min()
    k = (df - low) / (high - low) * 100
    res = k.rename(columns={k.columns[0]: "K"}).dropna()
    d = k.rolling(3).mean().dropna()
    res["D"] = d
    res.dropna(inplace=True)
    return res


def createBollinger(df):
    mean = df.rolling(window=20).mean()
    res = mean.rename(columns={mean.columns[0]: "MEAN"})
    std = df.rolling(window=20).std()
    l, u = mean - 2 * std, mean + 2 * std
    percent = (df - l) / (u - l)
    res["UPPER"] = u
    res["LOWER"] = l
    res["PERCENT"] = percent
    return res


def createRSI(df):
    diff = df.diff()
    positive = diff.clip(lower=0)
    negative = diff.clip(upper=0) * -1
    gain = positive.rolling(window=14).mean()
    loss = negative.rolling(window=14).mean()

    rs = gain / loss
    res = 100 - (100 / (1 + rs))
    res = res.rename(columns={res.columns[0]: "RSI"})
    res.dropna(inplace=True)
    return res

def createGoldenDeathCross(df):
    res = df.copy()
    res["SMA20"] = df.rolling(window=20).mean()
    res["SMA50"] = df.rolling(window=50).mean()

    return res
def createMACD(df):
    short = df.ewm(span=12).mean()
    long = df.ewm(span=26).mean()
    res = short - long
    res.rename(columns={res.columns[0]: "MACD"}, inplace=True)
    signal = res.ewm(span=9).mean()
    res["SIGNAL"] = signal

    # k = res["MACD"]-res["SIGNAL"]
    # res["diff"] = k
    # res["c_m"] = res["MACD"].diff()
    # res["c_m"] = res["c_m"].fillna(0)
    #
    #
    #
    # res["buy"] = np.where((res["c_m"] > 0) & (res['MACD'] > res['SIGNAL']), 1, 0)
    # res["sell"] = np.where((res["c_m"] < 0) & (res['MACD'] < res['SIGNAL']), 1, 0)
    # kd_diff = (k.shift(1) / k) / abs(k.shift(1) / k)
    # res['kd_diff'] = kd_diff
    # res["ori"] = np.where(kd_diff== -1, 1, 0)
    #
    # print(res)



    return res




