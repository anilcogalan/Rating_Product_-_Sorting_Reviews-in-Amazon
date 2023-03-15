

import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df = pd.read_csv("./Desktop/amazon_review.csv")
df.head()

df["overall"].mean()

# determination of time-based average weights
def time_based_weighted_average(dataframe, w1=50, w2=25, w3=15, w4=10):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

# There is no helpful_no variable in the data set, it must be generated over existing variables.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score

    - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
    - The score to be calculated is used for product ranking.
    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.head()

df.sort_values("wilson_lower_bound", ascending=False).head(20)