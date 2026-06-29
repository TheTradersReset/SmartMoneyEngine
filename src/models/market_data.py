from dataclasses import dataclass

import pandas as pd


@dataclass
class MarketData:
    """
    Represents validated market data.

    Every indicator will receive this object.
    """

    dataframe: pd.DataFrame

    @property
    def open(self):

        return self.dataframe["Open"]

    @property
    def high(self):

        return self.dataframe["High"]

    @property
    def low(self):

        return self.dataframe["Low"]

    @property
    def close(self):

        return self.dataframe["Close"]

    @property
    def volume(self):

        return self.dataframe["Volume"]

    @property
    def date(self):

        return self.dataframe["Date"]

    def rows(self):

        return len(self.dataframe)

    def columns(self):

        return self.dataframe.columns.tolist()