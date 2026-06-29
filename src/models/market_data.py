from __future__ import annotations

import pandas as pd


class MarketData:
    """
    Wrapper around pandas DataFrame.

    Every indicator in SmartMoneyEngine
    will work with this class instead
    of directly modifying DataFrames.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe

    # -------------------------
    # Basic Properties
    # -------------------------

    @property
    def rows(self) -> int:
        return len(self.data)

    @property
    def columns(self):
        return list(self.data.columns)

    @property
    def shape(self):
        return self.data.shape

    # -------------------------
    # Column Operations
    # -------------------------

    def has_column(self, column: str) -> bool:
        return column in self.data.columns

    def get_column(self, column: str):
        return self.data[column]

    def add_column(self, column: str, values):
        self.data[column] = values

    def remove_column(self, column: str):
        if self.has_column(column):
            self.data.drop(columns=[column], inplace=True)

    def rename_column(self, old: str, new: str):
        self.data.rename(columns={old: new}, inplace=True)

    # -------------------------
    # Copy
    # -------------------------

    def copy(self):
        return MarketData(self.data.copy())

    # -------------------------
    # Display
    # -------------------------

    def head(self, rows: int = 5):
        return self.data.head(rows)

    def tail(self, rows: int = 5):
        return self.data.tail(rows)

    # -------------------------
    # Utility
    # -------------------------

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)