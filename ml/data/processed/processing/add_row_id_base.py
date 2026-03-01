import pandas as pd


class AddRowIDBase():
    def add_row_id(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        ...