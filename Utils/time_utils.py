import pandas as pd

def get_date_offset(evaluation_period: str):
        evaluation_period = evaluation_period.lower()
        if evaluation_period == "daily":
            return pd.Timedelta(days=1)
        elif evaluation_period == "weekly":
            return pd.Timedelta(weeks=1)
        elif evaluation_period == "monthly":
            return pd.DateOffset(months=1)
        elif evaluation_period == "yearly":
            return pd.DateOffset(years=1)
        else:
            raise ValueError(f"Unsupported evaluation period: {evaluation_period}")