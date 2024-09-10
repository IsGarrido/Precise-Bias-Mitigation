import numpy as np
import pandas as pd
import argparse
from relhelperspy.io.rel_project_helper import RelProjectHelper


def calculate_statistic(df, columns, statistic, statistic_value=None):
    if statistic == 'sum':
        return df[columns].sum()
    elif statistic == 'mean':
        return df[columns].mean()
    elif statistic == 'median':
        return df[columns].median()
    elif statistic == 'min':
        return df[columns].min()
    elif statistic == 'max':
        return df[columns].max()
    elif statistic == 'std':
        return df[columns].std()
    elif statistic == 'var':
        return df[columns].var()
    elif statistic == 'quantile':
        if statistic_value is None or not 0 <= statistic_value <= 1:
            raise ValueError("Please specify a quantile value between 0 and 1.")
        return df[columns].quantile(statistic_value)
    elif statistic == 'iqr':
        return df[columns].quantile(0.75) - df[columns].quantile(0.25)
    elif statistic == 'skewness':
        return df[columns].skew()
    elif statistic == 'kurtosis':
        return df[columns].kurt()
    elif statistic == 'count_values_under':
        if statistic_value is None:
            raise ValueError("Please specify a value for 'statistic_value' when using 'count_values_under'.")
        return (df[columns] < statistic_value).sum()
    elif statistic == 'count_values_over':
        if statistic_value is None:
            raise ValueError("Please specify a value for 'statistic_value' when using 'count_values_over'.")
        return (df[columns] > statistic_value).sum()
    elif statistic == 'sum_under':
        if statistic_value is None:
            raise ValueError("Please specify a value for 'statistic_value' when using 'sum_under'.")
        return df[columns][df[columns] < statistic_value].sum()
    elif statistic == 'sum_over':
        if statistic_value is None:
            raise ValueError("Please specify a value for 'statistic_value' when using 'sum_over'.")
        return df[columns][df[columns] > statistic_value].sum()
    else:
        raise ValueError("Invalid statistic type. Please check your inputs.")

class ExtractRelevantLayers:
    
    def __init__(self, experiment: str, folder: str) -> None:
        _project = RelProjectHelper(experiment)
        data_path = _project.get_path(experiment, folder, "evaluation.feather")
        self.data = _project.load_result(data_path)

    def run(self):
        df = self.data
        prob_columns = [col for col in df.columns if '_prob' in col]
        top_k_columns = [col for col in df.columns if 'top_k_index' in col]

        original_column = f'original_top_k_index'

        diffs = df[top_k_columns].subtract(df[original_column], axis=0)
        
        # Keep only those column with all values above 0. All other columns are irrelevant
        relevant_columns = diffs.columns[(diffs >= 0).all()]
        
        x = 0
        



parser = argparse.ArgumentParser(description=".")
parser.add_argument("--experiment", type=str, default="single-layer__gpt2", help="Experiment name")
parser.add_argument("--train_folder", type=str, default="female", help="Folder containing the training data")
args = parser.parse_args()

ExtractRelevantLayers(args.experiment, args.train_folder).run()
