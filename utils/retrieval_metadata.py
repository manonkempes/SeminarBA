from pathlib import Path
from typing import Tuple

import pandas as pd


DEFAULT_METADATA_COLUMNS = [
    'external_code',
    'release_date',
    'split',
    'category',
    'color',
    'fabric',
    'season',
    'image_path',
]


def load_split_dataframes(data_folder: str, val_fraction: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_folder)
    train_df = pd.read_csv(data_path / 'train.csv', parse_dates=['release_date'])
    train_df = train_df.sort_values('release_date').reset_index(drop=True)
    val_size = max(1, int(val_fraction * len(train_df)))

    subtrain_df = train_df.iloc[:-val_size].copy()
    subtrain_df['split'] = 'subtrain'

    val_df = train_df.iloc[-val_size:].copy()
    val_df['split'] = 'val'

    test_df = pd.read_csv(data_path / 'test.csv', parse_dates=['release_date']).copy()
    test_df['split'] = 'test'
    return subtrain_df, val_df, test_df


def combine_splits(data_folder: str, val_fraction: float = 0.15) -> pd.DataFrame:
    subtrain_df, val_df, test_df = load_split_dataframes(data_folder=data_folder, val_fraction=val_fraction)
    combined = pd.concat([subtrain_df, val_df, test_df], axis=0, ignore_index=True)
    combined['store_idx'] = range(len(combined))
    return combined


def sales_column_names(data_df: pd.DataFrame, horizon: int = 12):
    return list(data_df.columns[:horizon])


def build_metadata_table(data_folder: str, val_fraction: float = 0.15, extra_columns=None) -> pd.DataFrame:
    combined = combine_splits(data_folder=data_folder, val_fraction=val_fraction)
    columns = list(DEFAULT_METADATA_COLUMNS)
    if extra_columns:
        columns.extend(extra_columns)
    columns = [col for col in columns if col in combined.columns]
    metadata = combined[columns].copy()
    metadata['store_idx'] = combined['store_idx'].values
    return metadata.reset_index(drop=True)
