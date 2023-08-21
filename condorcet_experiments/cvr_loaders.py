import pandas as pd
import pathlib
import os
from typing import Optional
from .profile import PreferenceProfile
from .ballot import Ballot
from pandas.errors import EmptyDataError, DataError
from fractions import Fraction


def rank_column_csv(
    fpath: str,
    *,
    weight_col: Optional[int] = None,
    delimiter: Optional[str] = None,
    id_col: Optional[int] = None,
) -> PreferenceProfile:
    """
    given a file path, loads cvr with ranks as columns and voters as rows
    (empty cells are treated as None)
    (if voter ids are missing, we're currently not assigning ids)
    Args:
        fpath (str): path to cvr file
        id_col (int, optional): index for the column with voter ids
    Raises:
        FileNotFoundError: if fpath is invalid
        EmptyDataError: if dataset is empty
        ValueError: if the voter id column has missing values
        DataError: if the voter id column has duplicate values
    Returns:
        PreferenceProfile: a preference schedule that
        represents all the ballots in the elction
    """
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"File with path {fpath} cannot be found")

    cvr_path = pathlib.Path(fpath)
    df = pd.read_csv(
        cvr_path,
        on_bad_lines="error",
        encoding="utf8",
        index_col=False,
        delimiter=delimiter,
    )

    if df.empty:
        raise EmptyDataError("Dataset cannot be empty")
    if id_col is not None and df.iloc[:, id_col].isnull().values.any():  # type: ignore
        raise ValueError(f"Missing value(s) in column at index {id_col}")
    if id_col is not None and not df.iloc[:, id_col].is_unique:
        raise DataError(f"Duplicate value(s) in column at index {id_col}")

    ranks = list(df.columns)
    if id_col is not None:
        ranks.remove(df.columns[id_col])
    grouped = df.groupby(ranks, dropna=False)
    ballots = []

    for group, group_df in grouped:
        ranking = [{None} if pd.isnull(c) else {c} for c in group]
        voters = None
        if id_col is not None:
            voters = set(group_df.iloc[:, id_col])
        weight = len(group_df)
        if weight_col is not None:
            weight = sum(group_df.iloc[:, weight_col])
        b = Ballot(ranking=ranking, weight=Fraction(weight), voters=voters)
        ballots.append(b)

    return PreferenceProfile(ballots=ballots)
