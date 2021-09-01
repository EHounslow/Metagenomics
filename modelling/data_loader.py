import pandas as pd
from pandas.core.frame import DataFrame


def extract_abundance_data(abundance) -> DataFrame:
    """
    Creates a transposed dataframe from a .txt or .csv file and sets the first row as the header.
    Args:
        abundance: (str): The path to the source file

    Returns:
        DataFrame: A dataframe of the file data
    """

    abundance_data = pd.read_csv(abundance, delimiter="\t", header=None, dtype=str).T
    new_header = abundance_data.iloc[0]

    abundance_data = abundance_data[1:]
    abundance_data.columns = new_header

    return abundance_data
