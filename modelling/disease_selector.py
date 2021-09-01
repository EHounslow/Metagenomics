from typing import Tuple
from pandas.core.frame import DataFrame


def get_disease_data(
    disease_to_select: str, disease_target: DataFrame
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """
    Creates a single disease subset dataframe from the full metagenomic dataframe.
    Args:
        disease_to_select: (str): Name of the target dataset
        disease_target: (DataFrame): Dataframe of the targeted disease subset

    Returns:
        Tuple [DataFrame, DataFrame, DataFrame]: 3 dataframes returning subsets
        of the dataset containing control samples, disease samples, and both,
        with a boolean (int) value in the disease column.
    """

    disease = disease_target["disease"]

    # Create subset of controls and target diseases
    combined_disease_control = disease_target[
        (disease == "n") | (disease == "nd") | (disease == disease_to_select)
    ]
    control = disease_target[(disease == "n") | (disease == "nd")]
    disease_target = disease_target[disease == disease_to_select]

    # transform target variable disease into numeric data
    combined_disease_control["disease"] = combined_disease_control["disease"].replace(
        ["n", "nd"], 0
    )
    combined_disease_control["disease"] = combined_disease_control["disease"].replace(
        disease_to_select, 1
    )
    return combined_disease_control, disease_target, control
