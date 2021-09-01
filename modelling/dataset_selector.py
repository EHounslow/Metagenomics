from pandas.core.frame import DataFrame


def get_dataset(dataset_to_select: str, abundance_data: DataFrame) -> DataFrame:
    """
    Creates a single data subset dataframe from the full metagenomic dataframe.
    Args:
        dataset_to_select: (str): Name of the target dataset
        abundance_data: (DataFrame): Dataframe of the full metagenomic dataset

    Returns:
        DataFrame: A dataframe containing a dataset subset
    """

    dataset = abundance_data["dataset_name"]
    disease_target = abundance_data[dataset == dataset_to_select]

    return disease_target
