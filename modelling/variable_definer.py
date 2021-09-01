from pandas.core.frame import DataFrame


def get_variables(data_subset: DataFrame):
    """
    Seperates the variables into 3 lists containing metadata, species,
    and all other taxonomic variables.
    Args:
        data_subset: (DataFrame): A dataframe containing the target data.

    Returns:
        species: (list): A list of all taxonomic variables at the species level only
        taxonomy: (list): A list of all other taxonomic variables
        metadata: (list): A list of all non-taxonomic variables
    """
    variables = data_subset.columns.values
    species = []
    taxonomy = []
    metadata = []

    for cell in variables:
        if cell.startswith("k_") and "s_" in cell and "t_" not in cell:
            species.append(cell)
        elif cell.startswith("k_") and ("s_" not in cell or "t_" in cell):
            taxonomy.append(cell)
        elif not cell.startswith("k_"):
            metadata.append(cell)

    return species, taxonomy, metadata
