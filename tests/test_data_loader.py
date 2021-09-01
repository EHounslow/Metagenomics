from pandas.core.frame import DataFrame
from modelling.data_loader import extract_abundance_data


def test_extract_abundance_data_success():
    # Arrange
    abundance = "data/abundance.txt"

    # Act
    abundance_data = extract_abundance_data(abundance)

    # Assert
    assert isinstance(abundance_data, DataFrame)
    assert "disease" in abundance_data.columns
