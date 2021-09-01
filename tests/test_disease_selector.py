from modelling.dataset_selector import get_dataset
from modelling.data_loader import extract_abundance_data
from modelling.disease_selector import get_disease_data


def test_get_disease_data():
    # Arrange
    abundance_data = extract_abundance_data("data/abundance.txt")
    data_subset = get_dataset("Quin_gut_liver_cirrhosis", abundance_data)

    # Act
    combined_disease_control, disease_target, control = get_disease_data("cirrhosis", data_subset)

    # Assert
    assert "cirrhosis" not in combined_disease_control["disease"].values
    assert "nd" not in combined_disease_control["disease"].values
    assert 1 in combined_disease_control["disease"].values
    assert 0 in combined_disease_control["disease"].values
