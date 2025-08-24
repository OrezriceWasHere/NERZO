from typing import Dict


def type_to_name() -> Dict[str, str]:
    """Return a mapping from MultiCoNER fine type to a human readable name."""

    # Mapping from fine type to a human readable description. All fine types are
    # listed explicitly so callers do not rely on any coarse type hierarchy.
    mapping = {
        "OtherLOC": "Location - Other",
        "HumanSettlement": "Human Settlement",
        "Facility": "Facility",
        "Station": "Station",
        "VisualWork": "Visual Work",
        "MusicalWork": "Musical Work",
        "WrittenWork": "Written Work",
        "ArtWork": "Art Work",
        "Software": "Software",
        "MusicalGRP": "Musical Group",
        "PublicCORP": "Public Corporation",
        "PrivateCORP": "Private Corporation",
        "AerospaceManufacturer": "Aerospace Manufacturer",
        "SportsGRP": "Sports Group",
        "CarManufacturer": "Car Manufacturer",
        "ORG": "Organization",
        "Scientist": "Scientist",
        "Artist": "Artist",
        "Athlete": "Athlete",
        "Politician": "Politician",
        "Cleric": "Cleric",
        "SportsManager": "Sports Manager",
        "OtherPER": "Person - Other",
        "Clothing": "Clothing",
        "Vehicle": "Vehicle",
        "Food": "Food",
        "Drink": "Drink",
        "OtherPROD": "Product - Other",
        "Medication/Vaccine": "Medication or Vaccine",
        "MedicalProcedure": "Medical Procedure",
        "AnatomicalStructure": "Anatomical Structure",
        "Symptom": "Symptom",
        "Disease": "Disease",
    }

    return mapping

