def build_classification_prompt(entity_type: str, entity_text: str, sentence: str) -> str:
    entity_type = entity_type.replace("_", " ")
    return (f"In the following sentence, specify if the word or word phrase '{entity_text.replace('`', '').strip()}' "
            f"represent a {entity_type}: {sentence} Please answer 'yes' or 'no'.")


def build_extraction_prompt(entity_type: str, sentence: str) -> str:
    entity_type = entity_type.replace("_", " ")
    return f"In the following sentence, write in a list the words or word phrases that represent a {entity_type}: {sentence}\n if there are no words or word phrases that represent a {entity_type} please write 'none'."


def pick_entities_of_type(document, entity_type: str) -> list[dict]:
    return [entity for entity in document["tagged_entities"] if entity["type"] == entity_type]

def pick_entities_of_finetype_fewnerd(document, fine_type: str) -> list[dict]:
    return [entity for entity in document if entity["fine_type"] == fine_type]



