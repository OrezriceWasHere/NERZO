
def build_prompt(entity_type: str, entity_text: str, sentence: str) -> str:
    entity_type = entity_type.replace("_", " ")
    return f"In the following sentence, predict if the word or word phrase '{entity_text}' represent a {entity_type}: {sentence} Please answer 'yes' or 'no'."

def pick_entities_of_type(document, entity_type:str) -> list[dict]:
    return [entity for entity in document["tagged_entities"] if entity["type"] == entity_type]
