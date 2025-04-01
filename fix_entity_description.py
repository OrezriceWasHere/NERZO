import dataset_provider
from clearml_pipelines.nertreieve_dataset import nertrieve_processor

all_nertrieve_types = dataset_provider.search(
	query={"query": {"match_all": {}}}, index="nertrieve_entity_name_to_embedding", size=200
)
type_to_name = nertrieve_processor.type_to_name()
bulk = []
for item in all_nertrieve_types["hits"]["hits"]:
	doc_id = item["_id"]
	type = item["_source"]["entity_name"]
	description = type_to_name[type]
	dataset_provider.es.update(index="nertrieve_entity_name_to_embedding", id=doc_id, body={"doc":{"entity_description":  description}})


