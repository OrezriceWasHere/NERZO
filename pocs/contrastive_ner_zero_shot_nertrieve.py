import asyncio
import json
import math
from typing import AsyncIterator, Tuple, List, Any, Dict
import torch
import random

import clearml_poc
import dataset_provider
import queries
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from contrastive import fewnerd_processor
from contrastive.args import FineTuneLLM, Arguments
from contrastive.fewnerd_processor import choose_llm_representation
from llm_interface import LLMInterface
from pocs.generic_contrastive_ner_zero_shot import AbstractDataProvider, AbstractMLPTrainer

ELASTIC_INDEX = "nertrieve_train"


def train_types():
	return ["Specie", "Eukaryote", "Film", "Company", "Actor", "City", "Musical work", "Building", "Written work",
	        "North American writer", "Scientist", "Composer", "Sportsman", "Body of water", "Educational institution",
	        "Songwriter", "Album", "Biomolecule", "British writer", "European musician",
	        "Disease or disorder", "Comics character", "City in the Americas", "Sports event", "Royalty",
	        "Protected area",
	        "Plant", "Company of Europe", "Religious leader", "Software", "Food", "University or college", "Vehicle",
	        "Stream",
	        "Video game", "Television show", "Painter", "Poet", "Drama film", "Jazz musician", "Invertebrate",
	        "Film director",
	        "Museum", "Street or road", "Political party", "Gene", "Drug", "Guitarist", "Lake", "Arthropod", "Designer",
	        "European film", "Soccer player", "Women writer", "Deity", "Soccer club", "Weapon", "Port or harbour",
	        "Military conflict", "War", "Eukaryote genetic", "Mammal genetic", "Ship", "Speculative fiction film",
	        "Insect",
	        "Diplomat", "Male voice actor", "Park", "Rock song", "Fiction book", "Character in written science fiction",
	        "Knight", "Broadcasting station or network", "Protein", "Male tennis player",
	        "Planet", "King", "Sportswoman", "Heavy metal musical group",
	        "Carbon compound", "Asian ruler", "Emperor", "Cricketer", "Animated film", "Political party in Europe",
	        "Aircraft",
	        "Basketball player", "Reptile", "Naval ship", "Mathematician", "Goddess", "Building or structure in India",
	        "Classical composition", "Newspaper", "Marvel Comics supervillain", "Comedy film", "Tree",
	        "Marvel Comics superhero", "Competitor at the 2008 Summer Olympics", "University or college in Asia",
	        "Racehorse",
	        "Toy", "Trade union", "Airport", "Ambassador", "Aquatic animal", "Baseball pitcher", "Fictional location",
	        "Mountain of Europe", "Fish", "Bishop", "Horror film", "Bridge", "Law enforcement organization",
	        "Journalist",
	        "Lake of North America", "Stadium", "Canadian political person", "Academic journal",
	        "Compilation album", "Protected area of Europe", "Leader of political parties", "Tribe", "Enzyme",
	        "Automotive company", "Heavy metal musician", "Natural disaster", "Illustrator", "Explosive weapon",
	        "Vegetable",
	        "Village in Poland", "Island of Asia", "Architect", "Dessert", "Economist", "Asteroid", "Space scientist",
	        "Mollusca", "Opera", "Murder", "Medical test", "Mayor", "Work about legendary creatures", "Skyscraper",
	        "Mountain of Asia", "Meat dish", "Equation", "Political office-holder in India", "Rocket or missile",
	        "Mythological place", "Bass guitarist", "Dialect", "Judge", "Artist from New York (state)",
	        "Music industry executive", "Fungus", "Newspaper published in Europe", "Fort", "Goalkeeper",
	        "District of London",
	        "Saxophonist", "Psychoactive drug", "Chess person", "Army general", "Amusement park", "Orbit",
	        "American biologist", "Bus company", "Empress", "Meme", "Japanese entertainer", "Progressive rock group",
	        "Fish of North America", "Astronomy organization", "Rebel militia group", "Japanese wrestler", "Castle",
	        "Volcano",
	        "British voice actor", "Christian saint of the Middle Ages", "German classical musician", "Japanese film",
	        "Body of water of New York (state)", "Cartoon", "Monk", "French musician", "Power station", "Satellite",
	        "Mine",
	        "Sculpture", "Village in India", "Jazz musician from New York (state)", "Bacterium", "Trade association",
	        "Reggae musician", "Jewish sportsperson", "Bundesliga player", "Palace", "Amphibian", "Photographer",
	        "Building or structure in Japan", "War crime", "Restaurant", "Lake of Canada", "Asian scientist", "Truck",
	        "Musical film", "Live album", "British humorist", "Desert flora", "Helicopter", "Body of water of India",
	        "University or college in India", "Soap opera actress", "Prison", "Warner Bros. film", "Observatory",
	        "Railway station located underground", "Medieval painter", "Butterfly", "Brazilian jiu-jitsu practitioner",
	        "Rodent", "Dam or reservoir", "Archer", "Politician in Ontario", "Animator", "Law school",
	        "Shopping district or street", "Main-belt asteroid", "Theatre", "Battle involving France", "Train",
	        "Ship of the Royal Navy", "Canal", "Moth", "Mobster", "Cemetery in Europe", "Vehicle simulation game",
	        "British criminal", "Murdered politician", "Membrane protein", "Children\'s book", "Brewery",
	        "World Heritage Site", "Korean actor", "Natural gas company", "Tank", "Castle in Europe",
	        "Insurance organization",
	        "Transport museum", "Bank", "Soft drink", "Truck manufacturer", "Aircraft engine",
	        "Victim of aviation accidents or incidents", "Pirate", "French film", "Ballet dancer", "Receptor",
	        "Trade union in Europe", "Earthquake", "Cretaceous dinosaur", "Festival in Europe", "Animation director",
	        "Waterfall", "Stadium in Europe", "Crustacean", "Shopping mall", "Church", "Monster movie",
	        "Flowering plant",
	        "Swedish musical group", "Beach", "Ice hockey centre", "European goddess", "Italian designer",
	        "Vegetable dish",
	        "Surgeon", "Book about war", "Ornithologist", "Submarine", "Military airbase", "Lake of Asia",
	        "Song about cities", "Middle Eastern deity", "Member of the Congress of the Philippines", "Frog",
	        "Aquatic mammal",
	        "Lepidoptera", "Telugu actor", "Tower in North America", "School in Maryland", "Hospital in Asia",
	        "Metal bridge",
	        "City or town in Piedmont", "Sibling duo", "Airline", "Pakistani cricketer", "Zoo", "Military cemetery",
	        "Monastery", "Park in New York (state)", "Casino", "Clothing company",
	        "Food and drink company of Oceania", "Yacht", "Mountain of the Alps", "Neuroscientist", "Genocide survivor",
	        "Parasite", "Song from musicals", "Seattle Mariners player", "River of Bavaria", "Tunnel in Europe",
	        "Irish poet",
	        "Climber", "Silent film", "Bengali politician", "Dog breeds", "Castle in Asia",
	        "Health minister", "Polish writer", "Female swimmer", "Brewery of Europe", "Toy company", "Mountain passes",
	        "Beauty queen", "French composer", "Castle in Scotland", "Endemic flora of Australia", "Soap opera",
	        "Female gymnast", "Operatic soprano", "German actress", "Serial killer", "Taekwondo practitioner",
	        "Pub in the United Kingdom", "Oil field", "Costume designer", "Beetle",
	        "Member of the Iowa House of Representatives", "Scholar of Sunni Islam", "Airport in Africa",
	        "British fort",
	        "Shark", "Clock tower", "Mine in Europe", "Polish footballer", "Catholic martyr", "Christmas album",
	        "Norwegian writer", "Female golfer", "Argentine film", "Lichen", "Cycling team", "Carnivorous plant",
	        "Documentary about sports", "Cocktail", "Aquarium", "Curling competition",
	        "Prison or jail in the United States",
	        "Fossil fuel power station", "Protected area of Florida", "Korean politician", "Marine mollusc",
	        "Soviet scientist", "Lake of Washington (state)", "Drink brand", "Fountain", "Reggae albums", "Canoeist",
	        "Dentist", "Windmill", "Cave of Europe", "Japanese motorcycle",
	        "Bobsledder", "Soviet black-and-white film", "Road in Malaysia", "Tanker", "Neighbourhood in Turkey",
	        "Member of the Parliament of Finland", "Vineyard or winery", "Lake of New Zealand",
	        "Dragonfly", "Dam in Europe", "Arachnid specie", "Windmill in the Netherlands", "Arthropod of India",
	        "Male hammer thrower", "German canoeist"]


def test_types():
	return [
		"Mean of transportation",
		"Painter",
		"Caribbean musician",
		"Japanese speculative fiction film",
		"Hotel",
		"Building or structure in Pennsylvania",
		"Philosopher",
		"Member of the Indiana General Assembly",
		"Museum in Ontario",
		"Ant",
		"Band",
		"Women computer scientist"
	]


def generate_embedding_batch(batch):
	eos_token = llm.tokenizer.eos_token
	texts = list(set(text + eos_token for text, indices in batch))
	tokens = llm.tokenize(texts).to('cuda')
	embeddings = llm.get_llm_at_layer(tokens, layer=layer)
	text_to_embedding = {text: encoding for text, encoding in zip(texts, embeddings)}
	bulk = []
	for text, indices in batch:
		text_in_llm = text + eos_token
		h = text_to_embedding[text_in_llm]
		llm_indices = llm.token_indices_given_text_indices(text_in_llm, indices)
		index_of_eos = llm.tokens_count(text_in_llm) - 1
		start = h[llm_indices[0] - 1]
		end = h[llm_indices[1]]
		eos = h[index_of_eos]
		bulk.append(
			{
				"start": start,
				"end": end,
				"eos": eos,
			}
		)
	del embeddings
	return bulk


def extract_elastic_block(elastic_block):
	results = []
	for item in elastic_block:
		text = item["all_text"]
		indices = (item["index_start"], item["index_end"])
		results.append((text, indices))
	return results





class NERtrieveDataProvider(AbstractDataProvider):


	def __init__(self):
		self.entity_name_embeddings = None
		self.semaphore = asyncio.Semaphore(20)

	async def yield_dataset(self,
			anchor_type, dataset_types, batch_size=50,
			instances_per_type=100,
			hard_negative_ratio=0,
			similarity_strategy='instance',
			llm_layer=None
	):
		async with self.semaphore:
			assert similarity_strategy in ('instance', 'type')

			extract = fewnerd_processor.extract_entities_from_es_response

			batches = [
				(start_index, min(start_index + batch_size, instances_per_type))
				for start_index in range(0, instances_per_type, batch_size)
			]
			batch_sizes = [end - start for start, end in batches]

			for batch_size in batch_sizes:
				random_query = queries.query_get_by_fine_grained_fewnerd_v3_randomized(
					fine_grained_type=[anchor_type],
					batch_size=batch_size,
					llm_layer=llm_layer,
					entity_type_key="entity_type"
				)
				if similarity_strategy == 'instance':
					anchor = await dataset_provider.search_async(index=ELASTIC_INDEX, query=random_query, size=1)
					anchor = extract(anchor["hits"]["hits"])
					assert len(anchor) == 1
					anchor = anchor[0]
					text = anchor["all_text"]
					result_type = anchor["entity_type"]

				else:
					text = anchor_type
					anchor = anchor_type
					result_type = anchor_type

				random_query["size"] = batch_size
				other_types = list(set(dataset_types) - {result_type})
				size_hard_negative = math.ceil(batch_size * hard_negative_ratio)
				query_hard_negative = queries.query_hard_negative(
					fine_grained_type=other_types,
					coarse_grained_type=None,
					anchor_text=text,
					size=size_hard_negative,
					entity_type_key="entity_type",
					llm_layer=llm_layer
				)
				size_easy_negative = batch_size - size_hard_negative
				query_easy_negative = queries.query_get_by_fine_grained_fewnerd_v3_randomized(
					fine_grained_type=other_types,
					batch_size=size_easy_negative,
					llm_layer=llm_layer,
					entity_type_key="entity_type"
				)
				bulk = [
					{},
					random_query,
					{},
					query_easy_negative,
					{},
					query_hard_negative,
				]
				for query in bulk:
					if query:
						query["_source"] = ["all_text", "index_start", "index_end"]
				response = await dataset_provider.multisearch(index=ELASTIC_INDEX, bulk=bulk)
				assert any(reply["_shards"]["failed"] != 0 for reply in response["responses"])
				good_batch, easy_negative, hard_negative = response["responses"]
				# good_batch = await dataset_provider.search_async(index=ELASTIC_INDEX, query=random_query)
				# easy_negative = await dataset_provider.search_async(index=ELASTIC_INDEX, query=query_easy_negative)
				# hard_negative = await dataset_provider.search_async(index=ELASTIC_INDEX, query=query_hard_negative)
				chunked_bad_batch = extract(hard_negative["hits"]["hits"]) + extract(easy_negative["hits"]["hits"])
				random.shuffle(chunked_bad_batch)
				chunked_good_batch = extract(good_batch["hits"]["hits"])
				assert len(chunked_good_batch) > 0
				assert len(chunked_bad_batch) == len(chunked_good_batch), (
					f"for type {anchor_type} and similaity {similarity_strategy} failed. "
					f"good batch: {len(chunked_good_batch)} "
					f"bad batch: {len(chunked_bad_batch)}")

				yield anchor, chunked_good_batch, chunked_bad_batch

	async def yield_train_dataset(
			self, anchor_type: str, batch_size: int, instances_per_type: int,
			hard_negative_ratio: int, llm_layer: int,
			similarity_strategy: str
	) -> AsyncIterator[Tuple[Any, Any, Any]]:
		all_types = train_types()
		iterator = self.yield_dataset(
			anchor_type=anchor_type,
			dataset_types=all_types,
			batch_size=batch_size,
			instances_per_type=instances_per_type,
			hard_negative_ratio=hard_negative_ratio,
			similarity_strategy=similarity_strategy,
			llm_layer=llm_layer
		)
		async for item in iterator:
			yield item

	async def yield_test_dataset(
			self, anchor_type: str, batch_size: int, instances_per_type: int,
			llm_layer: int, similarity_strategy: str
	) -> AsyncIterator[Tuple[Any, Any, Any]]:
		all_types = test_types()
		iterator = self.yield_dataset(
			anchor_type=anchor_type,
			dataset_types=all_types,
			batch_size=batch_size,
			instances_per_type=instances_per_type,
			hard_negative_ratio=0,
			similarity_strategy=similarity_strategy,
			llm_layer=llm_layer
		)
		async for item in iterator:
			yield item

	def pick_llm_output_for_document(
			self, device: torch.device, input_tokens: str, llm_layer: str,
			is_fine_tune_llm: bool, documents: List[Any]
	) -> torch.Tensor:
		assert is_fine_tune_llm == False
		if not isinstance(documents, str):
			elastic_block = extract_elastic_block(documents)
		else:
			elastic_block = [(documents, (0, len(documents)))]
		forward_in_llm = generate_embedding_batch(elastic_block)
		start = torch.stack([item["start"] for item in forward_in_llm])
		end = torch.stack([item["end"] for item in forward_in_llm])
		eos = torch.stack([item["eos"] for item in forward_in_llm])
		embedding = choose_llm_representation(end=end, start=start, eos=eos, input_tokens=input_tokens)
		embedding = embedding.to(device)
		del forward_in_llm, start, end, eos
		return embedding

	def train_fine_types(self) -> List[str]:
		"""Returns a list of fine-grained entity types for training."""
		# return ["Disease or disorder"]
		return train_types()

	def test_fine_types(self) -> List[str]:
		"""Returns a list of fine-grained entity types for testing."""
		return test_types()

	def load_entity_name_embeddings(self, layer_name: str, entity_name_strategy: str) -> Dict[str, torch.Tensor]:
		"""Loads embeddings for entity names."""
		if self.entity_name_embeddings:
			return self.entity_name_embeddings

		all_types = train_types() + test_types()
		texts_with_length = [(text, (0, len(text))) for text in all_types]
		forward_in_llm = generate_embedding_batch(texts_with_length)
		if entity_name_strategy == "end_eos":
			layer_to_tensor = {
				text: torch.cat((llm_output["end"], llm_output["eos"]))
				for text, llm_output in zip(all_types, forward_in_llm)
			}
		else:
			layer_to_tensor = {
				text: llm_output[entity_name_strategy]
				for text, llm_output in zip(all_types, forward_in_llm)
			}
		del forward_in_llm
		self.entity_name_embeddings = layer_to_tensor
		return layer_to_tensor

	def llm_and_layer_to_elastic_name(self, llm_id: str, layer: str) -> str:
		return fewnerd_dataset.llm_and_layer_to_elastic_name(
			llm_id=llm_id,
			layer=layer
		)


if __name__ == "__main__":
	fine_tune_llm = FineTuneLLM()

	####
	clearml_poc.clearml_init(queue_name='a100_gpu')
	mlp_args = Arguments()
	llm_args = FineTuneLLM()
	clearml_poc.clearml_connect_hyperparams(mlp_args, "mlp_args")
	clearml_poc.clearml_connect_hyperparams(llm_args, "llm_args")
	print('args are: ', json.dumps(mlp_args.__dict__, indent=4))
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	assert torch.cuda.is_available(), "no gpu available"

	layer = llm_args.layer
	llm = LLMInterface(
		interested_layers=[llm_args.layer],
		max_llm_layer=llm_args.max_llm_layer
	)
	llm.model.eval()

	# Instantiate the Fewnerd data provider
	nertrieve_data_provider = NERtrieveDataProvider()

	# Instantiate the abstract trainer with the Fewnerd data provider
	trainer = AbstractMLPTrainer(mlp_args, llm_args, clearml_poc, nertrieve_data_provider, device)

	# Run the training loop
	trainer.main_loop()

	print(f"Max benchmark achieved: {trainer.max_benchmark}")
