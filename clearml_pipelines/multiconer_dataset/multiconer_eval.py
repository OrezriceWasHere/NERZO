import asyncio
import json
import os

import torch
from clearml import Dataset
from tqdm import tqdm

import clearml_helper
import clearml_poc
import dataset_provider
from clearml_pipelines.fewnerd_pipeline import fewnerd_dataset
from clearml_pipelines.nertreieve_dataset import nertrieve_processor
from contrastive import fewnerd_processor
from contrastive.args import FineTuneLLM
from contrastive.retrieval_eval import RetrievalEval
from llm_interface import LLMInterface
from sentence_embedder import SentenceEmbedder


class MulticonerEval(RetrievalEval):


	def __init__(self, layer, entity_to_embedding):
		super(MulticonerEval, self).__init__(
			index='multiconer_validation,multiconer_train,multiconer_test',
			layer=layer,
			entity_to_embedding=entity_to_embedding,
		)

	def entity_type_field_name(self):
		return 'fine_type'

	def anchors(self):
		return {
			'artist': ['multiconer_159963909ba8142cd7f0268998f6a560691ac78f',
			           'multiconer_94de44911963f43002b5c6ea30340d978350287e',
			           'multiconer_dde2f5f3d4e388fbea5b66f4eaa4aca4ad656066'],
			'humansettlement': ['multiconer_617350df56e159e675fb6083af26bf80281bd41e',
			                    'multiconer_b1e0e0ddce69792b5764b81be4b7dc564243564f',
			                    'multiconer_da0e0d314f0551ec027d1075fb9fc3ce50ef5792'],
			'athlete': ['multiconer_4b39a0e8cc2f3cc1790bf5c7c87216ac8ad2e5b4',
			            'multiconer_1ec6fa4bf2e9cb59fcdeb9b7070819d1a1c842a6',
			            'multiconer_5d0f31cce35a453013b002f6f132dcb4888f0e32'],
			'org': ['multiconer_83f82444c84ac398e22caba936227dd11ac6ebae',
			        'multiconer_4ce7c9a49854b3ffb22015a286416045a6d0b64a',
			        'multiconer_1f39c4e426b376476611e79d02349bec31f15b09'],
			'otherper': ['multiconer_6d00bf1347ee7217d80d2edc358f4789925bcc6a',
			             'multiconer_ea8152df5350381149cc16f1ad29a4ec459d6a1c',
			             'multiconer_c5ba73496970630c0b211a9c705110f8576811b4'],
			'visualwork': ['multiconer_cdd57d728dc9e6c67e76d672f4eeaac6a091eabd',
			               'multiconer_07b6a8b69992b2e6049fe849861084763c3b60b1',
			               'multiconer_6cf08c0ecb8a1af1e72e449ec58e430e0497f1b4'],
			'facility': ['multiconer_4d406fccff57fe3f10b71fc6ab48df140eb10d78',
			             'multiconer_84c610ade3bb60ea029de16b4d06fd9b9c530bdc',
			             'multiconer_8787e27ef8607f55fd4da98c0619307af187b499'],
			'writtenwork': ['multiconer_cce142249bf72a4443f5fd46df675fc82faaf7d3',
			                'multiconer_1906422e6a6217e35e83d9b2eb5cdb87189b3ace',
			                'multiconer_dd3d207adfd38819a8dd46315e669327e0f5357f'],
			'politician': ['multiconer_d52426968f1432fd6c4a10619cef0a4692518514',
			               'multiconer_5fda04fcdbf85a22ff81fec7e6b110e2ad763e80',
			               'multiconer_9aa5adb46be7e4ebe68e0233434263380ee05415'],
			'musicalwork': ['multiconer_e05758e28ad5f178d61e38dcc09545f7ace81029',
			                'multiconer_676865c0a9aeff5bf85a9bbb5ecc96b5246a5fc5',
			                'multiconer_0e58ecce1ed7d4d4bd6dd24f2275c4cb4fd26027'],
			'sportsgrp': ['multiconer_e911d3079c90d309c65a8e3be6b25f458c21c831',
			              'multiconer_7f9e7e691b66ae9956c7d2bf5f909de76477323f',
			              'multiconer_beadc019f548bc83be37e49654971839402d7f35'],
			'musicalgrp': ['multiconer_483b7916ef10269a70e8381fdbb32e3b038401cf',
			               'multiconer_ae59fd89ae9fbbf6a395f8c9a89eaa8ad1a9da5c',
			               'multiconer_6d9220a49263518421e2d379625e51a28b3ac921'],
			'otherprod': ['multiconer_292648d959c941f5913a4cd4ae9e8c660c7f35c9',
			              'multiconer_70fa32da54848704e7ef40ee8929b5eef714ba28',
			              'multiconer_e82d4745a878e9b85ec73fcdc2bf64ccf1a0ffc1'],
			'software': ['multiconer_8365060d463eb6c2e5ffc6b9367905f320c304db',
			             'multiconer_98186b645d3ce9cf929674978e10ca8e47deaeb8',
			             'multiconer_a9f6ad3c0043942b1c674782a984fa36de3650e6'],
			'publiccorp': ['multiconer_0f360edf7a58a6277d3ff2b177e683eac601451f',
			               'multiconer_abc0d792df3c13848168bec17e57ca6d639dbda0',
			               'multiconer_f8157a974cf12582c4404b6adff6989bf88cfa6f'],
			'anatomicalstructure': ['multiconer_a4e66d6b61ec63ca84f891fdba3176b6e8840cf9',
			                        'multiconer_c13708a8e67733acca531444d6d0b4405cf52c0c',
			                        'multiconer_8eea6d03cf1343bdf16b07fe70ebc044fea8e782'],
			'station': ['multiconer_2ba9de3a4a5d9aaf6f3b5bb2f6191959ae7a0d93',
			            'multiconer_8e566c88a630023022a23534b3682a46bc4ba549',
			            'multiconer_4d03ca5b5f338d7aa947e30bd82506ebc76c6936'],
			'vehicle': ['multiconer_3d11198bbd90b859ce64c5bdfed17eadfdbe8e34',
			            'multiconer_88d22b70e53de0747b3bc41062b578bded0e179d',
			            'multiconer_d145fb14d5e7b532b8b16a2c4954f99d26807948'],
			'disease': ['multiconer_20ad2e07977817bd6ae03ba1137898d01dc13195',
			            'multiconer_0a4a5cfe0af1e59b8410cd8d09a2997eaec2fb5c',
			            'multiconer_faebdbac4a5155ff1552c20cea2814bc2d5e9913'],
			'medication/vaccine': ['multiconer_c5acd2a9f9679f7c6cf18b9024438ed620227940',
			                       'multiconer_c670b1c5e12eceeced24380c944a4b2486115c7f',
			                       'multiconer_c23cc6f3bb278f3674cf74cacd5550268ca91759'],
			'food': ['multiconer_c0d1cbe7d3635bc6614a89bb3460935b59853759',
			         'multiconer_06d9b63f5d19ecdfd6eb66c47430da90fd4840a0',
			         'multiconer_fe1cd0f6345de83c2188fc68aac802c6088871cc'],
			'sportsmanager': ['multiconer_2c3fa004106f41c45fcf35506235f44045212262',
			                  'multiconer_2f89f09b17f9fa36b214598c1b87245e540a59cc',
			                  'multiconer_415535799a2f2c996e3bc50db5579da3e44f7f77'],
			'scientist': ['multiconer_97297d7007415592934546d7fbae4d78b08a3d0f',
			              'multiconer_36a583abda870ed55c0db8ff712e0e7b47b773b3',
			              'multiconer_bfb752743b3d3446a6756968f1f7dc2fde739c75'],
			'cleric': ['multiconer_fdbebeea3bf66a02cdf75006fd2b0f91c4a6b6f5',
			           'multiconer_2961a84eb1e2f4f9f9a7ad1572f8da32e469ebc2',
			           'multiconer_5e8be1490cdfe2f1902982eb42defa3a20bf294b'],
			'otherloc': ['multiconer_09d97a7919b7548c2f7639511481948ed2fb1682',
			             'multiconer_ccbb15b5586f8e25d29f201fccd2d69d03d68d30',
			             'multiconer_08f3e8b4a0b9021fd951b881f7ab2af2ea786c88'],
			'medicalprocedure': ['multiconer_bb12cb7ef4e78f595fe2a5fd2778d09d7821aca0',
			                     'multiconer_71ab9f792ac4cc47602363afd9201a91aa22a5dc',
			                     'multiconer_172eec0e5b2c6702ee91952d7e61353da7a9c634'],
			'carmanufacturer': ['multiconer_bad1e5b2dba5da78a4356183e07f90dea8af81ed',
			                    'multiconer_d7bfa791a6e19b24da893c6ede7bdce8f33c68c3',
			                    'multiconer_1508beefc935c07c344935002ccd9e41dad57167'],
			'clothing': ['multiconer_2133e0d73dd77c710e91686d83104c373bb96ad7',
			             'multiconer_dcee01923213e88ccc36a5b6529a8d35d14df856',
			             'multiconer_e9d1e1870dd67f29eacb7ed84540cb4d593502e7'],
			'drink': ['multiconer_3eecd5b44097452ae07f381712f214eb9973edf3',
			          'multiconer_2e87309fecc2151689379b7112f85802934da37f',
			          'multiconer_5ddaa5ba0701480688fe13a741577abda8bcac33'],
			'symptom': ['multiconer_909f9bba8eff8953883c32220a4f5ee1da9dd234',
			            'multiconer_66def5bcfda32265a72aae9f55d37dbe51e36271',
			            'multiconer_86dcfaa42a9cfa573bc8aeff7ab7d628a66a75fd'],
			'artwork': ['multiconer_4cac51bb85bd6276dbe46d8b8b0e0d0cb33cc416',
			            'multiconer_b71649a49a1dfac3f7871ef99820e03029379332',
			            'multiconer_9c0a47ab187c9d668ced3d01365013a843a08673'],
			'aerospacemanufacturer': ['multiconer_c8e3bb239b9df47487b4e38b7a5002ef125ec9f8',
			                          'multiconer_1c20e3b476e8ed6416b05036062706730d8f173f',
			                          'multiconer_0039e6793e965623c1264133a423b57b4b2dcfd3'],
			'privatecorp': ['multiconer_1bfb6f68ef4424a223edecc3bf94c0cf5d4d106f',
			                'multiconer_abbade5cfbb7d06993be76106ccbd748c6fce25d',
			                'multiconer_52e0400961d3fc5396bed80f510bda66e73786d1']
		}

	def calc_text_id_to_labels(self):
		dataset_dir = Dataset.get(
			dataset_name="text_id_to_labels.json",
			dataset_tags=['entity_to_labels'],
			dataset_project="multiconer_pipeline"
			).get_local_copy()

		path = os.path.join(dataset_dir, 'text_id_to_labels.json')
		with open(path, 'r') as f:
			text_to_labels = json.load(f)

		return text_to_labels

	def get_embedding_field_name(self):
		return 'nvidia/nv-embed-v2@output'

if __name__ == '__main__':
	layer_obj = {
		"layer_id": "48d1f5c0237149aa9dedd0c028b25b3c",
		"llm_layer": FineTuneLLM.layer,
		"llm_id": FineTuneLLM.llm_id,
		"elasticsearch_index": 'multiconer_validation,multiconer_test,multiconer_train',
	}
	clearml_poc.clearml_init(task_name='eval multiconer', queue_name='dsicsgpu')


	clearml_poc.clearml_connect_hyperparams(name='eval multiconer', hyperparams=layer_obj)
	llm_layer = layer_obj["llm_layer"]
	llm_id = layer_obj["llm_id"]
	args = clearml_helper.get_args_by_mlp_id(layer_obj['layer_id'])
	layer = layer_obj['layer_id']
	mlp = clearml_helper.get_mlp_by_id(layer)
	mlp = mlp.double()
	#
	# if args.input_tokens != "start_eos_pair":
	# 	entity_type_to_embedding = fewnerd_processor.load_entity_name_embeddings(
	# 		layer_name=fewnerd_dataset.llm_and_layer_to_elastic_name(
	# 			llm_id=llm_id,
	# 			layer=llm_layer
	# 		),
	# 		index='multiconer_entity_name_to_embedding',
	# 		entity_name_strategy=args.entity_name_embeddings
	# 	)
	# else:
	# 	entity_types = dataset_provider.search(query={"query":{"match_all":{}}}, index="multiconer_entity_name_to_embedding", size=100)
	# 	entity_type_to_embedding = {}
	# 	layer_name = fewnerd_dataset.llm_and_layer_to_elastic_name(
	# 		llm_id=llm_id,
	# 		layer=llm_layer
	# 	)
	# 	embedding_per_fine_type = {}
	# 	for db_record in tqdm(entity_types["hits"]["hits"]):
	# 		entity_name = db_record["_source"]["entity_name"].lower()
	# 		embedding_per_fine_type[entity_name] = db_record["_source"]["intfloat/e5-mistral-7b-instruct@output@output"]
	llm_args = FineTuneLLM()
	entity_types = dataset_provider.search(
			query={"query": {"match_all": {}}}, index="multiconer_entity_name_to_embedding", size=100
			)

	entity_type_to_embedding = {}
	for db_record in tqdm(entity_types["hits"]["hits"]):
			name = db_record["_source"]["entity_name"].lower()
			embedding = db_record["_source"]["nvidia/nv-embed-v2@output"]
			entity_type_to_embedding[name] = embedding
			# embedding = db_record["_source"]["intfloat/e5-mistral-7b-instruct@output"]
			# embedding = mlp(torch.tensor(llama_embedding)).tolist()
			entity_type_to_embedding[name] = embedding
		# end =  torch.tensor(db_record["_source"][f'embedding.{layer}.end'])
		# eos =  torch.tensor(db_record["_source"][f'embedding.{layer_name}.eos'])
		#



	layer = layer_obj["layer_id"]
	llm_layer = layer_obj["llm_layer"]
	llm_id = layer_obj["llm_id"]
	multiconer = MulticonerEval(layer=layer, entity_to_embedding=entity_type_to_embedding,)
	multiconer.eval_zero_shot()
