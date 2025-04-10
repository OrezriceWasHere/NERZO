from sentence_embedder import SentenceEmbedder

sentence_embedder = SentenceEmbedder("nvidia/NV-Embed-v2")



if __name__ == "__main__":
	# Test the SentenceEmbedder
	passage = "This is a test passage."
	query = "What is the test passage about?"

	passage_embedding = sentence_embedder.forward_passage(passage)
	query_embedding = sentence_embedder.forward_query(query)

	print("Passage Embedding:", passage_embedding)
	print("Query Embedding:", query_embedding)