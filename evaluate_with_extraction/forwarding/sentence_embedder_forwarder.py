from typing import Dict, List, Tuple

from sentence_embedder import SentenceEmbedder

BATCH_SIZE = 32


def _forward_batch(batch: List[str], embedder: SentenceEmbedder) -> List[List[float]]:
    """Forward a batch of sentences through the embedder."""
    embeddings = embedder.forward_passage(batch)
    return [emb.cpu().tolist() for emb in embeddings]


def forward_dataset(
    records: Dict[str, Dict],
    embedder: SentenceEmbedder,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, List[float]]:
    """Generate an embedding for each record using the sentence embedder."""
    result: Dict[str, List[float]] = {}
    batch: List[Tuple[str, str]] = []

    for text_id, record in records.items():
        batch.append((text_id, record["sentence"]))
        if len(batch) >= batch_size:
            sentences = [text for _, text in batch]
            embs = _forward_batch(sentences, embedder)
            for (tid, _), emb in zip(batch, embs):
                result[tid] = emb
            batch = []

    if batch:
        sentences = [text for _, text in batch]
        embs = _forward_batch(sentences, embedder)
        for (tid, _), emb in zip(batch, embs):
            result[tid] = emb

    return result
