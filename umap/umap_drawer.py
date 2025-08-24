#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

##\ CONFIGURATION

# Path to your input JSON
INPUT_JSON     = 'result.json'

# Output filenames
OUTPUT_PDF     = 'umap_scatter_labeled.pdf'
OUTPUT_PNG     = 'umap_scatter_labeled.png'

# Exact class labels to drop
EXCLUDE_CLASSES = ['other','biologything','GPE','sportsteam','award']
# Only include fine types of these values
INCLUDE_FINE_TYPES = ['chemicalthing', 'sportsevent']

# Keep only the N most frequent classes (or None to keep all)
TOP_N_CLASSES   = 20

# Max points to plot (random subset), or None to plot everything
MAX_POINTS      = None

# RNG seed for reproducibility
SEED            = 42

##\ END CONFIGURATION

def main():
    # 0) Publication\-style fonts
    plt.rcParams.update({
        'font.size':       12,
        'axes.titlesize':  14,
        'axes.labelsize':  13,
    })

    # 1) Load data
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    # 2) Gather embeddings & labels (drop excluded; include desired fine types)
    embeddings, labels = [], []
    for entry in data.values():
        lbl = entry.get('fine_type')
        if lbl in EXCLUDE_CLASSES:
            continue
        if lbl not in INCLUDE_FINE_TYPES:
            continue
        if entry['embedding'][1] <= 0:
            continue
        embeddings.append(entry['embedding'])
        labels.append(lbl)

    embeddings = np.array(embeddings)
    labels     = np.array(labels)

    # 3) Keep only top\-N classes by frequency
    if TOP_N_CLASSES is not None:
        freq = Counter(labels)
        top_labels = {lbl for lbl, _ in freq.most_common(TOP_N_CLASSES)}
        mask = np.array([lbl in top_labels for lbl in labels])
        embeddings = embeddings[mask]
        labels     = labels[mask]

    # 4) Subsample points if needed
    N = len(labels)
    if MAX_POINTS is not None and N > MAX_POINTS:
        rng  = np.random.RandomState(SEED)
        keep = rng.choice(N, size=MAX_POINTS, replace=False)
        embeddings = embeddings[keep]
        labels     = labels[keep]

    # 5) Map each label to a color: chemicalthing as #4ED7F1, sportsevent as #FFFA8D
    label_to_color = {'chemicalthing': '#03A791', 'sportsevent': '#F1BA88'}
    colors = [label_to_color.get(lbl, 'black') for lbl in labels]

    # 6) Plot scatter without labels, axes, or title
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        embeddings[:, 0], embeddings[:, 1],
        c=colors, s=3, alpha=0.6, linewidths=0, zorder=2
    )
    ax.axis('off')

    # 7) Finalize with high dpi resolution
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=600, format='pdf')
    plt.savefig(OUTPUT_PNG, dpi=600)
    plt.show()

if __name__ == '__main__':
    main()