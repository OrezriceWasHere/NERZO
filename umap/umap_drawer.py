#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import cm

## CONFIGURATION

# Path to your input JSON
INPUT_JSON     = 'result.json'

# Output filenames
OUTPUT_PDF     = 'umap_scatter_labeled.pdf'
OUTPUT_PNG     = 'umap_scatter_labeled.png'

# Exact class labels to drop
EXCLUDE_CLASSES = ['other','biologything','GPE','sportsteam','award']          # e.g. ['education', 'health']

# Keep only the N most frequent classes (or None to keep all)
TOP_N_CLASSES   = 20            # e.g. 5–10 for clarity

# Max points to plot (random subset), or None to plot everything
MAX_POINTS      = None

# RNG seed for reproducibility
SEED            = 42

## END CONFIGURATION

def main():
    # 0) Publication-style fonts
    plt.rcParams.update({
        'font.size':       12,
        'axes.titlesize':  14,
        'axes.labelsize':  13,
    })

    # 1) Load data
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    # 2) Gather embeddings & labels (drop excluded)
    embeddings, labels = [], []
    for entry in data.values():
        lbl = entry.get('fine_type')
        if lbl in EXCLUDE_CLASSES:
            continue
        if entry['embedding'][1] <= 0:
            continue
        embeddings.append(entry['embedding'])
        labels.append(lbl)

    embeddings = np.array(embeddings)
    labels     = np.array(labels)

    # 3) Keep only top-N classes by frequency
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

    # 5) Map each label to an index
    unique_labels = sorted(set(labels))
    label_to_idx  = {lbl: i for i, lbl in enumerate(unique_labels)}
    indices       = np.array([label_to_idx[l] for l in labels])
    C             = len(unique_labels)

    # 6) Choose a qualitative palette
    if C <= 10:
        cmap          = cm.get_cmap('tab10', C)
        palette       = [cmap(i) for i in range(C)]
    elif C <= 20:
        cmap          = cm.get_cmap('tab20', C)
        palette       = [cmap(i) for i in range(C)]
    else:
        c1 = cm.get_cmap('tab20', 20)
        c2 = cm.get_cmap('tab20b',20)
        c3 = cm.get_cmap('tab20c',20)
        palette = list(c1(range(20))) + list(c2(range(20))) + list(c3(range(20)))
        palette = palette[:C]

    colors = [palette[i] for i in indices]

    # 7) Plot scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        embeddings[:,0], embeddings[:,1],
        c=colors, s=3, alpha=0.6, linewidths=0, zorder=2
    )
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.set_title(f'2D UMAP of Top {C} fine_types with Cluster Labels')

    # 8) Compute centroids & add text labels
    for lbl, color in zip(unique_labels, palette):
        pts = embeddings[labels == lbl]
        if len(pts) == 0:
            continue
        centroid = pts.mean(axis=0)
        ax.text(
            centroid[0], centroid[1], lbl,
            color=color,
            fontsize=10,
            fontweight='bold',
            ha='center', va='center',
            bbox=dict(facecolor='gray', edgecolor='none', alpha=0.7, pad=1)
        )

    # 9) Finalize
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF, dpi=300, format='pdf')
    plt.savefig(OUTPUT_PNG, dpi=300)
    plt.show()

if __name__ == '__main__':
    main()