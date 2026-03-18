"""
core/embeddings.py
------------------
Embedding space analysis and projection.

Two visualisation modes:
  1. Frames of the current clip projected in 2D
  2. Same + top-N AudioSet class reference points
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

try:
    import umap as umap_lib
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

SEED = 42


def has_umap() -> bool:
    return _HAS_UMAP


def project_frames(
    embeddings: np.ndarray,
    method: str = "pca",
    **kwargs,
) -> np.ndarray:
    """
    Project (N_frames, 1024) embeddings to (N_frames, 2).
    """
    method = method.lower()

    if method == "pca":
        reducer = PCA(n_components=2, random_state=SEED)

    elif method == "umap":
        if not _HAS_UMAP:
            raise ImportError("umap-learn not installed. Run: pip install umap-learn")
        n_neighbors = min(kwargs.get("n_neighbors", 15), len(embeddings) - 1)
        reducer = umap_lib.UMAP(
            n_components=2, random_state=SEED,
            n_neighbors=n_neighbors,
            min_dist=kwargs.get("min_dist", 0.1),
        )

    elif method == "tsne":
        perplexity = min(kwargs.get("perplexity", 30), len(embeddings) - 1)
        reducer = TSNE(n_components=2, random_state=SEED,
                       perplexity=max(2, perplexity))
    else:
        raise ValueError(f"Unknown method: {method}")

    return reducer.fit_transform(embeddings)


def get_class_reference_points(
    embeddings: np.ndarray,
    scores: np.ndarray,
    class_names: list,
    top_n: int = 10,
) -> tuple[np.ndarray, list]:
    """
    Build synthetic reference points for the top-N AudioSet classes.
    Weighted mean of frame embeddings by each class score.
    """
    mean_scores = scores.mean(axis=0)
    top_indices = np.argsort(mean_scores)[::-1][:top_n]

    ref_embeddings = []
    ref_labels     = []

    for idx in top_indices:
        weights       = scores[:, idx]
        weights       = weights / (weights.sum() + 1e-9)
        weighted_mean = (embeddings * weights[:, np.newaxis]).sum(axis=0)
        ref_embeddings.append(weighted_mean)
        ref_labels.append(class_names[idx])

    return np.array(ref_embeddings), ref_labels


def project_with_references(
    embeddings: np.ndarray,
    scores: np.ndarray,
    class_names: list,
    method: str = "pca",
    top_n_refs: int = 10,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, list, list, list | None]:
    """
    Project clip frames + AudioSet reference points together.

    Returns
    -------
    proj_frames   : (N_frames, 2)   projected frame embeddings
    proj_refs     : (top_n, 2)      projected reference class points
    frame_labels  : list of str     top-1 class per frame
    ref_labels    : list of str     reference class names
    explained_var : list | None     PCA explained variance ratio (or None)
    """
    # top-1 class label per frame
    frame_top1 = [class_names[np.argmax(scores[i])] for i in range(len(scores))]

    # reference points
    ref_embs, ref_labels = get_class_reference_points(
        embeddings, scores, class_names, top_n=top_n_refs
    )

    # stack frames + refs and project together so they share the same space
    all_embs = np.vstack([embeddings, ref_embs])
    proj_all = project_frames(all_embs, method=method, **kwargs)

    # split back into frames and references
    n_frames    = len(embeddings)
    proj_frames = proj_all[:n_frames]
    proj_refs   = proj_all[n_frames:]

    # explained variance — only available for PCA
    explained_var = None
    if method == "pca":
        pca = PCA(n_components=2, random_state=SEED)
        pca.fit(all_embs)
        explained_var = pca.explained_variance_ratio_.tolist()

    return proj_frames, proj_refs, frame_top1, ref_labels, explained_var