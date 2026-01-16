"""
Visualization script for Projection Layer Analysis.

This script visualizes the embeddings collected from run_analysis.py
using t-SNE and UMAP to show clustering patterns.

Usage:
    python visualize.py --input_dir ./results/20260116_123456
    python visualize.py --input_dir ./results/20260116_123456 --method tsne
"""

import os
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def compute_tsne(embeddings: np.ndarray, perplexity: int = 30, max_iter: int = 1000) -> np.ndarray:
    """Compute t-SNE embeddings."""
    from sklearn.manifold import TSNE

    logger.info(f"Computing t-SNE (perplexity={perplexity}, max_iter={max_iter})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=42)
    return tsne.fit_transform(embeddings)


def compute_umap(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Compute UMAP embeddings."""
    try:
        import umap
    except ImportError:
        logger.warning("UMAP not installed. Install with: pip install umap-learn")
        return None

    logger.info(f"Computing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeddings)


def compute_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA embeddings."""
    from sklearn.decomposition import PCA

    logger.info(f"Computing PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(embeddings)


def plot_embeddings(
    embeddings_2d: np.ndarray,
    problem_ids: np.ndarray,
    title: str,
    output_path: str,
    num_problems: int = None
):
    """Plot 2D embeddings colored by problem ID."""
    if num_problems is None:
        num_problems = len(np.unique(problem_ids))

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create colormap
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_problems)))
    if num_problems > 20:
        colors = plt.cm.viridis(np.linspace(0, 1, num_problems))

    # Plot each problem with a different color
    for prob_id in range(num_problems):
        mask = problem_ids == prob_id
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[prob_id % len(colors)]],
            label=f"Problem {prob_id}",
            alpha=0.7,
            s=50
        )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # Add legend (only show first 20 to avoid clutter)
    if num_problems <= 20:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved plot to {output_path}")


def plot_comparison(
    point1_2d: np.ndarray,
    point2_2d: np.ndarray,
    point3_2d: np.ndarray,
    problem_ids: np.ndarray,
    output_path: str,
    method_name: str = "t-SNE"
):
    """Plot side-by-side comparison of all three points."""
    num_problems = len(np.unique(problem_ids))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Create colormap
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, num_problems)))
    if num_problems > 20:
        colors = plt.cm.viridis(np.linspace(0, 1, num_problems))

    titles = [
        f"Point 1: After reasoner_to_weaver\n({method_name})",
        f"Point 2: Weaver Hidden States\n(Before weaver_to_reasoner, {method_name})",
        f"Point 3: After weaver_to_reasoner\n({method_name})"
    ]
    data = [point1_2d, point2_2d, point3_2d]

    for ax, embeddings, title in zip(axes, data, titles):
        for prob_id in range(num_problems):
            mask = problem_ids == prob_id
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=[colors[prob_id % len(colors)]],
                alpha=0.7,
                s=30
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")

    plt.suptitle(
        "Projection Layer Analysis: Embedding Clustering by Problem\n"
        "(Same color = same problem, different points = different random query latents)",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison plot to {output_path}")


def compute_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """Compute clustering quality metrics."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    metrics = {}

    try:
        metrics['silhouette'] = silhouette_score(embeddings, labels)
    except Exception as e:
        metrics['silhouette'] = None
        logger.warning(f"Could not compute silhouette score: {e}")

    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
    except Exception as e:
        metrics['calinski_harabasz'] = None
        logger.warning(f"Could not compute Calinski-Harabasz score: {e}")

    try:
        metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
    except Exception as e:
        metrics['davies_bouldin'] = None
        logger.warning(f"Could not compute Davies-Bouldin score: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Visualize Projection Layer Analysis Results")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory with embeddings.pt")
    parser.add_argument('--method', type=str, default='all', choices=['tsne', 'umap', 'pca', 'all'],
                        help="Dimensionality reduction method")
    parser.add_argument('--perplexity', type=int, default=30, help="t-SNE perplexity")
    parser.add_argument('--n_neighbors', type=int, default=15, help="UMAP n_neighbors")
    args = parser.parse_args()

    # Load embeddings
    embeddings_path = os.path.join(args.input_dir, "embeddings.pt")
    if not os.path.exists(embeddings_path):
        logger.error(f"Embeddings file not found: {embeddings_path}")
        return

    logger.info(f"Loading embeddings from {embeddings_path}")
    data = torch.load(embeddings_path, map_location='cpu', weights_only=False)

    # Convert BFloat16 to Float32 before numpy conversion
    point1 = data['point1_after_r2w'].float().numpy()
    point2 = data['point2_weaver_hidden'].float().numpy()
    point3 = data['point3_after_w2r'].float().numpy()
    problem_ids = data['problem_ids']
    config = data['config']

    logger.info(f"Loaded {len(problem_ids)} samples from {config['num_problems']} problems")
    logger.info(f"Each problem has {config['num_random_latents']} random latents")
    logger.info(f"Embedding dimension: {point1.shape[1]}")

    # Create output directory for visualizations
    viz_dir = os.path.join(args.input_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Compute clustering metrics on original embeddings
    logger.info("\n=== Clustering Metrics (Original High-Dim Space) ===")
    for name, embeddings in [("Point1 (after r2w)", point1),
                              ("Point2 (weaver hidden)", point2),
                              ("Point3 (after w2r)", point3)]:
        metrics = compute_clustering_metrics(embeddings, problem_ids)
        logger.info(f"\n{name}:")
        logger.info(f"  Silhouette Score: {metrics['silhouette']:.4f}" if metrics['silhouette'] else "  Silhouette Score: N/A")
        logger.info(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}" if metrics['calinski_harabasz'] else "  Calinski-Harabasz: N/A")
        logger.info(f"  Davies-Bouldin: {metrics['davies_bouldin']:.4f}" if metrics['davies_bouldin'] else "  Davies-Bouldin: N/A")

    # Compute and plot visualizations
    methods = ['tsne', 'umap', 'pca'] if args.method == 'all' else [args.method]

    for method in methods:
        logger.info(f"\n=== Computing {method.upper()} ===")

        if method == 'tsne':
            point1_2d = compute_tsne(point1, perplexity=args.perplexity)
            point2_2d = compute_tsne(point2, perplexity=args.perplexity)
            point3_2d = compute_tsne(point3, perplexity=args.perplexity)
            method_name = "t-SNE"
        elif method == 'umap':
            point1_2d = compute_umap(point1, n_neighbors=args.n_neighbors)
            point2_2d = compute_umap(point2, n_neighbors=args.n_neighbors)
            point3_2d = compute_umap(point3, n_neighbors=args.n_neighbors)
            method_name = "UMAP"
            if point1_2d is None:
                continue
        elif method == 'pca':
            point1_2d = compute_pca(point1)
            point2_2d = compute_pca(point2)
            point3_2d = compute_pca(point3)
            method_name = "PCA"

        # Plot individual
        plot_embeddings(
            point1_2d, problem_ids,
            f"Point 1: After reasoner_to_weaver ({method_name})",
            os.path.join(viz_dir, f"point1_{method}.png"),
            config['num_problems']
        )
        plot_embeddings(
            point2_2d, problem_ids,
            f"Point 2: Weaver Hidden States ({method_name})",
            os.path.join(viz_dir, f"point2_{method}.png"),
            config['num_problems']
        )
        plot_embeddings(
            point3_2d, problem_ids,
            f"Point 3: After weaver_to_reasoner ({method_name})",
            os.path.join(viz_dir, f"point3_{method}.png"),
            config['num_problems']
        )

        # Plot comparison
        plot_comparison(
            point1_2d, point2_2d, point3_2d,
            problem_ids,
            os.path.join(viz_dir, f"comparison_{method}.png"),
            method_name
        )

    # Save metrics to file
    metrics_path = os.path.join(args.input_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("=== Projection Layer Analysis Metrics ===\n\n")
        f.write(f"Number of problems: {config['num_problems']}\n")
        f.write(f"Random latents per problem: {config['num_random_latents']}\n")
        f.write(f"Latent length: {config['latent_len']}\n")
        f.write(f"Model: {config['model_name']}\n")
        f.write(f"Weaver checkpoint: {config['load_weaver_path']}\n\n")

        for name, embeddings in [("Point1 (after r2w)", point1),
                                  ("Point2 (weaver hidden)", point2),
                                  ("Point3 (after w2r)", point3)]:
            metrics = compute_clustering_metrics(embeddings, problem_ids)
            f.write(f"{name}:\n")
            f.write(f"  Silhouette Score: {metrics['silhouette']:.4f}\n" if metrics['silhouette'] else "  Silhouette Score: N/A\n")
            f.write(f"  Calinski-Harabasz: {metrics['calinski_harabasz']:.2f}\n" if metrics['calinski_harabasz'] else "  Calinski-Harabasz: N/A\n")
            f.write(f"  Davies-Bouldin: {metrics['davies_bouldin']:.4f}\n\n" if metrics['davies_bouldin'] else "  Davies-Bouldin: N/A\n\n")

    logger.info(f"\nSaved metrics to {metrics_path}")
    logger.info(f"Visualizations saved to {viz_dir}")
    logger.info("\nDone!")


if __name__ == "__main__":
    main()
