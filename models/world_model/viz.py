"""UMAP -> RGB map of cell embeddings (PNG + folium HTML)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def to_rgb(E: np.ndarray, seed: int = 0) -> tuple[np.ndarray, str]:
    try:
        import umap  # noqa: F401

        Z = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(E)
        how = "umap"
    except Exception:  # umap-learn missing/broken (numba vs numpy): fall back to PCA so the map still renders
        from sklearn.decomposition import PCA

        Z = PCA(n_components=3, random_state=seed).fit_transform(E)
        how = "pca"
    lo, hi = np.percentile(Z, 2, axis=0), np.percentile(Z, 98, axis=0)
    return np.clip((Z - lo) / np.maximum(hi - lo, 1e-9), 0, 1), how


def render(cells, E: np.ndarray, png: Path, html: Path | None = None, title: str = "", seed: int = 0) -> str:
    import h3
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    rgb, how = to_rgb(E, seed)
    polys = [[(lng, lat) for lat, lng in h3.cell_to_boundary(c)] for c in cells]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.add_collection(PolyCollection(polys, facecolors=rgb, edgecolors="none"))
    xy = np.concatenate(polys)
    ax.set_xlim(xy[:, 0].min(), xy[:, 0].max())
    ax.set_ylim(xy[:, 1].min(), xy[:, 1].max())
    ax.set_aspect(1 / np.cos(np.deg2rad(xy[:, 1].mean())))
    ax.set_axis_off()
    ax.set_title(f"{title}\ncells colored by 3-D {how.upper()} of learned embeddings (no labels used)", fontsize=9)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    if html is not None:
        try:
            import folium

            lat, lng = xy[:, 1].mean(), xy[:, 0].mean()
            m = folium.Map(location=[lat, lng], zoom_start=12, tiles="OpenStreetMap")
            for c, poly, col in zip(cells, polys, rgb):
                hexc = "#%02x%02x%02x" % tuple((col * 255).astype(int))
                folium.Polygon([(y, x) for x, y in poly], color=None, weight=0, fill=True, fill_color=hexc,
                               fill_opacity=0.75, tooltip=c).add_to(m)
            m.save(str(html))
        except ImportError:
            pass
    return how


def render_from_parquet(path: Path, cutoff: int, year: int, png: Path, html: Path | None, seed=0) -> str:
    df = pd.read_parquet(path)
    df = df[(df["cutoff"] == cutoff) & (df["year"] == year)]
    ecols = [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]
    return render(df["cell"].tolist(), df[ecols].to_numpy(np.float32), png, html,
                  title=f"SF World Model, cutoff={cutoff}, year={year}", seed=seed)
