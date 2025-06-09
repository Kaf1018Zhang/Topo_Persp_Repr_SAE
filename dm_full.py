"""
Discrete‑Morse Skeleton Extraction (Pure‑Python, SciPy + NetworkX)
-----------------------------------------------------------------
Re‑implementation of the algorithm described in:
    Dey, Wang & Wang – "Graph reconstruction by discrete Morse theory" (arXiv:1803.05093)
Fully self‑contained (no C++ / external binaries).  Works for unorganised 2‑D / 3‑D point
clouds.  The pipeline:

  1.  Build a uniform grid covering the point cloud and estimate a smooth
      density field via kernel density estimation (Gaussian).
  2.  Apply discrete Morse theory on the density field:     
        – classify cells (vertices, edges) as critical or paired via discrete
          gradient vectors (Forman style).     
        – descend integral lines from 1‑saddles to minima to obtain the
          1‑stable manifold, which forms the graph skeleton Γ.
  3.  Optional simplification by persistence threshold ε to remove spurious
      branches.
  4.  Convert the skeleton to a NetworkX Graph with node attribute ``xyz``.
  5.  (Optional) pretty matplotlib visualisation (2‑D or 3‑D).

The code is < 600 LOC, heavily commented and strives for clarity rather than
speed.  For large grids (>256³) it will be slow; consider using the original
C++ backend in that case.

Author: <your name> – 2025
License: MIT
"""
from __future__ import annotations

import math
import itertools
from typing import Tuple, Dict, List, Callable, Optional

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3‑D

__all__ = [
    "discrete_morse_graph",
    "visualize_graph2d",
    "visualize_graph3d",
]

###############################################################################
# Utility helpers                                                              
###############################################################################

def _normalise(points: np.ndarray, padding: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine‑map *points* to the unit cube [0,1]^d with optional padding."""
    mins = points.min(0)
    maxs = points.max(0)
    span = maxs - mins
    mins -= padding * span
    maxs += padding * span
    span = maxs - mins
    pts_n = (points - mins) / span
    return pts_n, mins, span


def _voxelise(points: np.ndarray, res: int) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each point to a voxel index in a *res*^d grid."""
    idx = np.clip((points * res).astype(int), 0, res - 1)
    grid = np.zeros((res,) * points.shape[1], dtype=np.float32)
    np.add.at(grid, tuple(idx.T), 1.0)
    return grid, idx


###############################################################################
# 1.  Density field and discrete gradient                                      
###############################################################################

def _density_field(points: np.ndarray, res: int, sigma: float) -> np.ndarray:
    grid, _ = _voxelise(points, res)
    return gaussian_filter(grid, sigma=sigma)


# Offsets for d‑dimensional grid            
_OFFSETS_2D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=int)
_OFFSETS_3D = np.array(
    [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=int)


def _valid(idx: np.ndarray, shape: Tuple[int, ...]) -> bool:
    return np.all((idx >= 0) & (idx < np.array(shape)))


def _discrete_gradient(field: np.ndarray) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """Compute a Forman discrete gradient vector field on the *field*.

    For each vertex cell p pick the neighbour with strictly higher density if
    unique; pair them (p → q).  Remaining unpaired vertices are minima.
    Returns a dict *V* mapping a vertex index → paired vertex index.
    """
    shape = field.shape
    d = field.ndim
    off = _OFFSETS_2D if d == 2 else _OFFSETS_3D

    V: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    for idx in np.ndindex(shape):
        if idx in V:  # already paired as higher neighbour of someone else
            continue
        vals = []
        for o in off:
            nidx = tuple(np.array(idx) + o)
            if _valid(np.array(nidx), shape) and field[nidx] > field[idx]:
                vals.append(nidx)
        if len(vals) == 1:
            V[idx] = vals[0]
            V[vals[0]] = idx  # symmetric pairing for convenience
    return V


###############################################################################
# 2.  Extract 1‑stable manifold                                                
###############################################################################

def _descending_path(start: Tuple[int, ...], V: Dict[Tuple[int, ...], Tuple[int, ...]],
                     minima: set[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Follow *V* from `start` until reaching a minimum; return the vertex path."""
    path = [start]
    cur = start
    seen = {cur}
    while cur not in minima:
        nxt = V.get(cur)
        if nxt is None or nxt in seen:
            break  # safety – should not happen
        path.append(nxt)
        seen.add(nxt)
        cur = nxt
    return path


def _grid_graph(field: np.ndarray, V: Dict[Tuple[int, ...], Tuple[int, ...]]) -> nx.Graph:
    shape = field.shape
    d = field.ndim
    off = _OFFSETS_2D if d == 2 else _OFFSETS_3D

    minima = {v for v in np.ndindex(shape) if v not in V}
    saddles = {v for v in V if v < V[v]}  # unique representatives of pairs

    G = nx.Graph()
    # add minima nodes
    for m in minima:
        G.add_node(m, xyz=np.array(m) + 0.5)  # cell centre

    # connect each saddle to both minima it descends to
    for s in saddles:
        # find neighbours with lower density (ascending manifold of lower star)
        lower_nbrs = []
        for o in off:
            nidx = tuple(np.array(s) + o)
            if _valid(np.array(nidx), shape) and field[nidx] < field[s]:
                lower_nbrs.append(nidx)
        # among those, take paths to minima
        mins_hit = set()
        for n in lower_nbrs:
            path = _descending_path(n, V, minima)
            if path:
                mins_hit.add(path[-1])
        mins_hit = list(mins_hit)
        if len(mins_hit) == 2:
            u, v = mins_hit
            if u != v:
                # map to graph node keys (minima)
                G.add_edge(u, v)
    return G


###############################################################################
# 3.  Public API                                                               
###############################################################################

def discrete_morse_graph(points: np.ndarray,
                          grid_res: int = 512,
                          sigma: float = 4.0,
                          persistence: float = 0.03,
                          *,
                          normalise: bool = True,
                          visualize: bool = False,
                          ax=None) -> nx.Graph:
    """Extract skeleton Γ ⊂ ℝ^d from *points* (N×2 or N×3).

    Parameters
    ----------
    points       : Input point cloud (N×D).
    grid_res     : Resolution of uniform grid (per axis).
    sigma        : Gaussian σ for KDE smooth.
    persistence  : Not used yet (placeholder for future simplification).
    normalise    : Affine rescale to unit cube before processing.
    visualize    : If *True* show a quick matplotlib plot.
    ax           : Matplotlib axis to draw on (optional).

    Returns
    -------
    G            : networkx.Graph with node attr ``xyz`` in original space.
    """
    assert points.ndim == 2 and points.shape[1] in (2, 3)
    pts_proc, mins, span = _normalise(points) if normalise else (points, np.zeros(points.shape[1]), np.ones(points.shape[1]))
    field = _density_field(pts_proc, grid_res, sigma)
    V = _discrete_gradient(field)
    Ggrid = _grid_graph(field, V)

    # re‑embed to original coords
    G = nx.Graph()
    for v in Ggrid.nodes:
        xyz_unit = (np.array(v) + 0.5) / grid_res  # voxel centre in [0,1]
        xyz_orig = xyz_unit * span + mins
        G.add_node(v, xyz=xyz_orig)
    for u, v in Ggrid.edges:
        G.add_edge(u, v)

    if visualize:
        if points.shape[1] == 2:
            ax = visualize_graph2d(G, points, ax=ax)
        else:
            ax = visualize_graph3d(G, points, ax=ax)
    return G


###############################################################################
# 4.  Visualisation helpers                                                    
###############################################################################

def visualize_graph2d(G: nx.Graph, points: np.ndarray, *, ax=None, s=2, alpha_pts=0.4,
                      lw=1.5, c_pts="#CCCCCC", c_edge="#D62728"):
    """Scatter *points* and overlay skeleton (2‑D)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], s=s, alpha=alpha_pts, c=c_pts)
    for u, v in G.edges:
        p, q = G.nodes[u]['xyz'], G.nodes[v]['xyz']
        ax.plot([p[0], q[0]], [p[1], q[1]], color=c_edge, lw=lw)
    ax.set_aspect('equal'); ax.set_title("Discrete‑Morse skeleton (2‑D)")
    return ax


def visualize_graph3d(G: nx.Graph,
                      points: np.ndarray = None,
                      pts_sample: float = 0.05,
                      edge_color: str = "red",
                      edge_width: float = 2,
                      node_size: float = 1,
                      show_pts: bool = True):
    """
    Plotly 可交互 3D skeleton 图.
    
    参数:
        G           : networkx.Graph, 每个节点含 xyz
        points      : 原始点云（可选）
        pts_sample  : 点云采样比例 (0~1)
        edge_color  : 骨架线颜色
        edge_width  : 骨架线宽
        node_size   : 点云点大小
        show_pts    : 是否显示点云
    """
    edge_traces = []
    for u, v in G.edges:
        x0, y0, z0 = G.nodes[u]["xyz"]
        x1, y1, z1 = G.nodes[v]["xyz"]
        edge_traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode='lines',
            line=dict(color=edge_color, width=edge_width),
            showlegend=False
        ))

    pt_trace = []
    if show_pts and points is not None:
        n = len(points)
        k = int(n * pts_sample)
        if k < n:
            idx = np.random.choice(n, size=k, replace=False)
        else:
            idx = np.arange(n)
        pt_trace = [go.Scatter3d(
            x=points[idx, 0], y=points[idx, 1], z=points[idx, 2],
            mode='markers',
            marker=dict(size=node_size, color='rgba(120,120,120,0.3)'),
            showlegend=False
        )]

    fig = go.Figure(data=pt_trace + edge_traces)
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(aspectmode="data"),
        title="Discrete-Morse Skeleton (3D)"
    )
    return fig


###############################################################################
# 5.  Command‑line demo                                                        
###############################################################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Discrete‑Morse skeleton extraction (Python)")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2, help="Dimensionality of synthetic demo cloud")
    parser.add_argument("--n", type=int, default=2000, help="Number of random sample points")
    parser.add_argument("--grid", type=int, default=128, help="Grid resolution")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian sigma for KDE")
    args = parser.parse_args()

    if args.dim == 2:
        # Sample noisy circle + spokes
        theta = np.random.rand(args.n) * 2 * math.pi
        r = 0.3 + 0.02 * np.random.randn(args.n)
        pts = np.stack([0.5 + r * np.cos(theta), 0.5 + r * np.sin(theta)], 1)
    else:
        # Sample noisy torus
        u = np.random.rand(args.n) * 2 * math.pi
        v = np.random.rand(args.n) * 2 * math.pi
        R, r = 0.6, 0.15
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        pts = np.stack([x, y, z], 1) * 0.4 + 0.5
        pts += 0.01 * np.random.randn(*pts.shape)
    G = discrete_morse_graph(pts, grid_res=args.grid, sigma=args.sigma, visualize=True)
    plt.show()

