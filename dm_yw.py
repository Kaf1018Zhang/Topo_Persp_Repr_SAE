# dm_yw.py  — Yusu-Wang Discrete-Morse skeleton   (Gudhi ≥ 3.11)
# =============================================================
import numpy as np, networkx as nx
from scipy.ndimage import gaussian_filter
from gudhi import CubicalComplex

__all__ = ["yw_morse_skeleton"]

# -------------------------------------------------------------
def yw_morse_skeleton(points, *,            # (N,2) / (N,3) ndarray
                      grid_res=128,         # 体素分辨率
                      sigma=2.0,            # KDE 高斯平滑
                      eps_cancel=0.02,      # <0 关闭 persistence cancellation
                      return_complex=False):
    """Yusu-Wang Discrete Morse graph (调用 Gudhi C++ coreduction_gvf).

    Parameters
    ----------
    points : ndarray      N×2 或 N×3 点云
    grid_res : int        KDE 体素分辨率 (resᵈ 个体素)
    sigma : float         高斯核宽度
    eps_cancel : float    若 >0 取消持久度小于该值的 1-维對
    return_complex : bool 若 True 额外返回 CubicalComplex

    Returns
    -------
    G : networkx.Graph()  节点属性 xyz=坐标
    (option) cub : Gudhi CubicalComplex
    """
    d = points.shape[1]
    assert d in (2, 3), "仅支持 2-D / 3-D 点云"

    # === 1. KDE → voxel grid  ===============================
    mins, maxs = points.min(0), points.max(0)
    steps      = (maxs - mins) / grid_res
    voxel_idx  = ((points - mins) / steps).astype(int)
    voxel_idx  = np.clip(voxel_idx, 0, grid_res - 1)

    grid = np.zeros((grid_res,) * d, dtype=np.float32)
    np.add.at(grid, tuple(voxel_idx.T), 1.0)          # 计数
    grid = gaussian_filter(grid, sigma=sigma)

    # Gudhi CubicalComplex 约定：低值 = 高密度 ⇒ 取负
    cub = CubicalComplex(top_dimensional_cells=-grid)

    # === 2. Gudhi C++ Coreduction GVF  ======================
    try:
        gvf = cub.coreduction_gvf()      # Gudhi 3.11+
    except AttributeError:
        raise RuntimeError(
            "你的 Gudhi 版本不包含 coreduction_gvf() —— "
            "请执行  pip install 'gudhi[core] --upgrade'  "
            "或使用之前的纯-Python 实现"
        )

    # === 3. 可选：小持久度对 cancellation  ==================
    if eps_cancel > 0:
        cub.compute_persistence()
        pairs = [pair for pair in cub.persistence()
                 if pair[0] == 1
                 and pair[1][1] - pair[1][0] < eps_cancel]
        if pairs:
            cub.collapse_persistence_pairs(pairs=pairs)

    # === 4. 追踪 1-维流形 → skeleton graph  =================
    G = nx.Graph()

    # 所有 1-维 critical cells
    crit_1 = [c for c in cub.get_critical_cells()
              if cub.dimension(c) == 1]

    for cid in crit_1:
        u, v = _endpoints_along_gvf(cub, gvf, cid)
        for w in (u, v):
            if w not in G:
                G.add_node(w, xyz=_cell_coord(cub, w, d, mins, steps))
        G.add_edge(u, v)

    return (G, cub) if return_complex else G

# -----------------------------------------------------------------
# ↓↓↓ Helper functions ↓↓↓
# -----------------------------------------------------------------
def _cell_coord(cub, cell_id, d, mins, steps):
    """将 cell barycenter 还原到原始坐标系 (len=d)."""
    bc = np.array(cub.barycenter(cell_id)[:d])
    return (bc * steps + mins).tolist()

def _endpoints_along_gvf(cub, gvf, cid1):
    """给定 1-cell, 沿梯度向下走到两个 0-维 critical 节点."""
    ends = []
    for v in cub.boundary(cid1):   # 1-cell 的两个 0-cells
        cur = v
        # gvf: dict {higher_cell:int -> lower_cell:int}
        while cur in gvf:
            cur = gvf[cur]
        ends.append(cur)
    return tuple(ends)
