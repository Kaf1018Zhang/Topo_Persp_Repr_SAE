from __future__ import annotations
import math, itertools
from typing import Tuple, Dict

import numpy as np, networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

__all__ = ["discrete_morse_graph", "visualize_graph2d", "visualize_graph3d"]

def _normalise(pts: np.ndarray, pad=0.05):
    mins, maxs = pts.min(0), pts.max(0); span = maxs - mins
    mins -= pad*span; maxs += pad*span; span = maxs - mins
    return (pts - mins)/span, mins, span

def _voxelise(pts: np.ndarray, res: int):
    idx = np.clip((pts*res).astype(int), 0, res-1)
    grid = np.zeros((res,)*pts.shape[1], np.float32)
    np.add.at(grid, tuple(idx.T), 1.0)
    return grid, idx

_OFF = {2: np.array([[1,0],[-1,0],[0,1],[0,-1]]),
        3: np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])}

def _density_field(pts, res, sig): return gaussian_filter(_voxelise(pts,res)[0], sig)

def _discrete_gradient(fld: np.ndarray):
    shp=fld.shape; d=fld.ndim; off=_OFF[d]; V={}
    for idx in np.ndindex(shp):
        if idx in V: continue
        higher=[tuple(np.array(idx)+o) for o in off
                if all(0<=idx[i]+o[i]<shp[i] for i in range(d))
                and fld[tuple(np.array(idx)+o)]>fld[idx]]
        if len(higher)==1: V[idx]=higher[0]; V[higher[0]]=idx
    return V

def _desc(p,V,M):
    path=[p]; cur=p
    while cur not in M and cur in V:
        cur=V[cur]
        if cur in path: break
        path.append(cur)
    return path

def _grid_skeleton(fld,V):
    shp=fld.shape; d=fld.ndim; off=_OFF[d]
    mins={i for i in np.ndindex(shp) if i not in V}
    G=nx.Graph(); [G.add_node(m,xyz=np.array(m)+.5) for m in mins]
    saddles={v for v in V if v<V[v]}
    for s in saddles:
        lower=[tuple(np.array(s)+o) for o in off
               if all(0<=s[i]+o[i]<shp[i] for i in range(d)) and fld[tuple(np.array(s)+o)]<fld[s]]
        hit=set()
        for n in lower:
            p=_desc(n,V,mins)
            if p: hit.add(p[-1])
        if len(hit)==2:
            a,b=hit
            if a!=b: G.add_edge(a,b)
    return G, fld

def _simplify_graph_len(G: nx.Graph, eps: float):
    if eps<=0: return G
    H=G.copy()
    short=[(u,v) for u,v in H.edges
           if np.linalg.norm(np.asarray(H.nodes[u]['xyz'])-
                             np.asarray(H.nodes[v]['xyz']))<eps]
    H.remove_edges_from(short)
    H.remove_nodes_from([n for n in H if H.degree(n)==0])
    return H

def _simplify_graph_rho(G: nx.Graph, fld: np.ndarray, eps_rho: float):
    if eps_rho<=0: return G
    H=G.copy()
    drop=[]
    for u,v in H.edges:
        rho = abs(fld[u]-fld[v]) 
        if rho<eps_rho: drop.append((u,v))
    H.remove_edges_from(drop)
    H.remove_nodes_from([n for n in H if H.degree(n)==0])
    return H

def discrete_morse_graph(points: np.ndarray, *, grid_res=256, sigma=2.0,
                         pers_len=0.0, pers_rho=0.0,
                         normalise=True, visualize=False):
    assert points.ndim==2 and points.shape[1] in (2,3)
    pts,mins,span=_normalise(points) if normalise else (points,np.zeros(points.shape[1]),np.ones(points.shape[1]))
    fld=_density_field(pts,grid_res,sigma)
    V=_discrete_gradient(fld)
    Ggrid,fld=_grid_skeleton(fld,V)

    G=nx.Graph()
    for v in Ggrid.nodes:
        xyz=(np.array(v)+.5)/grid_res*span+mins
        G.add_node(v,xyz=xyz)
    G.add_edges_from(Ggrid.edges)

    G=_simplify_graph_len(G,pers_len)
    G=_simplify_graph_rho(G,fld,pers_rho)

    if visualize:
        if points.shape[1]==2: visualize_graph2d(G,points)
        else: visualize_graph3d(G,points)
    return G

def visualize_graph2d(G, pts=None, *, s=1, alpha=0.4, lw=1.2, c_pts='#999', c_edge='#d62728'):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    if pts is not None:
        ax.scatter(pts[:, 0], pts[:, 1], s=s, alpha=alpha, c=c_pts)
    
    for u, v in G.edges:
        p, q = G.nodes[u]['xyz'], G.nodes[v]['xyz']
        ax.plot([p[0], q[0]], [p[1], q[1]], c=c_edge, lw=lw)
    
    ax.set_aspect('equal')
    ax.set_title("Discrete-Morse skeleton (2D)")
    return ax

def visualize_graph3d(G,pts=None,*,sample=.05,edge_color='crimson',edge_width=2,marker_size=2):
    edges=[go.Scatter3d(x=[G.nodes[u]['xyz'][0],G.nodes[v]['xyz'][0]],
                        y=[G.nodes[u]['xyz'][1],G.nodes[v]['xyz'][1]],
                        z=[G.nodes[u]['xyz'][2],G.nodes[v]['xyz'][2]],
                        mode='lines',line=dict(color=edge_color,width=edge_width),showlegend=False)
           for u,v in G.edges]
    pts_tr=[]
    if pts is not None and sample>0:
        idx=np.random.choice(len(pts),int(len(pts)*sample),False)
        spts=pts[idx]
        pts_tr=[go.Scatter3d(x=spts[:,0],y=spts[:,1],z=spts[:,2],
                             mode='markers',marker=dict(size=marker_size,color='rgba(120,120,120,0.25)'),showlegend=False)]
    fig=go.Figure(data=pts_tr+edges)
    fig.update_layout(scene=dict(aspectmode='data'),
                      margin=dict(l=0,r=0,b=0,t=30),
                      title="Discrete-Morse Skeleton (3D)")
    return fig
