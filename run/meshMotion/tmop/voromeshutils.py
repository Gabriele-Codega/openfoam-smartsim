import torch
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)

def build_voronoi(seeds):
    """
    Build clipped Voronoi tessellation of unit square.

    seeds: (n,2) torch tensor
    returns:
        nodes: list of [x,y]
        elements: list of node-index lists (CCW ordered)
        cell_polygons: list of shapely Polygons (CCW oriented)
    """
    seeds_np = seeds.cpu().numpy()

    padding = 10.0
    ghost = np.array([
        [-padding, -padding],
        [-padding, 1+padding],
        [1+padding, -padding],
        [1+padding, 1+padding],
    ])

    all_pts = np.vstack([seeds_np, ghost])
    vor = Voronoi(all_pts)

    square = box(0.0, 0.0, 1.0, 1.0)

    nodes = []
    node_map = {}
    elements = []
    cell_polygons = []

    def get_node(pt):
        key = (round(pt[0], 12), round(pt[1], 12))
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append([pt[0], pt[1]])
        return node_map[key]

    for i in range(len(seeds)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if -1 in region or len(region) == 0:
            continue

        poly = Polygon([vor.vertices[v] for v in region])
        clipped = poly.intersection(square)

        if clipped.is_empty:
            continue

        # Force CCW orientation
        clipped = orient(clipped, sign=1.0)

        coords = list(clipped.exterior.coords)[:-1]
        element = [get_node(pt) for pt in coords]

        elements.append(element)
        cell_polygons.append(clipped)

    return nodes, elements, cell_polygons

def polygon_centroid(poly):
    """
    Compute centroid of shapely polygon.
    """
    c = poly.centroid
    return torch.tensor([c.x, c.y])

def lloyd_step(seeds):
    """
    Perform one Lloyd iteration.
    """
    _, _, cells = build_voronoi(seeds)

    new_seeds = []
    for cell in cells:
        new_seeds.append(polygon_centroid(cell))

    return torch.stack(new_seeds)


def compute_cvt(n_seeds, n_iters=10, seed=None):
    """
    Compute centroidal Voronoi tessellation in unit square.
    """
    if seed is not None:
        torch.manual_seed(seed)

    seeds = torch.rand(n_seeds, 2)

    for _ in range(n_iters):
        seeds = lloyd_step(seeds)

    nodes, elements, _ = build_voronoi(seeds)
    return seeds, nodes, elements

def plot_mesh(nodes, elements, seeds=None, ax=None, fill_kwargs={}):
    nodes = np.array(nodes)

    if ax is None:
        fig, ax = plt.subplots()

    for elem in elements:
        poly = nodes[elem]
        ax.fill(poly[:,0], poly[:,1],
                **fill_kwargs
               )

    if seeds is not None:
        s = seeds.numpy()
        ax.scatter(s[:,0], s[:,1], color="red", s=15)

    ax.set_aspect("equal")
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)
    return ax


