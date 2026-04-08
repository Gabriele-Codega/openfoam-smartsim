from tmop import Config, TMOPMesh

import torch
import numpy as np
from smartredis import Client
try:
    import soap
except:
    pass

from jsonargparse import ArgumentParser, set_docstring_parse_options
import time
import os

set_docstring_parse_options(attribute_docstrings=True)
torch.set_default_dtype(torch.float64)

bulk_points_key = lambda i: f"points_MPI_{i}"
indices_key = lambda i: f"indices_MPI_{i}"
distances_key = lambda i: f"distances_MPI_{i}"
displacements_key = lambda i: f"displacements_MPI_{i}"
elements_key = lambda i: f"elements_MPI_{i}"

default_device = "cuda" if torch.cuda.is_available() else "cpu"

def retrieve_point_fields(client, mpi_ranks, key_constructor):
    point_field_by_rank = {r: client.get_tensor(key_constructor(r)) for r in mpi_ranks}
    try:
        point_field = np.vstack(list(point_field_by_rank.values()))
    except:
        point_field = np.hstack(list(point_field_by_rank.values()))
    start = 0
    indices = {}
    for r, rank_points in point_field_by_rank.items():
        end = start + rank_points.shape[0]
        indices[r] = np.arange(start, end)
        start = end

    return point_field, indices

def main(args):

    mpi_ranks = range(args.mpi_ranks)
    dev = args.device if args.device else default_device

    tmopargs = args.tmop

    client = Client()

    log_db_address = os.getenv("LOG_DB")
    if log_db_address : 
        log_client = Client(address = log_db_address, cluster = False)
    else:
        print("Could not find Redis DB for logging")
        log_client = None

    # Pause until the OpenFOAM simulation has posted the boundary points
    elements_ready = client.poll_key("elements_MPI_0", 1, 60000)
    if not elements_ready:
        raise Exception("'elements_MPI_0' key not found. Simulation may have failed")
    dimension = int(client.get_tensor("solution_dim")[0])

    # Retrieve the boundary and bulk points
    points = client.get_tensor("points") ## bd points
    bulk_points, rank_indices = retrieve_point_fields(client, mpi_ranks, bulk_points_key) ## unique pts
    global_indices, _ = retrieve_point_fields(client, mpi_ranks, indices_key)
    distance_to_boundary, _ = retrieve_point_fields(client, mpi_ranks, distances_key)
    elements_padded, _ = retrieve_point_fields(client, mpi_ranks, elements_key)

    if len(bulk_points) != len(global_indices):
        raise RuntimeError("Size mismatch: number of bulk points does not match number of point indices.")

    nodes = np.empty((global_indices.max()+1, dimension))
    nodes[global_indices] = bulk_points # reorder so that global_indices are in fact indices of the tensor (nodes[i] = bulk_points[global_indices[i]])
    dist = np.empty(global_indices.max()+1)
    dist[global_indices] = distance_to_boundary.squeeze()

    # might not need.
    # for gpu it is better to use the padded elements
    elements = []
    for ep in elements_padded:
        elements.append(ep[ep >= 0].tolist())

    # make a copy of the original mesh
    points0 = nodes.copy()

    # set masks for boundary and interior points
    all_ids = torch.arange(nodes.shape[0])
    bd_mask = dist < 1e-6
    bd_ids = all_ids[bd_mask]
    int_ids = all_ids[~bd_mask]

    # initialise the mesh object
    tmesh = TMOPMesh(
                 boundary_points=torch.from_numpy(nodes[bd_ids]), 
                 interior_points=torch.from_numpy(nodes[int_ids]), 
                 boundary_ids=bd_ids,
                 interior_ids=int_ids,
                 elements=torch.from_numpy(elements_padded),
                 config=tmopargs,
                 log_client = log_client
        )
    print(f"{tmesh.n_pts} points: {tmesh.n_bd_pts} boundary, {tmesh.n_int_pts} interior.\n{tmesh.n_elements} elements")

    tmesh.to(dev)

    timestep = 1
    while True:
        print("\n"+"-"*10)
        print(f"TIMESTEP {timestep}")

        # Block until the data is ready
        data_ready = client.poll_key("data_ready", 1, 60000)
        if (not data_ready):
            raise RuntimeError("Data not found in SmartRedis; aborting training.")

        # Get boundary displacements and corresponding node ids
        displacements = client.get_tensor("displacements")
        disp_gids = client.get_tensor("displacements_gids")
        client.delete_tensor("data_ready")

        start = time.perf_counter()
        # update the boundaries
        newp = points0.copy()
        newp[disp_gids] += displacements
        with torch.no_grad():
            tmesh.bd_pts.copy_(torch.from_numpy(newp[bd_ids]).to(dev))
        if cfg.mode == "points0":
            with torch.no_grad():
                tmesh.int_pts.copy_(torch.from_numpy(newp[int_ids]).to(dev))

        optim = tmopargs.optim.optimiser(tmesh.parameters(), **tmopargs.optim.optimiser_kwargs);
        sched = tmopargs.optim.scheduler(optim, **tmopargs.optim.scheduler_kwargs);
        tmesh.optimise(optim,
                       sched,
                       tmopargs.optim
                       )

        # get the displacements as optimised_points - initial_points
        newdisp = tmesh.pts.cpu().detach().numpy() - points0

        # send displacements back together with the corresponding node label for OpenFOAM
        for r in mpi_ranks:
            displacements_rank = newdisp[global_indices[rank_indices[r]]]
            client.put_tensor(displacements_key(r), displacements_rank)
            indices_rank = global_indices[rank_indices[r]]
            client.put_tensor(indices_key(r), indices_rank)

        client.put_tensor("displacements_ready", np.array([0]))
        send_time = time.perf_counter() - start
        print(f"Solution sent in {send_time}s")
        # Increase CFD+ML iteration
        timestep += 1

if __name__ == "__main__":
    parser = ArgumentParser(description="TMOP optimisation for mesh motion")
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(Config, "cfg")

    args = parser.parse_args()
    cfg: Config = args.cfg

    main(cfg)
    exit()
