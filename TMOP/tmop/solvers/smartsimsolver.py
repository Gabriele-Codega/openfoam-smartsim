import torch
import numpy as np
from smartredis import Client

import os
import time

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..config import MotionConfig
from ..mesh import Mesh

class SmartSimMotionSolver:
    def __init__(self,
                 conf: "MotionConfig"
                 ):
        self.bulk_points_key    = lambda i: f"points_MPI_{i}"
        self.indices_key        = lambda i: f"indices_MPI_{i}"
        self.distances_key      = lambda i: f"distances_MPI_{i}"
        self.displacements_key  = lambda i: f"displacements_MPI_{i}"
        self.elements_key       = lambda i: f"elements_MPI_{i}"

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = conf.device if conf.device else default_device
        self.mpi_ranks = range(conf.mpi_ranks)
        self.mode = conf.mode

        self.client = Client()

        log_db_address = os.getenv("LOG_DB")
        if log_db_address : 
            self.log_client = Client(address = log_db_address, cluster = False)
        else:
            print("Could not find Redis DB for logging")
            self.log_client = None

        self._setup_mesh()

        self.toptim = conf.tmop_optimiser(
            self.mesh, 
            config=conf.tmop, 
            log_client = self.log_client
        )

        self.toptim.to(self.device)

    def solve(self):
        timestep = 1
        while True:
            print("\n"+"-"*10)
            print(f"TIMESTEP {timestep}")

            # Block until the data is ready
            data_ready = self.client.poll_key("data_ready", 1, 60000)
            if (not data_ready):
                raise RuntimeError("Data not found in SmartRedis; aborting training.")

            # Get boundary displacements and corresponding node ids
            displacements = self.client.get_tensor("displacements")
            disp_gids = self.client.get_tensor("displacements_gids")
            self.client.delete_tensor("data_ready")

            start = time.perf_counter()
            # update the boundaries
            newp = self.points0.copy()
            newp[disp_gids] += displacements
            with torch.no_grad():
                self.mesh.bd_pts.copy_(torch.from_numpy(newp[self.bd_ids]).to(self.device))
            if self.mode == "points0":
                with torch.no_grad():
                    self.toptim.mesh.int_pts.copy_(torch.from_numpy(newp[self.int_ids]).to(self.device))

            self.toptim.optimise()

            # get the displacements as optimised_points - initial_points
            newdisp = self.mesh.pts.cpu().detach().numpy() - self.points0

            # send displacements back together with the corresponding node label for OpenFOAM
            for r in self.mpi_ranks:
                displacements_rank = newdisp[self.global_indices[self.rank_indices[r]]]
                self.client.put_tensor(self.displacements_key(r), displacements_rank)
                indices_rank = self.global_indices[self.rank_indices[r]]
                self.client.put_tensor(self.indices_key(r), indices_rank)

            self.client.put_tensor("displacements_ready", np.array([0]))
            send_time = time.perf_counter() - start
            print(f"Solution sent in {send_time}s")
            # Increase CFD+ML iteration
            timestep += 1

    def _retrieve_point_fields(self, key_constructor):
        point_field_by_rank = {r: self.client.get_tensor(key_constructor(r)) for r in self.mpi_ranks}
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

    def _setup_mesh(self):
        # Pause until the OpenFOAM simulation has posted the boundary points
        elements_ready = self.client.poll_key("elements_MPI_0", 1, 60000)
        if not elements_ready:
            raise Exception("'elements_MPI_0' key not found. Simulation may have failed")
        self.spacedim = int(self.client.get_tensor("solution_dim")[0])

        # Retrieve the boundary and bulk points
        bulk_points, self.rank_indices  = self._retrieve_point_fields(self.bulk_points_key) ## unique pts
        self.global_indices, _               = self._retrieve_point_fields(self.indices_key)
        distance_to_boundary, _         = self._retrieve_point_fields(self.distances_key)
        self.elements, _                = self._retrieve_point_fields(self.elements_key)

        if len(bulk_points) != len(self.global_indices):
            raise RuntimeError("Size mismatch: number of bulk points does not match number of point indices.")

        self.nodes = np.empty((self.global_indices.max()+1, self.spacedim))
        self.nodes[self.global_indices] = bulk_points # reorder so that global_indices are in fact indices of the tensor (nodes[i] = bulk_points[global_indices[i]])
        self.dist = np.empty(self.global_indices.max()+1)
        self.dist[self.global_indices] = distance_to_boundary.squeeze()

        # make a copy of the original mesh
        self.points0 = self.nodes.copy()

        # set masks for boundary and interior points
        all_ids = np.arange(self.nodes.shape[0])
        bd_mask = self.dist < 1e-6
        self.bd_ids = all_ids[bd_mask]
        self.int_ids = all_ids[~bd_mask]

        # initialise the mesh object
        self.mesh = Mesh(
                     boundary_points=torch.from_numpy(self.nodes[self.bd_ids]), 
                     interior_points=torch.from_numpy(self.nodes[self.int_ids]), 
                     boundary_ids=torch.from_numpy(self.bd_ids),
                     interior_ids=torch.from_numpy(self.int_ids),
                     elements=torch.from_numpy(self.elements),
            )
