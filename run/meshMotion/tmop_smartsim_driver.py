#!/usr/bin/env python

import argparse
import os
import logging
import sys
import time

from pathlib import Path
from PyFoam.RunDictionary.ParsedParameterFile import ParsedParameterFile

from smartredis import Client
from smartsim import Experiment
from smartsim.status import TERMINAL_STATUSES


platform_config = {
    "local": {
        "launcher": "local",
        "interface": "lo",
        "run_command": "mpirun"
    },
    "hotlum": {
        "launcher": "slurm",
        "interface": "bond0",
        "run_command": "srun"
    },
    "vader": {
        "launcher": "slurm",
        "interface": "bond0",
        "run_command": "srun"
    },
}

def main(args):

    # ----------------------------------------------------------------
    # Create the SmartSim experiment
    # ----------------------------------------------------------------

    input_case_path = Path(args.case)
    case_name = input_case_path.stem
    experiment_name = f"{args.experiment}_{case_name}"

    exp = Experiment(experiment_name, launcher=platform_config[args.platform]["launcher"])

    # ----------------------------------------------------------------
    # Launch the database
    # ----------------------------------------------------------------

    db = exp.create_database(port=8000, interface=platform_config[args.platform]["interface"])
    exp.generate(db, overwrite=True)

    # create a database to store logs from TMOP
    log_db = exp.create_database(port=9000, interface=platform_config[args.platform]["interface"], db_identifier="log_db")
    exp.generate(log_db, overwrite=True)
    exp.start(log_db)
    log_client = Client(address = log_db.get_address()[0], cluster = False)
    # ----------------------------------------------------------------
    # Get the number of MPI ranks from system/decomposeParDict
    # ----------------------------------------------------------------

    # build the full path to the decomposeParDict
    decompose_dict_path = input_case_path / 'system' / 'decomposeParDict'

    # load the dictionary
    decompose = ParsedParameterFile(decompose_dict_path)

    # extract the numberOfSubdomains entry
    num_mpi_ranks = decompose['numberOfSubdomains']

    # ----------------------------------------------------------------
    # Configure and create the OpenFOAM mesh-motion model
    # ----------------------------------------------------------------

    # Create OpenFOAM moveDynamicMesh run settings
    openfoam_rs = exp.create_run_settings(
        exe="moveDynamicMesh",
        exe_args="-parallel",
        run_command=platform_config[args.platform]["run_command"]
    )
    openfoam_rs.set_tasks(num_mpi_ranks)
    openfoam_rs.set_nodes(1)

    # Create the model from the OpenFOAM case argument
    openfoam_model = exp.create_model(
        name=args.case,
        run_settings=openfoam_rs
    )
    openfoam_model.attach_generator_files(to_copy=str(input_case_path.absolute()))

    # ----------------------------------------------------------------
    # Configure and create the ML training model
    # ----------------------------------------------------------------

    tmop_rs = exp.create_run_settings(
        exe="tmopSmartsimMotion",
        exe_args=f"--config config.yaml --motion.mpi_ranks {num_mpi_ranks}",
        env_vars={"LOG_DB": log_db.get_address()[0]}
    )
    tmop_rs.set_tasks(1)
    # tmop_rs.set_nodes(1)
    # tmop_rs.set_cpus_per_task(8)

    tmop_motion = exp.create_model(
        name="tmop_motion",
        run_settings=tmop_rs
    )
    tmop_motion.attach_generator_files(
        to_copy=["tmopSmartsimMotion", "config.yaml"],
        to_symlink=["./tmop"]
    )

    exp.generate(tmop_motion, overwrite=True)

    # ----------------------------------------------------------------
    # Run the experiment
    # ----------------------------------------------------------------

    try:
        exp.start(db)
        print(f"Database started at: {db.get_address()}")
        print("Running the OpenFOAM case")
        exp.generate(openfoam_model, overwrite=True)
        exp.start(openfoam_model, tmop_motion, block=False)

        last = -1
        timestep = 1
        while True:
            if log_client.key_exists("tmop_epoch") and log_client.key_exists("tmop_string"):
                logstr = bytes(log_client.get_tensor("tmop_string")).decode('utf-8')
                epoch = log_client.get_tensor("tmop_epoch")[0]
                logstr = f'Timestep {timestep} - '+logstr
                if epoch < last:
                    timestep += 1
                print('\r\x1b[2K'+logstr,end='',flush=True)
                last = epoch
            time.sleep(0.1)
            if exp.get_status(openfoam_model)[0] in TERMINAL_STATUSES:
                exp.stop(tmop_motion)
                print()
                break
        print()

    except Exception as e:
        print("Caught an exception:", e)

    finally:
        exp.stop(db)
        exp.stop(log_db)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a SmartSim Machine-Learning mesh deformation experiment"
    )
    parser.add_argument(
        "--experiment", "-e",
        default="meshMotion",
        help="Name of the SmartSim experiment (e.g., mesh_deformation)"
    )
    parser.add_argument(
        "--case", "-c",
        required=True,
        help="Name of the OpenFOAM case folder (e.g., ellipsoid3D)"
    )
    parser.add_argument(
        "--platform",
        default="local",
        help="The platform on which this is being run"
    )
    args = parser.parse_args()
    main(args)
