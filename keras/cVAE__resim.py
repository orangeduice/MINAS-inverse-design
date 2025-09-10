import json
import subprocess
import sys
import argparse
import os

import slurm_job_submitter


run_id = "1-1-1"

designs_folder = os.path.join("designs", str(run_id))

local_param_file = os.path.join(designs_folder, run_id+'_suggested_designs_physical.txt')

remote_path = "/Documents/" # Location on Iridis where files will be copied to

job_script = "run_re_simulation.sh"

slurm_job_submitter.run_submission(run_id, local_param_file, remote_path, job_script)


