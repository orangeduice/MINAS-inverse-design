import json
import subprocess
import sys
import argparse
import os

import slurm_job_submitter


run_id = "23-15-55"

designs_folder = os.path.join("designs", str(run_id))

local_param_file = os.path.join(designs_folder, run_id.replace('-', '')+'_params.csv')

remote_path = "/home/okcj1g19/" # Location on Iridis where files will be copied to

job_script = "js_micropillar_resim.sh"

slurm_job_submitter.run_submission(run_id, local_param_file, remote_path, job_script)


