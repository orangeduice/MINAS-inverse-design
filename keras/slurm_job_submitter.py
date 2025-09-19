import json
import subprocess
import sys
import argparse
import os

def load_config():
    """
    Loads connection configuration from config.json.
    Handles potential errors like the file not being found or being invalid.
    """
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        # Basic validation to ensure required keys exist
        if "hostname" not in config or "username" not in config:
            print("Error: 'hostname' and 'username' must be set in config.json")
            sys.exit(1)
        return config
    except FileNotFoundError:
        print("Error: config.json not found. Please create it based on the example.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Could not parse config.json. Please ensure it is valid JSON.")
        sys.exit(1)

def sync_files_rsync(config, local_path, remote_path):
    """
    Synchronizes a local file or directory to the remote server using rsync.
    rsync is efficient as it only transfers changed files.
    """
    username = config['username']
    hostname = config['hostname']
    
    # Construct the full remote destination path.
    destination = f"{username}@{hostname}:{remote_path}"
    
    print(f"--> Starting file transfer to {destination}...")
    
    # The rsync command
    command = [
        "rsync",
        "-a",
        "-e",
        "ssh",
        local_path,
        destination
    ]
    
    try:
        # We use subprocess.run to execute the command.
        # `check=True` will raise a CalledProcessError if rsync returns a non-zero exit code.
        subprocess.run(command, check=True)
        print("--> File transfer successful.")
        return True
    except FileNotFoundError:
        print("Error: 'rsync' command not found. Is rsync installed on your system?")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error during rsync: {e}")
        return False

def submit_slurm_job(run_id, config, remote_job_script_path):
    """
    Connects to the remote server via SSH and submits a job using sbatch.
    """
    username = config['username']
    hostname = config['hostname']
    
    print(f"--> Submitting job: {remote_job_script_path}...")
    
    # The SSH command to execute sbatch on the remote server.
    command = [
        "ssh",
        f"{username}@{hostname}",
        f"sbatch {remote_job_script_path}",
        run_id.replace('-', '')
    ]
    
    try:
        # We capture the output to display the job submission confirmation.
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print("--> Job submission successful.")
        # Print the output from sbatch, which usually contains the job ID.
        print(f"    Server response: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("Error: 'ssh' command not found. Is an SSH client installed and in your PATH?")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error submitting SLURM job: {e}")
        print(f"    Server error output: {e.stderr.strip()}")
        return False

def run_submission(run_id, local_path, remote_path, job_script):
    """
    Orchestrates the file transfer and job submission process.
    This function is designed to be imported and called from other scripts.

    Returns:
        bool: True if successful, False otherwise.
    """
    config = load_config()

    # 1. Sync the files/directory to the remote server
    if not sync_files_rsync(config, local_path, remote_path):
        print("\nAborting due to file transfer failure.")
        return False

    # 2. Construct the full path to the job script on the remote server and submit
    # We use os.path.join to correctly handle the path construction.
    remote_job_script_full_path = os.path.join(remote_path, job_script)

    if not submit_slurm_job(run_id, config, remote_job_script_full_path):
        print("\nJob submission failed.")
        return False

    print("\nProcess completed successfully!")
    return True

def main():
    """
    Main function to parse arguments when the script is run from the command line.
    """
    parser = argparse.ArgumentParser(
        description="A tool to transfer files to a remote cluster and submit a SLURM job.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "local_path",
        help="The local file or directory to transfer to the server."
    )
    parser.add_argument(
        "remote_path",
        help="The destination directory on the remote server.\nThis path should be absolute."
    )
    parser.add_argument(
        "job_script",
        help="The path to the SLURM job script to be executed.\nThis path should be relative to the destination directory on the remote server."
    )
    
    args = parser.parse_args()
    
    # Call the core logic function with the parsed arguments
    if not run_submission(args.local_path, args.remote_path, args.job_script):
        sys.exit(1)


if __name__ == "__main__":
    main()