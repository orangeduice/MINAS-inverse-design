# keras implementation

ADD CITATION TO NEWCOMERS GUIDE

## Files Included

### Main files:

- **`cVAE__training.py`**:

- **`cVAE__inference_target.py`**:

- **`cVAE__inference.py`**:

- **`cVAE__resim.py`**:

### Helper files:

- **`experiment_tracker.py`**:

- **`helper.py`**:

- **`model_architecture.py`**:

- **`slurm_job_submitter.py`**: interface with iridis super computer
    
- **`config.json`**: Where you store your remote server's hostname and your username.
    
- **`README.md`**: This instruction file.
    

# Process of running cVAE experiments

TODO



# Details on Supercomputer Job Submission Helper

This is a simple system to help you send files to a remote supercomputer and submit a job to the SLURM workload manager with a single command. It uses `rsync` for efficient file transfers and `ssh` to submit the job.

## 1. Prerequisites: Passwordless SSH

This script **requires** you to have passwordless SSH access to your supercomputer using SSH keys. This allows the script to connect without prompting you for a password. If you haven't done this, it's the most critical step.

### How to Set Up SSH Keys (Simplified)

1. **Check for Existing Keys:** Open a terminal on your local machine and run:
    
    ```
    ls -al ~/.ssh/id_*.pub
    ```
    
    If you see a file (like `id_rsa.pub` or `id_ed25519.pub`), you already have keys. You can skip to step 3.
    
2. **Generate New SSH Keys:** If you don't have keys, generate them. `ed25519` is a modern and secure choice.
    
    ```
    ssh-keygen -t ed25519 -C "your_email@example.com"
    ```
    
    Press Enter to accept the default file location and you can optionally add a passphrase for more security.
    
3. **Copy Your Public Key to the Supercomputer:** Use the `ssh-copy-id` command to automatically copy your public key to the server. Replace `username` and `your-supercomputer.university.edu` with your details.
    
    ```
    ssh-copy-id username@your-supercomputer.university.edu
    ```
    
    It will ask for your password one last time.
    
4. **Test It:** Try to SSH into the server again.
    
    ```
    ssh username@your-supercomputer.university.edu
    ```
    
    If it logs you in without asking for a password, you're all set!
    

## 2. Configuration

Open the `config.json` file and edit the two fields:

- `"hostname"`: The full address of your supercomputer's login node (e.g., `hpc-login.your.edu`).
    
- `"username"`: Your username for the supercomputer.
    

**Example `config.json`:**

```
{
  "hostname": "pegasus.uni.ac.uk",
  "username": "ab123"
}
```

## 3. How to Use the Script

Let's say you have a project in a local folder named `my_simulation` that contains a data file `input.dat` and your SLURM script `run_sim.slurm`.

```
my_simulation/
├── input.dat
└── run_sim.slurm
```

You want to copy this entire folder to your home directory on the supercomputer and then run the `run_sim.slurm` script.

Open your terminal, navigate to the directory **containing** `my_simulation`, and run the Python script with three arguments:

1. **Local Path:** The local folder you want to send (`my_simulation`).
    
2. **Remote Path:** The **absolute path** to the destination on the supercomputer (e.g., `/home/users/ab123/projects`).
    
3. **Job Script:** The name of your SLURM script (`run_sim.slurm`).
    

**Command:**

```
python3 slurm_job_submitter.py my_simulation /home/users/ab123/projects/my_simulation run_sim.slurm
```

### What Happens:

1. The script will use `rsync` to copy the `my_simulation` folder to `/home/users/ab123/projects/` on the server. If the destination folder doesn't exist, `rsync` will create it.
    
2. After the transfer completes, it will connect via `ssh` and execute `sbatch /home/users/ab123/projects/my_simulation/run_sim.slurm`.
    
3. The output from `sbatch` (like "Submitted batch job 123456") will be printed to your terminal.