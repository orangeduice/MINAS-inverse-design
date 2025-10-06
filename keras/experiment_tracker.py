import json
import datetime
import os
from pprint import pprint
import glob
import numpy as np

class ExperimentTracker:
    """
    A framework to track hyperparameters for model training and parameters
    for inference runs. It uses a directory and file system to store all metadata.
    """

    def __init__(self, root_dir='experiment_log'):
        """
        Initializes the tracker and sets up the root directory for storing experiments.
        It will create the necessary subdirectories if they don't exist.

        Args:
            root_dir (str): The path to the root directory for storing all runs.
        """
        self.root_dir = root_dir
        self.training_dir = os.path.join(self.root_dir, "training_runs")
        self.targets_dir = os.path.join(self.root_dir, "inference_targets")
        self.inference_dir = os.path.join(self.root_dir, "inference_runs")
        self._setup_directories()

    def _setup_directories(self):
        """Creates the necessary subdirectories for storing data."""
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.targets_dir, exist_ok=True)
        os.makedirs(self.inference_dir, exist_ok=True)

    def _get_next_id(self, directory: str) -> int:
        """
        Determines the next available ID by checking the files in a directory.
        It finds the highest existing ID and returns ID + 1.

        Args:
            directory (str): The directory to scan for existing IDs.

        Returns:
            int: The next available integer ID.
        """
        files = glob.glob(os.path.join(directory, "*.json"))
        if not files:
            return 1
        
        max_id = 0
        for f in files:
            try:
                # Extract the base name (e.g., '1.json') and get the number
                file_id = int(os.path.splitext(os.path.basename(f))[0])
                if file_id > max_id:
                    max_id = file_id
            except (ValueError, IndexError):
                # Ignore files that don't have a numeric name
                continue
                
        return max_id + 1

    def create_training_run(self, model_type: str, hyperparameters: dict, wavelengths: list, test_loss: list = None, val_loss: list = None) -> int:
        """
        Logs a new model training run by saving its metadata to a JSON file.

        Args:
            model_type (str): The type of model being trained (e.g., 'cVAE').
            hyperparameters (dict): A dictionary of all hyperparameters used for training.
            wavelengths (list): The wavelengths used in the training data.
            test_loss (list, optional): A list of the training loss per epoch. Defaults to None.
            val_loss (list, optional): A list of the validation loss per epoch. Defaults to None.

        Returns:
            int: The unique 'train_id' for this training run.
        """
        train_id = self._get_next_id(self.training_dir)
        timestamp = datetime.datetime.now().isoformat()
        
        run_data = {
            "train_id": train_id,
            "timestamp": timestamp,
            "model_type": model_type,
            "hyperparameters": hyperparameters,
            "wavelengths": wavelengths,
            "test_loss": test_loss,
            "val_loss": val_loss,
        }
        
        file_path = os.path.join(self.training_dir, f"{train_id}.json")
        with open(file_path, 'w') as f:
            json.dump(run_data, f, indent=4)
            
        return train_id

    def create_inference_target(self, target_name: str, target_metadata: dict) -> int:
        """
        Logs a new target for inference runs by saving its metadata to a JSON file.

        Args:
            target_name (str): A human-readable name for the target.
            target_metadata (dict): A dictionary of metadata about the target.

        Returns:
            int: The unique 'target_id' for this inference target.
        """
        target_id = self._get_next_id(self.targets_dir)
        timestamp = datetime.datetime.now().isoformat()

        target_data = {
            "target_id": target_id,
            "timestamp": timestamp,
            "target_name": target_name,
            "target_metadata": target_metadata
        }

        file_path = os.path.join(self.targets_dir, f"{target_id}.json")
        with open(file_path, 'w') as f:
            json.dump(target_data, f, indent=4)

        return target_id

    def create_inference_run(self, train_id: int, target_id: int, inference_params: dict, results_path: str) -> int:
        """
        Logs a new inference run, linking a trained model with an inference target.

        Args:
            train_id (int): The ID of the training run being used.
            target_id (int): The ID of the inference target.
            inference_params (dict): A dictionary of parameters specific to this inference run.
            results_path (str): The path to where the results are saved.

        Returns:
            int: The unique 'inference_id' for this run.
        """
        inference_id = self._get_next_id(self.inference_dir)
        timestamp = datetime.datetime.now().isoformat()

        inference_data = {
            "inference_id": inference_id,
            "train_id": train_id,
            "target_id": target_id,
            "timestamp": timestamp,
            "inference_params": inference_params,
            "results_path": results_path
        }
        
        file_path = os.path.join(self.inference_dir, f"{inference_id}.json")
        with open(file_path, 'w') as f:
            json.dump(inference_data, f, indent=4)
            
        return inference_id

    def get_training_run(self, train_id: int) -> dict:
        """
        Retrieves the metadata for a single training run.

        Args:
            train_id (int): The ID of the training run to look up.

        Returns:
            dict: A dictionary containing the training run's data, or None if not found.
        """
        file_path = os.path.join(self.training_dir, f"{train_id}.json")
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def get_inference_target(self, target_id: int) -> dict:
        """
        Retrieves the metadata for a single inference target.

        Args:
            target_id (int): The ID of the inference target to look up.

        Returns:
            dict: A dictionary containing the target's data, or None if not found.
        """
        file_path = os.path.join(self.targets_dir, f"{target_id}.json")
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def get_inference_run(self, inference_id: int) -> dict:
        """
        Retrieves the metadata for a single inference run.

        Args:
            inference_id (int): The ID of the inference run to look up.

        Returns:
            dict: A dictionary containing the inference run's data, or None if not found.
        """
        file_path = os.path.join(self.inference_dir, f"{inference_id}.json")
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def get_full_run_details(self, inference_id: int) -> dict:
        """
        Retrieves all combined information for a given inference run by reading the
        corresponding JSON files.

        Args:
            inference_id (int): The ID of the inference run to look up.

        Returns:
            dict: A dictionary containing all details from training, target, and inference.
                  Returns None if any of the required files are not found.
        """
        inference_data = self.get_inference_run(inference_id)
        if not inference_data:
            return None

        train_id = inference_data.get("train_id")
        target_id = inference_data.get("target_id")

        training_data = self.get_training_run(train_id)
        if not training_data:
            return None
            
        target_data = self.get_inference_target(target_id)
        if not target_data:
            return None

        # Assemble the final dictionary
        return {
            "full_run_id": f"{train_id}-{target_id}-{inference_id}",
            "inference_details": {
                "inference_id": inference_data["inference_id"],
                "timestamp": inference_data["timestamp"],
                "params": inference_data["inference_params"],
                "results_path": inference_data["results_path"]
            },
            "training_details": {
                "train_id": training_data["train_id"],
                "timestamp": training_data["timestamp"],
                "model_type": training_data["model_type"],
                "hyperparameters": training_data["hyperparameters"],
            },
            "target_details": {
                "target_id": target_data["target_id"],
                "timestamp": target_data["timestamp"],
                "name": target_data["target_name"],
                "metadata": target_data["target_metadata"]
            }
        }



# --- Example Usage ---
def main():
    """Demonstrates how to use the file-based ExperimentTracker."""
    print("Initializing File-Based Experiment Tracker...")
    # All data will be saved in a folder named 'experiment_log_test'
    tracker = ExperimentTracker(root_dir='experiment_log_test')

    # --- 1. Simulate a Training Run ---
    print("\n--- Simulating a Keras cVAE Training Run ---")
    cvae_hyperparams = {
        "latent_dim": 2,
        "encoder_layers": [128, 64],
        "decoder_layers": [64, 128],
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "dataset": "MNIST"
    }
    # Dummy data for demonstration
    dummy_wavelengths = np.linspace(400, 700, 100).tolist()
    dummy_test_loss = [0.5, 0.4, 0.3]
    dummy_val_loss = [0.55, 0.45, 0.35]

    train_id = tracker.create_training_run(
        model_type="cVAE",
        hyperparameters=cvae_hyperparams,
        wavelengths=dummy_wavelengths,
        test_loss=dummy_test_loss,
        val_loss=dummy_val_loss
    )
    print(f"Logged new training run. Train ID: {train_id}")
    print(f"Data saved in: {tracker.training_dir}/{train_id}.json")


    # --- 2. Simulate Defining an Inference Target ---
    print("\n--- Simulating an Inference Target ---")
    target_metadata = {
        "description": "Generating images of the digit '8'",
        "condition_vector": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # One-hot for '8'
        "num_samples": 100
    }
    target_id = tracker.create_inference_target(
        target_name="Generate Eights",
        target_metadata=target_metadata
    )
    print(f"Logged new inference target. Target ID: {target_id}")
    print(f"Data saved in: {tracker.targets_dir}/{target_id}.json")


    # --- 3. Simulate an Inference Run ---
    print("\n--- Simulating an Inference Run ---")
    inference_parameters = {
        "latent_sampling_method": "gaussian",
        "latent_sampling_stddev": 0.5,
        "temperature": 0.7
    }
    inference_id = tracker.create_inference_run(
        train_id=train_id,
        target_id=target_id,
        inference_params=inference_parameters,
        results_path=f"results/"
    )
    print(f"Logged new inference run. Inference ID: {inference_id}")
    print(f"Data saved in: {tracker.inference_dir}/{inference_id}.json")
    
    # --- 4. Retrieve and Display the Full Run Details ---
    print("\n--- Retrieving Full Run Details ---")
    full_details = tracker.get_full_run_details(inference_id)
    
    if full_details:
        print("Successfully retrieved all linked information by reading files:")
        pprint(full_details)
    else:
        print(f"Could not find details for inference ID: {inference_id}")

if __name__ == '__main__':
    main()