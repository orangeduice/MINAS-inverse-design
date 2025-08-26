from pprint import pprint
from experiment_tracker import ExperimentTracker

tracker = ExperimentTracker(root_dir='cvae_file_experiments')

inference_id = 1  # ID of the inference target to load

print("\n--- Retrieving Full Run Details ---")
full_details = tracker.get_full_run_details(inference_id)

if full_details:
    print("Successfully retrieved all linked information by reading files:")
    pprint(full_details)
else:
    print(f"Could not find details for inference ID: {inference_id}")