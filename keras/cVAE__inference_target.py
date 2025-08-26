import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.interpolate import interp1d

from experiment_tracker import ExperimentTracker


tracker = ExperimentTracker()

training_run_id = 1

training_info = tracker.get_training_run(training_run_id)

freq = training_info['wavelengths']

print(freq)

target_spectra =  -0.3*np.linspace(0, 1, 64) + 0.95

dip = cauchy.pdf(freq, 28, 0.2)
dip = dip / 2.5

target_spectra -= dip


plt.plot(freq,target_spectra, marker='o', ls='')
plt.xlabel('Frequency (THz)')
plt.ylabel('Reflectivity')
plt.ylim([0,1])
plt.show()

# save target
target_metadata = {
        "description": "np.linspace latent space sampling",
        "condition_vector": target_spectra
    }

target_id = tracker.create_inference_target(
        target_name="gaussian dip at 28THz",
        target_metadata=target_metadata
    )

print(f"Logged new inference target. Target ID: {target_id}")
print(f"Data saved in: {tracker.targets_dir}/{target_id}.json")


