# %%
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import norm
from scipy.stats import cauchy
from scipy.interpolate import interp1d
from pathlib import Path

from experiment_tracker import ExperimentTracker

# %%

def create_reflectance_target(frequencies, center_freq, width, depth, order=2):
    """
    Generates a design target with a super-Gaussian (flattop) dip.

    Args:
        frequencies (np.ndarray): The array of frequency points.
        center_freq (float): The center frequency for the dip.
        width (float): The full width at half maximum (FWHM) of the dip.
        depth (float): The maximum depth of the dip (e.g., 1.0 for a dip to 0).
        order (float): The order of the super-Gaussian. 2 is a standard
                       Gaussian. Higher values create a flatter bottom.

    Returns:
        np.ndarray: The target reflectance spectrum.
    """
    # The baseline reflectance is 1.
    baseline = np.ones_like(frequencies)

    # Gaussian function to create a flatter dip.
    # The relationship between FWHM and standard deviation (sigma) is
    # maintained from the standard Gaussian to keep the 'width' parameter intuitive.
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    
    # Calculate Gaussian dip.
    # The 'order' parameter controls the flatness of the dip's bottom.
    flattop_dip = depth * np.exp(-(np.abs((frequencies - center_freq) / (np.sqrt(2) * sigma)))**order)

    # Subtract the dip from the baseline to get the final target.
    target_reflectance = baseline - flattop_dip
    
    return target_reflectance



# %%
tracker = ExperimentTracker()

#training_run_id = 1

#training_info = tracker.get_training_run(training_run_id)

#freq = training_info['wavelengths']

# print(freq)

# target_spectra =  -0.05*np.linspace(0, 1, 64) + 1

# dip = cauchy.pdf(freq, 27, 0.1)
# dip = dip / 4

# target_spectra -= dip

data_path = Path.cwd() / "datasets" / "pisa_data_average_spectrum.csv"

target_path = Path.cwd() / "datasets" / "75-21-8-IR.jdx"

import jcamp

print(target_path)

jcamp_dict = jcamp.jcamp_read(str(target_path))

print(jcamp_dict)
plt.plot(jcamp_dict['x'], jcamp_dict['y'])
plt.title(target_path)
# plt.xlabel(jcamp_dict['xunits'])
# plt.ylabel(jcamp_dict['yunits'])
plt.show()

# %%
average_spectrum = np.loadtxt(data_path, delimiter=',')

freq = average_spectrum[:,0]

target_frequency_THz = 27.0  # The center frequency of the dip (in THz).
dip_width_THz = 0.25      # The full width at half maximum (FWHM) of the dip.
dip_depth = 0.8            # The depth of the dip (1.0 means it goes to zero).
# An order of 2 is a standard Gaussian (pointy).
# Higher even numbers (4, 6, 8, etc.) create a flatter bottom.
flatness_order = 2.0       # New parameter to control the shape.

target_spectra  = create_reflectance_target(
    np.array(freq), 
    target_frequency_THz, 
    dip_width_THz, 
    dip_depth,
    order=flatness_order
) * average_spectrum[:,1]




freq = freq[-64:]

target_spectra = target_spectra[-64:]



# %%

plt.plot(freq,target_spectra, marker='o', ls='')
plt.xlabel('Frequency (THz)')
plt.ylabel('Reflectivity')
plt.ylim([0,1])
plt.show()

# %%

# save target
target_metadata = {
        "description": "new target script based on avreage spectra",
        "condition_vector": target_spectra.tolist()
    }

target_id = tracker.create_inference_target(
        target_name="new target script based on avreage spectra, dip at 27THz",
        target_metadata=target_metadata
    )

print(f"Logged new inference target. Target ID: {target_id}")
print(f"Data saved in: {tracker.targets_dir}/{target_id}.json")



# %%
