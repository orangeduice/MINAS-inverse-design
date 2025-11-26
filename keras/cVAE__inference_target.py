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

# import jcamp

#print(target_path)

#jcamp_dict = jcamp.jcamp_readfile(str(target_path))



#freq_jdx = np.array(jcamp_dict['x']*0.0299792458)

#average_spectrum = np.loadtxt(data_path, delimiter=',')

#freq = average_spectrum[:,0]

#freq = freq[-64:]

#low_val = freq[0]
#high_val = freq[-1]

#print(low_val, high_val)

#mask = (freq_jdx >= low_val) & (freq_jdx <= high_val)

#freq_jdx = freq_jdx[mask]

#y_jdx = jcamp_dict['y'][mask]

#plt.plot(freq_jdx, (1 - y_jdx)*1)
#plt.show()

# # The desired number of points for the new, low-resolution array
# num_points = 64

# # 1. Create a set of original "x" coordinates (the indices)
# #original_indices = np.arange(len(freq_jdx))

# # 2. Create a new set of evenly spaced "x" coordinates for the new points
# new_indices = np.linspace(original_indices[0], original_indices[-1], num=num_points)

# # 3. Interpolate the original data at the new index locations
# low_res_array_x = np.interp(new_indices, original_indices, freq_jdx)
# low_res_array_y = np.interp(new_indices, original_indices, y_jdx)
# print(low_res_array_x .shape, low_res_array_y.shape)
# plt.plot(low_res_array_x, 1 - (3000*low_res_array_y))
# plt.ylabel(jcamp_dict["yunits"])
# plt.xlabel("THz")
# plt.show()

# # %%


# target_frequency_THz = 27.0  # The center frequency of the dip (in THz).
# dip_width_THz = 0.25      # The full width at half maximum (FWHM) of the dip.
# dip_depth = 0.8            # The depth of the dip (1.0 means it goes to zero).
# # An order of 2 is a standard Gaussian (pointy).
# # Higher even numbers (4, 6, 8, etc.) create a flatter bottom.
# flatness_order = 2.0       # New parameter to control the shape.

# # target_spectra  = create_reflectance_target(
# #     np.array(freq),
# #     target_frequency_THz,
# #     dip_width_THz,
# #     dip_depth,
# #     order=flatness_order
# # ) * average_spectrum[:,1]

# target_spectra = 1 - (3000*low_res_array_y)


# freq = freq[-64:]

# # target_spectra = target_spectra[-64:]



# %%

# plt.plot(freq,target_spectra, marker='o', ls='')
plt.xlabel('Frequency (THz)')
plt.ylabel('Reflectivity')
plt.ylim([0,1])
plt.show()
# %%

def create_mask(size, start_w, end_w, fill_value=0, dtype=int):
    """
    Creates a 1D NumPy array of 'ones' with a randomly positioned segment
    of 'fill_value' (default 0). The start and end points of the segment
    are chosen randomly.

    Args:
        size (int): The total length of the output array.
        fill_value (int/float): The value to use for the masked segment.
        dtype (type): The data type of the array elements.

    Returns:
        np.ndarray: The resulting array.
    """
    def indx_convert(w, w_i = 25, w_f = 29, size=81):
        """Convert frequency to index in the array."""
        print((w - w_i)*(size/(w_f-w_i)))
        return int((w - w_i)*(size/(w_f-w_i)))

    # 2. Determine the start and end of the slice
    start_index = indx_convert(start_w)
    end_index = indx_convert(end_w) # This is the exclusive end index

    # 3. Create and set the array
    arr = np.zeros(size, dtype=dtype)
    arr[start_index:end_index] = fill_value

    return arr

# %%

target_spectra = create_mask(81, 26.95, 27.05, fill_value=1.0, dtype=float)

print(target_spectra)
# %%

# save target
target_metadata = {
        "description": "new target create_mask(81, 26.85, 27.15) for spectra cVAE",
        "condition_vector": target_spectra.tolist()
    }

target_id = tracker.create_inference_target(
        target_name="spectra cVAE dip at 27THz",
        target_metadata=target_metadata
    )

print(f"Logged new inference target. Target ID: {target_id}")
print(f"Data saved in: {tracker.targets_dir}/{target_id}.json")



# %%
