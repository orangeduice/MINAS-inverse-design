# -*- coding: utf-8 -*-
"""
Processes reflectance spectra from an H5 file to identify dips.

This script reads spectral data, applies a peak-finding algorithm to an
inverted version of the data to find "dips," and then generates binary masks
(0s for no dip, 1s for a dip) as the output. The parameters for defining a
dip (prominence, width, and threshold) are configurable.

The script includes a function to generate a dummy H5 file for demonstration
and testing purposes.
"""
# %%
import h5py
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import random

# %%
def process_spectra(h5_path, dataset_name, output_dataset_name, **kwargs):
    """
    Processes spectra from an H5 file to find dips and save binary masks.

    Args:
        h5_path (str): Path to the input H5 file.
        dataset_name (str): Name of the dataset containing the spectra.
        output_dataset_name (str): Name for the output dataset of binary masks.
        **kwargs: Keyword arguments passed directly to scipy.signal.find_peaks.
                  This is how you control the dip definition.
                  - prominence (float): Required prominence of dips.
                  - width (float): Required width of dips.
                  - rel_height (float): Used for calculating the dip width.
    """
    print("\nProcessing spectra...")
    try:
        with h5py.File(h5_path, 'a') as f:
            if dataset_name not in f:
                print(f"Error: Dataset '{dataset_name}' not found in '{h5_path}'.")
                return

            spectra = f[dataset_name][:]
            num_spectra, num_points = spectra.shape
            dip_masks = np.zeros_like(spectra, dtype=int)

            print(f"Found {num_spectra} spectra to process.")

            for i in range(num_spectra):
                spectrum = spectra[i, :]
                # We look for peaks in the inverted spectrum to find dips
                inverted_spectrum = -spectrum

                # find_peaks returns indices of the peaks (dips in our case)
                peaks, properties = find_peaks(inverted_spectrum, **kwargs)

                # Create the binary mask based on the dip widths
                widths = properties.get('widths', [])
                left_ips = properties.get('left_ips', [])
                right_ips = properties.get('right_ips', [])

                if len(widths) > 0:
                    for j in range(len(peaks)):
                        # Get the start and end of the dip based on the calculated width
                        start_index = int(np.floor(left_ips[j]))
                        end_index = int(np.ceil(right_ips[j]))
                        dip_masks[i, start_index:end_index] = 1

            # Save the results back into the H5 file
            if output_dataset_name in f:
                del f[output_dataset_name] # Overwrite if exists

            f.create_dataset(output_dataset_name, data=dip_masks)
            print(f"Processing complete. Dip masks saved as '{output_dataset_name}'.")

    except Exception as e:
        print(f"An error occurred: {e}")


def visualize_result(h5_path, spectrum_index=0):
    """
    Visualizes a single spectrum and its corresponding dip mask.

    Args:
        h5_path (str): Path to the processed H5 file.
        spectrum_index (int): The index of the spectrum to plot.
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            wavelengths = f['wavelengths'][:]
            spectrum = f['R'][spectrum_index, :]
            dip_mask = f['dip_masks'][spectrum_index, :]

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot the original spectrum
        ax1.plot(wavelengths, spectrum, label='Original Reflectance Spectrum', color='royalblue')
        ax1.set_xlabel('Wavelength (nm)', fontsize=12)
        ax1.set_ylabel('Reflectance', fontsize=12, color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        ax1.set_title(f'Spectrum {spectrum_index} and Detected Dips', fontsize=14)
        ax1.set_ylim([0,1])
        # Create a second y-axis for the dip mask
        ax2 = ax1.twinx()
        ax2.fill_between(wavelengths, 0, dip_mask, alpha=0.3, color='coral', label='Detected Dip Regions (Mask)')
        ax2.set_ylabel('Dip Mask', fontsize=12, color='coral')
        ax2.tick_params(axis='y', labelcolor='coral')
        ax2.set_ylim(0, 1.1)
        ax2.set_yticks([0, 1])

        fig.tight_layout()
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='lower right')
        plt.savefig("dataset_test.png")
        # plt.show()

    except Exception as e:
        print(f"Could not visualize the result. Error: {e}")

# %%

# --- Configuration ---
H5_FILE_PATH = 'keras/mounted_datasets/pisa_data.h5'

# Create the dummy data file for the example

# --- Dip Definition Parameters (CONTROL THESE VALUES) ---
# These keyword arguments are passed to scipy.signal.find_peaks.
# See the SciPy documentation for more details.
dip_params = {
    #"prominence": 0.25,  # Minimum vertical distance between the dip and its surroundings.
    "width": 0.01,         # Minimum width of the dip in data points.
    "height": 0.5    # Relative height at which the dip width is measured. 0.5 is a full-width half-maximum.
}

# --- Run Processing ---
process_spectra(
    h5_path=H5_FILE_PATH,
    dataset_name='R',
    output_dataset_name='dip_masks',
    **dip_params
)
# %%

# --- Visualize the output for the first spectrum ---
visualize_result(h5_path=H5_FILE_PATH, spectrum_index=random.randint(0, 10000))

# %%
