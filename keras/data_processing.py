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
import helper

class DataProcessing:
    def __init__(self, h5_path):
        """
        Initialize the DataProcessing class.

        Args:
            h5_path (str): Path to the H5 file to process
        """
        self.h5_path = h5_path

    def _save_to_h5(self, processed_data, output_dataset_name):
        """
        Internal method to save processed data to the H5 file.

        Args:
            processed_data (np.ndarray): The processed data to save
            output_dataset_name (str): Name for the output dataset
        """
        try:
            with h5py.File(self.h5_path, 'a') as f:
                if output_dataset_name in f:
                    del f[output_dataset_name]  # Overwrite if exists
                f.create_dataset(output_dataset_name, data=processed_data)
                print(f"Processing complete. Data saved as '{output_dataset_name}'.")
        except Exception as e:
            print(f"Error saving to H5 file: {e}")

    def filter_by_parameter(self, new_h5_path, param_index, threshold, comparison='ge', geo_dataset_name='geo'):
        """
        Creates a new H5 file containing only data where a specific parameter meets a condition.
        
        Args:
            new_h5_path (str): Path for the new filtered H5 file.
            param_index (int): The column index in 'geo' corresponding to the parameter (e.g., h_pill).
            threshold (float): The value to filter against (e.g., 0.2).
            comparison (str): 'ge' (>=), 'gt' (>), 'le' (<=), 'lt' (<). Default is 'ge' (keep values >= threshold).
            geo_dataset_name (str): Name of the dataset containing parameters.
        """
        print(f"\nFiltering data based on parameter at index {param_index}...")
        
        try:
            with h5py.File(self.h5_path, 'r') as src, h5py.File(new_h5_path, 'w') as dst:
                
                # 1. Load the geometry/parameter data to determine which indices to keep
                if geo_dataset_name not in src:
                    print(f"Error: '{geo_dataset_name}' not found in source file.")
                    return

                geo_data = src[geo_dataset_name][:]
                total_samples = geo_data.shape[0]
                
                # Select the specific column (parameter)
                param_values = geo_data[:, param_index]

                # 2. Create mask based on condition
                if comparison == 'ge': # Greater than or equal (Keep >= 0.2)
                    mask = param_values >= threshold
                elif comparison == 'gt':
                    mask = param_values > threshold
                elif comparison == 'le':
                    mask = param_values <= threshold
                elif comparison == 'lt':
                    mask = param_values < threshold
                else:
                    print("Invalid comparison operator. Use 'ge', 'gt', 'le', or 'lt'.")
                    return

                indices_to_keep = np.where(mask)[0]
                print(f"Original samples: {total_samples}")
                print(f"Samples matching criteria: {len(indices_to_keep)}")

                # 3. Iterate through ALL datasets in source and copy/slice them
                for name, item in src.items():
                    if isinstance(item, h5py.Dataset):
                        data = item[:]
                        
                        # logic: If the first dimension matches the total number of samples,
                        # assume it's data that needs to be filtered (sliced).
                        # Otherwise (like 'wavelengths'), copy it as-is.
                        if data.shape[0] == total_samples and name != 'wavelengths':
                            print(f"  > Filtering and copying dataset: {name}")
                            dst.create_dataset(name, data=data[indices_to_keep])
                        else:
                            print(f"  > Copying static/metadata dataset: {name}")
                            dst.create_dataset(name, data=data)
            
            print(f"Successfully created filtered file: {new_h5_path}")

        except Exception as e:
            print(f"An error occurred during filtering: {e}")


    def normalise(self, dataset_name, output_dataset_name = 'R_norm', base_spectrum_path = '/datasets/SiC_bulk__25-29THz.csv', **kwargs):

        try:
            with h5py.File(self.h5_path, 'r') as f:
                data = f[dataset_name][:]

            # print(data.shape, type(data))

            bulk_spectrum = helper.read_comsol_csv(base_spectrum_path)
            normlised_data = data + np.array( 1 - bulk_spectrum["Reflectance (1)"]).reshape(1, -1)

            # for idx, i in enumerate(data):
            #     data[idx] = data[idx] + ( 1 - bulk_spectrum["Reflectance (1)"])
            # print(normlised_data.shape, type(normlised_data))

            # print(normlised_data[0])

            # Use the shared saving method
            self._save_to_h5(normlised_data, output_dataset_name)

        except Exception as e:
            print(f"An error occurred: {e}")

    def process_dips(self, dataset_name, output_dataset_name, **kwargs):
        """
        Process spectra to find dips and save binary masks.

        Args:
            dataset_name (str): Name of the dataset containing the spectra
            output_dataset_name (str): Name for the output dataset of binary masks
            **kwargs: Keyword arguments passed to scipy.signal.find_peaks
        """
        print("\nProcessing spectra for dips...")
        try:
            with h5py.File(self.h5_path, 'r') as f:
                if dataset_name not in f:
                    print(f"Error: Dataset '{dataset_name}' not found.")
                    return
                spectra = f[dataset_name][:]

            num_spectra, num_points = spectra.shape
            dip_masks = np.zeros_like(spectra, dtype=int)
            print(f"Found {num_spectra} spectra to process.")

            for i in range(num_spectra):
                spectrum = spectra[i, :]
                inverted_spectrum = 1 - spectrum
                peaks, properties = find_peaks(inverted_spectrum, **kwargs)

                widths = properties.get('widths', [])
                left_ips = properties.get('left_ips', [])
                right_ips = properties.get('right_ips', [])

                if len(widths) > 0:
                    for j in range(len(peaks)):
                        start_index = int(np.floor(left_ips[j]))
                        end_index = int(np.ceil(right_ips[j]))
                        dip_masks[i, start_index:end_index] = 1

            self._save_to_h5(dip_masks, output_dataset_name)

        except Exception as e:
            print(f"An error occurred: {e}")

    def visualize_result(self, to_plot = 'R', spectrum_index=0, save_path=None):
        """
        Visualize a single spectrum and its corresponding processed data.

        Args:
            spectrum_index (int): Index of the spectrum to plot
            save_path (str, optional): Path to save the plot
        """
        try:
            with h5py.File(self.h5_path, 'r') as f:
                wavelengths = f['wavelengths'][:]
                spectrum = f[to_plot][spectrum_index, :]
                dip_mask = f['dip_masks'][spectrum_index, :]
                geo = f['geo'][spectrum_index, :]

            fig, ax1 = plt.subplots(figsize=(12, 6))

            ax1.plot(wavelengths, spectrum, "o", label='Original Reflectance Spectrum', color='royalblue')
            ax1.set_xlabel('Wavelength (nm)', fontsize=12)
            ax1.set_ylabel(f'Reflectance ({to_plot})', fontsize=12, color='royalblue')
            ax1.tick_params(axis='y', labelcolor='royalblue')
            ax1.set_title(f'Spectrum {spectrum_index} and Detected Dips, geo vector: {geo}', fontsize=14)
            ax1.set_ylim([0, 1])

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

            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
            # plt.close()

        except Exception as e:
            print(f"Could not visualize the result. Error: {e}")

# Example usage:
# if __name__ == "__main__":
# %%
H5_FILE_PATH = 'datasets/pisa_data_filtered.h5'#'datasets/pisa_data.h5'

FILTERED_H5_PATH = 'datasets/pisa_data_filtered.h5'

# Initialize the processor
processor = DataProcessing(H5_FILE_PATH)


# processor.filter_by_parameter(
#     new_h5_path=FILTERED_H5_PATH,
#     param_index=0,       # Change this to the column index of h_pill
#     threshold=0.2,       # The cutoff value
#     comparison='ge'      # 'ge' = Greater or Equal. (Keeps >= 0.2, Discards < 0.2)
# )

# processor.normalise(
#     dataset_name='R',
#     output_dataset_name='R_norm',
#     base_spectrum_path='datasets/SiC_bulk__25-29THz.csv'
# )

# # Define dip parameters
# dip_params = {
#     'prominence': 0.05,
#     'width': 0.1,
#     'height': 0.5,
#     # 'prominence': 0.05,
#     # 'width': 0.1,
#     # 'height': 0.02
# }

# # Process the spectra
# processor.process_dips(
#     dataset_name='R_norm',
#     output_dataset_name='dip_masks',
#     **dip_params
# )
# %%
# Visualize results
processor.visualize_result(
    to_plot='R_norm',
    spectrum_index=random.randint(0, 10000),#9035,#9722,
    save_path="dataset_test.png"
)



# %%
