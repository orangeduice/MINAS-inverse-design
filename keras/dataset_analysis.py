from helper import read_comsol_csv, plot_comsol, sample_from_h5, plot_average_spectrum

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.axes import Axes
import umap

## -------------------
## Data Loading, Calculation, and Saving
## -------------------

def sample_from_h5(h5_filepath: str, n_samples: int = 5, random_sample: bool = True):
    """Samples data points from an H5 file with a specific structure."""
    if not os.path.exists(h5_filepath):
        print(f"Error: File not found at {h5_filepath}")
        return None
    try:
        with h5py.File(h5_filepath, 'r') as f:
            r_dset = f['R']
            total_records = r_dset.shape[0]
            if n_samples > total_records:
                n_samples = total_records
            if random_sample:
                indices = np.random.choice(total_records, size=n_samples, replace=False)
                indices.sort()
            else:
                indices = np.arange(n_samples)
            return {
                "R": r_dset[indices, :],
                "wavelengths": f['wavelengths'][:],
                "geo": f['geo'][indices, :],
                "indices": indices
            }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_average_spectrum(h5_filepath: str):
    """Calculates the average spectrum from all records in the H5 file."""
    print("Calculating average spectrum...")
    try:
        with h5py.File(h5_filepath, 'r') as f:
            return {
                'wavelengths': f['wavelengths'][:],
                'average_R': np.mean(f['R'][:], axis=0),
                'total_records': f['R'].shape[0]
            }
    except Exception as e:
        print(f"Could not calculate average spectrum. Error: {e}")
        return None

def save_sampled_data_to_csv(output_filepath: str, data_dict: dict):
    """Saves the sampled data (index, geo, R) to a headerless CSV file."""
    print(f"\nSaving sampled data to {output_filepath} using numpy.savetxt...")
    try:
        indices_col = data_dict['indices'].reshape(-1, 1)
        combined_data = np.hstack([indices_col, data_dict['geo'], data_dict['R']])
        np.savetxt(output_filepath, combined_data, delimiter=',', fmt='%.6f')
        print("Data saved successfully.")
    except Exception as e:
        print(f"Failed to save data. Error: {e}")

def save_average_spectrum_to_csv(output_filepath: str, avg_data_dict: dict):
    """Saves the average spectrum data to a two-column CSV with a header."""
    print(f"\nSaving average spectrum to {output_filepath}...")
    try:
        data_to_save = np.column_stack([
            avg_data_dict['wavelengths'],
            avg_data_dict['average_R']
        ])
        header = "wavelength,average_reflectance"
        np.savetxt(output_filepath, data_to_save, delimiter=',', header=header, comments='', fmt='%.6f')
        print("Average spectrum saved successfully.")
    except Exception as e:
        print(f"Failed to save average spectrum. Error: {e}")

## -------------------
## Plotting Functions
## -------------------

def plot_sampled_spectra(ax: Axes, data_dict: dict):
    """Plots the sampled 'R' spectra on a given Matplotlib Axes."""
    for i in range(data_dict['R'].shape[0]):
        ax.plot(
            data_dict['wavelengths'],
            data_dict['R'][i, :],
            label=f"Index {data_dict['indices'][i]}"
        )
    ax.set_title(f'Random Sample of {len(data_dict["indices"])} Spectra')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Reflectance (R)')
    ax.legend(title="Original Index", fontsize='small')
    ax.grid(True)

def plot_average_spectrum(ax: Axes, avg_data_dict: dict):
    """Plots the pre-calculated average spectrum on a given Matplotlib Axes."""
    ax.plot(
        avg_data_dict['wavelengths'],
        avg_data_dict['average_R'],
        color='crimson',
        linewidth=2
    )
    ax.set_title(f'Average of All {avg_data_dict["total_records"]} Spectra')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylim([0, 1])
    ax.grid(True)

def plot_umap_embedding(ax: Axes, h5_filepath: str, color_by_geo_index: int = 0):
    """
    Computes and plots a 2D UMAP embedding of the spectral data.

    Args:
        ax (Axes): The Matplotlib axes object to plot on.
        h5_filepath (str): The path to the input H5 file.
        color_by_geo_index (int): The index of the 'geo' column to use for coloring points.
    """
    print("Generating UMAP embedding... This may take a moment.")
    try:
        with h5py.File(h5_filepath, 'r') as f:
            spectra = f['R'][:]
            geo_data = f['geo'][:]

            spectra_max = np.min(spectra, axis=1, keepdims=True)


        reducer = umap.UMAP(random_state=42) #n_neighbors=15, min_dist=0.1, n_components=2, 
        embedding = reducer.fit_transform(geo_data)

        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=spectra_max,
            cmap='viridis',
            s=1
        )

        ax.set_title('UMAP Projection')
        ax.set_xlabel('UMAP Dimension 1')
        ax.set_ylabel('UMAP Dimension 2')
        ax.grid(True)

        fig = ax.get_figure()
        fig.colorbar(scatter, ax=ax, label=f'Min Reflectance (R)')

    except Exception as e:
        print(f"Could not plot UMAP embedding. Error: {e}")


# C = 299792458

# def cm_to_thz(wavenumber_cm):
#     return (wavenumber_cm * C * 100) / 1e12

# def thz_to_cm(frequency_thz):
#     return (frequency_thz * 1e12) / (C * 100)

# w_TO_planer_strong = 796.6 # cm^-1
# w_LO_planer_strong = 972.7 # cm^-1
# epsilon_inf_planer_strong = 6.56

# w_TO_axial_strong = 783.6 # cm^-1
# w_LO_axial_strong = 967.7 # cm^-1
# epsilon_inf_axial_strong = 6.78

# w_TO_axial_weak = 838.0 # cm^-1
# w_LO_axial_weak = 838.9 # cm^-1

# def lim_beta_inf(transverse_mode_freq, longitudinal_mode_freq, epsilon_inf):
#     return np.sqrt(transverse_mode_freq**2 + epsilon_inf*(longitudinal_mode_freq**2 - transverse_mode_freq**2)/(1 + epsilon_inf))

# lim_beta_inf_planer_strong = lim_beta_inf(cm_to_thz(w_TO_planer_strong), cm_to_thz(w_LO_planer_strong), epsilon_inf_planer_strong)
# lim_beta_inf_axial_strong = lim_beta_inf(cm_to_thz(w_TO_axial_strong), cm_to_thz(w_LO_axial_strong), epsilon_inf_axial_strong)

# print("lim_beta_inf_planer_strong (THz):", lim_beta_inf_planer_strong)
# print("lim_beta_inf_axial_strong (THz):", lim_beta_inf_axial_strong)




# fig, axes = plt.subplots(1, 1, figsize=(10, 6))




# axes.axvline(
#     x=lim_beta_inf_planer_strong,
#     color='teal',
#     label=r'$\lim_{\beta \to \infty}\omega_{(\mathrm{planer\ strong})}$',
#     linestyle='dotted'
# )
# axes.axvline(
#     x=lim_beta_inf_axial_strong,
#     color='coral',
#     label=r'$\lim_{\beta \to \infty}\omega_{(\mathrm{axial\ strong})}$',
#     linestyle='dotted'
# )

# num_to_sample = 1000

# file = Path.cwd() / "datasets" / "pisa_data.h5"

# sampled_data = sample_from_h5(file, n_samples=num_to_sample, random_sample=True)

# if sampled_data:
#     # 3. Print the shapes and some info to verify
#     print(f"Successfully sampled {num_to_sample} data points.")
#     print(f"Shape of sampled 'R' data: {sampled_data['R'].shape}")
#     print(f"Shape of 'wavelengths' data: {sampled_data['wavelengths'].shape}")
#     print(f"Shape of sampled 'geo' data: {sampled_data['geo'].shape}")
#     print(f"Sampled indices: {sampled_data['indices']}")
#     print("\nFirst row of associated 'geo' data:")
#     print(sampled_data['geo'][0])
#     print("-" * 30)

#     # 4. Plot the sampled 'R' data against 'wavelengths'
#     print("Generating plot...")

#     for i in range(sampled_data['R'].shape[0]):
#         # Plot each sampled spectrum
#         axes.plot(
#             sampled_data['wavelengths'],
#             sampled_data['R'][i, :],
#             label=f"Sample Index {sampled_data['indices'][i]}"
#         )

#     axes.set_title(f'Random Sample of {num_to_sample} Spectra', fontsize=16)
#     axes.set_xlabel('Wavelength (nm)', fontsize=12)
#     axes.set_ylabel('Reflectance (R)', fontsize=12)
#     axes.legend(title="Original Index", fontsize='small')
#     axes.grid(True)
#     plt.tight_layout()
#     plt.show()

#     print("-" * 30)

#     # --- Part 2: Plot the average of ALL spectra ---

# plot_average_spectrum(sampled_data)


# axes.set_xlabel("Frequency (THz)")
# axes.set_ylabel("Reflectance")
# axes.legend()
# plt.show()

if __name__ == "__main__":

    H5_FILENAME = Path.cwd() / "keras" / "datasets" / "pisa_data.h5"
    NUM_SAMPLES = 8

    

    sampled_data = sample_from_h5(H5_FILENAME, n_samples=NUM_SAMPLES)
    average_data = calculate_average_spectrum(H5_FILENAME)

    if sampled_data and average_data:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 7))
        fig.suptitle("Spectral Data Analysis", fontsize=20)

        plot_sampled_spectra(ax1, sampled_data)
        plot_average_spectrum(ax2, average_data)
        plot_umap_embedding(ax3, H5_FILENAME)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()