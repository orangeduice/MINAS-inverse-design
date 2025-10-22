import pickle
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import os
import matplotlib
import h5py



# date: 06/2023 @author: P. Wiecha
def load_reflection_spectra_data(path_h5, save_scalers=True, test_fraction=0.05):

    with h5py.File(path_h5) as f_read:
        R_spec = np.array(f_read['R'], dtype=np.float32)
        geo = np.array(f_read['geo'], dtype=np.float32)
        wavelengths = np.array(f_read['wavelengths'], dtype=np.float32)

    geo_mask = [True, False, True, True, True]

    geo = geo[:,geo_mask]

    # # clip R spectra to 64 points
    # if R_spec.shape[1] > 64:
    #     R_spec = R_spec[:, -64:]

    # if wavelengths.shape[0] > 64:
    #     wavelengths = wavelengths[-64:]

    # if necessary, add a channel dimension to the spectra (keras: channels last)
    if R_spec.shape[-1] != 1:
        R_spec = np.expand_dims(R_spec, -1)

    #  separately standardize permittivities and thicknesses
    scaler_geo = StandardScaler().fit(geo)

    # save the scalers using pickle
    if save_scalers:
        pickle.dump(scaler_geo,
                    open('{}_scalers.pkl'.format(os.path.splitext(path_h5)[0]), 'wb'))

    # apply scaler
    geo = scaler_geo.transform(geo)

    if geo.shape[-1] != 1:
        geo = np.expand_dims(geo, -1)

    # split into training and test datasets. Set random state for a reproducible splitting
    x_train, x_test, y_train, y_test = train_test_split(
        geo, R_spec, test_size=test_fraction, random_state=42)

    return x_train, x_test, y_train, y_test, wavelengths

def inverse_scale_geo(geo, scaler_geo):
    """mat and thick are predicted, normalized values. return their inverse transforms"""
    geo_physical = scaler_geo.inverse_transform(geo)

    return geo_physical


def read_comsol_csv(filename):
    """
    Reads a COMSOL-exported CSV file and returns a pandas DataFrame.
    Ignores metadata lines starting with '%', uses the last such line as the header.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Separate header and data lines
    header_lines = [line.strip() for line in lines if line.startswith('%')]
    data_lines = [line.strip() for line in lines if not line.startswith('%') and line.strip()]

    # Extract column names from the last '%' line
    column_line = header_lines[-1].lstrip('%').strip()
    column_names = [col.strip() for col in column_line.split(',')]

    # Join the data lines and load into pandas
    data_str = '\n'.join(data_lines)
    df = pd.read_csv(StringIO(data_str), names=column_names)

    return df

def sample_from_h5(h5_filepath: str, n_samples: int = 5, random_sample: bool = True):
    """
    Samples data points from an H5 file with a specific structure.

    The H5 file is expected to contain:
    - 'R': The y-data of shape (N, 81), where N is the total number of records.
    - 'wavelengths': The x-data of shape (81,).
    - 'geo': Miscellaneous data of shape (N, 5).

    Args:
        h5_filepath (str): The full path to the input H5 file.
        n_samples (int, optional): The number of data points to sample. Defaults to 5.
        random_sample (bool, optional): If True, samples are chosen randomly.
                                        If False, the first n_samples are taken.
                                        Defaults to True.

    Returns:
        dict: A dictionary containing the sampled data:
              {
                  "R": numpy.ndarray of shape (n_samples, 81),
                  "wavelengths": numpy.ndarray of shape (81,),
                  "geo": numpy.ndarray of shape (n_samples, 5),
                  "indices": numpy.ndarray of the indices that were sampled.
              }
        None: If the file or datasets cannot be found.
    """
    if not os.path.exists(h5_filepath):
        print(f"Error: File not found at {h5_filepath}")
        return None

    try:
        with h5py.File(h5_filepath, 'r') as f:
            # Check if datasets exist
            required_datasets = ['R', 'wavelengths', 'geo']
            for dset in required_datasets:
                if dset not in f:
                    print(f"Error: Dataset '{dset}' not found in the H5 file.")
                    return None

            # Get handles to the datasets
            r_dset = f['R']
            geo_dset = f['geo']

            total_records = r_dset.shape[0]

            # Validate n_samples
            if n_samples > total_records:
                print(f"Warning: Requested {n_samples} samples, but only {total_records} are available. "
                      f"Returning all {total_records} samples.")
                n_samples = total_records

            # --- 2. Generate indices for sampling ---
            if random_sample:
                # Choose n_samples unique random indices from the available range
                indices = np.random.choice(total_records, size=n_samples, replace=False)
                # Sorting indices is good practice for HDF5; it can lead to faster reads
                indices.sort()
            else:
                # Take the first n_samples sequentially
                indices = np.arange(n_samples)

            # --- 3. Read the sampled data using the indices ---
            # Fancy indexing in h5py is very efficient
            sampled_r = r_dset[indices, :]
            sampled_geo = geo_dset[indices, :]

            # Wavelengths are the x-axis, common to all samples
            wavelengths = f['wavelengths'][:]

            return {
                "R": sampled_r,
                "wavelengths": wavelengths,
                "geo": sampled_geo,
                "indices": indices
            }

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def sample_and_plot_spectra(x_csv_path='X.csv', y_csv_path='Y.csv', n_samples=5, seed=0):
    # Load data
    X = pd.read_csv(x_csv_path)
    Y = pd.read_csv(y_csv_path)

    # Extract frequency from column names
    num_freqs = len(Y.columns) // 2
    reflectance_cols = Y.columns[:num_freqs]
    transmission_cols = Y.columns[num_freqs:]
    frequencies = [float(c.split('_')[-1]) for c in reflectance_cols]

    # Sample indices
    np.random.seed(seed)
    indices = np.random.choice(len(X), size=n_samples, replace=False)

    # Plot
    for idx in indices:
        x_row = X.iloc[idx]
        y_row = Y.iloc[idx]

        reflectance = y_row[reflectance_cols].values.astype(float)
        transmission = y_row[transmission_cols].values.astype(float)

        plt.figure(figsize=(8, 4))
        plt.plot(frequencies, reflectance, label='Reflectance',ls='',marker='x')
        plt.plot(frequencies, transmission, label='Transmission',ls='',marker='x')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Value')
        plt.title(
            f"h={x_row[0]}μm, sep={x_row[1]}μm, d={x_row[2]}μm, w={x_row[3]}μm"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_comsol(
    df,
    axes,
    column_names,
    parameter_indx=0,
    xdata_indx=1,
    separate_figures=False,
    show=True,
    export_path=None,
    line_style = "-",
    resim = False,
):
    parameter_col = df.columns[parameter_indx]
    xdata_col = df.columns[xdata_indx]
    grouped = df.groupby(parameter_col)

    # if isinstance(column_names, str):
    #     column_names = [column_names]

    if isinstance(axes, matplotlib.axes.Axes):
        cmap = plt.get_cmap('terrain')
        print("single axis")
        for col_name in column_names:
            for i, (param_val, group) in enumerate(grouped):
                if not resim:
                    axes.plot(group[xdata_col], group[col_name], label=f'{param_val:.3f}', linestyle = line_style)
                else:
                    color = cmap(i / 8)


                    params_pred = np.array2string(
                        group.iloc[0,0:4].to_numpy(),
                        precision=2,
                        suppress_small=True,
                        separator=', ',
                        prefix='>>> ',
                        formatter={'float_kind': lambda x: f"{x:.3f}"}
                        ).replace('[', '').replace(']', '')
                    print(params_pred)
                    axes.plot(group[xdata_col], group[col_name], label=params_pred, linestyle = line_style, color=color)


        if not resim:
            axes.set_title(f"{column_names} vs {xdata_col}")
            axes.set_xlabel(xdata_col)
            axes.set_ylabel(column_names)
            axes.legend(title=parameter_col)
            axes.grid(True)

    else:
        print("axis num:", len(axes))
        for i, col_name in enumerate(column_names):
            ax = axes[i]
            for param_val, group in grouped:
                ax.plot(group[xdata_col], group[col_name], label=f'{param_val:.3f}', linestyle = line_style)
            ax.set_title(f"{col_name} vs {xdata_col}")
            ax.set_xlabel(xdata_col)
            ax.set_ylabel(col_name)
            ax.legend(title=parameter_col)
            ax.grid(True)



def plot_comsol_2d(
    df,
    ax,
    color_column,
    parameter_indx=0,
    xdata_indx=1,
    cmap="viridis",
    show=True,
    export_path=None,
    vmin=None,
    vmax=None
):
    """
    Plots a 2D colormap: parameter vs xdata, colored by the selected quantity.

    Parameters:
    - df: DataFrame returned by `read_comsol_csv`
    - color_column: str, column name to map to color (e.g., 'ED (rad)')
    - parameter_indx: int, index of the sweep parameter (default: 0)
    - xdata_indx: int, index of the x-axis variable (default: 1)
    - cmap: str, colormap name
    - show: bool, whether to call plt.show()
    - export_path: str or None. If given, saves the figure to this path
    - vmin, vmax: float or None. Color scale limits.
    """
    parameter_col = df.columns[parameter_indx]
    xdata_col = df.columns[xdata_indx]

    # Create pivot table: rows=parameter, columns=xdata, values=color_column
    pivot_table = df.pivot_table(index=parameter_col, columns=xdata_col, values=color_column)

    # Sorting to ensure axes are correct
    pivot_table = pivot_table.sort_index(ascending=True)
    pivot_table = pivot_table.sort_index(axis=1, ascending=True)

    X, Y = np.meshgrid(pivot_table.columns.values, pivot_table.index.values)
    Z = pivot_table.values

    # fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)

    fig = ax.get_figure()

    fig.colorbar(c, ax=ax, label=color_column)

    ax.set_xlabel(xdata_col)
    ax.set_ylabel(parameter_col)
    ax.set_title(f"{color_column} vs {xdata_col} and {parameter_col}")



    if export_path:
        fig.savefig(export_path, dpi=300)
        print(f"Saved plot to {export_path}")