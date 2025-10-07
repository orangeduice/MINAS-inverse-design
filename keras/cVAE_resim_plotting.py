from helper import read_comsol_csv, plot_comsol

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

C = 299792458

def cm_to_thz(wavenumber_cm):
    return (wavenumber_cm * C * 100) / 1e12

def thz_to_cm(frequency_thz):
    return (frequency_thz * 1e12) / (C * 100)

w_TO_planer_strong = 796.6 # cm^-1
w_LO_planer_strong = 972.7 # cm^-1
epsilon_inf_planer_strong = 6.56

w_TO_axial_strong = 783.6 # cm^-1
w_LO_axial_strong = 967.7 # cm^-1
epsilon_inf_axial_strong = 6.78

w_TO_axial_weak = 838.0 # cm^-1
w_LO_axial_weak = 838.9 # cm^-1

def lim_beta_inf(transverse_mode_freq, longitudinal_mode_freq, epsilon_inf):
    return np.sqrt(transverse_mode_freq**2 + epsilon_inf*(longitudinal_mode_freq**2 - transverse_mode_freq**2)/(1 + epsilon_inf))

lim_beta_inf_planer_strong = lim_beta_inf(cm_to_thz(w_TO_planer_strong), cm_to_thz(w_LO_planer_strong), epsilon_inf_planer_strong)
lim_beta_inf_axial_strong = lim_beta_inf(cm_to_thz(w_TO_axial_strong), cm_to_thz(w_LO_axial_strong), epsilon_inf_axial_strong)

print("lim_beta_inf_planer_strong (THz):", lim_beta_inf_planer_strong)
print("lim_beta_inf_axial_strong (THz):", lim_beta_inf_axial_strong)




fig, axes = plt.subplots(1, 1, figsize=(12, 6))

# Run ID
# ID = "7-8-14"
# ID = "7-5-9"
# ID = "7-9-15"
# ID = "7-6-10"
ID = "25-9-58"

run_folder = Path.cwd() / "designs" / ID

# df = read_comsol_csv(run_folder / "micro_pillar_resim.csv")
df = read_comsol_csv(run_folder / (ID + "_micro_pillar_resim.csv"))

target = np.loadtxt(run_folder / (ID + "_target_spectra.csv"), delimiter=",")

axes.plot(target[:, 0], target[:, 1], label = "Target", linestyle='--', color='red')

plot_comsol(
    df,
    axes,
    ["Reflectance (1)"], # ["abs(emw.S11)^2 (1)"], #
    xdata_indx=4,
    resim=True
)

axes.axvline(
    x=lim_beta_inf_planer_strong,
    color='teal',
    label=r'$\lim_{\beta \to \infty}\omega_{(\mathrm{planer\ strong})}$',
    linestyle='dotted'
)
axes.axvline(
    x=lim_beta_inf_axial_strong,
    color='coral',
    label=r'$\lim_{\beta \to \infty}\omega_{(\mathrm{axial\ strong})}$',
    linestyle='dotted'
)


# axes.set_title("Re-simulation of test dataset sample")
axes.set_title("Re-simulation of suggested designs run " + ID)
axes.set_xlabel("Frequency (THz)")
axes.set_ylabel("Reflectance")
axes.legend(title="Params. $h_{pill}$, $sep$, $d_{pill}$, $w_{pill}$ all $\mu m$")
# plt.savefig(run_folder / "testdata_micro_pillar_resim.png", dpi=300)
plt.savefig(run_folder / (ID + "_micro_pillar_resim.png"), dpi=300)
plt.show()
