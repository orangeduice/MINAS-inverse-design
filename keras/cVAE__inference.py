import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import os
import pprint
import model_architecture as model_arch
import helper

from experiment_tracker import ExperimentTracker


# %%

# Dynamic GPU memory allocation: For local prototyping, don't block GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

data_path = 'datasets/pisa_data.h5'

# load data and scalers
x_train, x_test, y_train, y_test, wavelengths = helper.load_reflection_spectra_data(
    data_path, save_scalers=False)
scaler_geo = pickle.load(
    open('{}_scalers.pkl'.format(os.path.splitext(data_path)[0]), 'rb'))

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# %%

tracker = ExperimentTracker()

keras.backend.clear_session()

train_id = 1 # ID of the trained model to load

model_folder = "models"

decoder_file = os.path.join(model_folder, train_id,"pisa_cVAE_dec.h5")

# load trained decoder
decoder = keras.models.load_model(decoder_file)

print("loaded decoder from ", decoder_file)

# train_info = tracker.get_training_run(train_id)

# %%

target_id = 1  # ID of the target to use

target_info = tracker.get_inference_target(target_id)

print(target_info)

target_spectra = target_info["target_metadata"]["condition_vector"]

print(target_spectra)
# %%
# evaluate 9 positions in latent space
num_pred = 9
z_sample = np.array([[i, i] for i in np.linspace(-1, 1, num_pred)])

target_spectra_tf = tf.expand_dims(tf.convert_to_tensor(target_spectra),1)
target_spectra_tf = tf.repeat(tf.expand_dims(target_spectra_tf, axis=0), repeats=num_pred, axis=0)

print(target_spectra_tf.shape)


# %%

decoder_in = [z_sample, target_spectra_tf]
designs_predict = tf.squeeze(decoder.predict(decoder_in, verbose=0))

print(designs_predict)


# %%

designs_predict_physical = scaler_geo.inverse_transform(designs_predict)

sep = np.expand_dims(designs_predict_physical[:,3] - np.maximum(designs_predict_physical[:,1], designs_predict_physical[:,2]), axis=1)

print(sep.shape)

designs_predict_physical = np.hstack((np.expand_dims(designs_predict_physical[:,0], axis=1), sep, designs_predict_physical[:,[1,2,3]]))

# np.savetxt(ID+'_suggested_designs.csv', designs_predict_physical.T, delimiter=',')
print('h_pill (um)', 'sep (um)', 'd_pill (um)', 'w_pill (um)', 'Period Probe')
print(designs_predict_physical)


# %%

target = np.expand_dims(target_spectra, axis=1)
wavelengths_cut = np.expand_dims(wavelengths[-64:], axis=1)
target_spectra = np.hstack((wavelengths_cut, target))
# np.savetxt(ID+'_target_spectra.csv', target_spectra , delimiter=',')
# print(target_spectra)



# %%
# save everything

inference_parameters = {
    "latent_sampling_details": "np.linspace(-1, 1, num_pred)",
    "num_predictions": num_pred,
    "z_samples": z_sample.tolist(),
}
inference_id = tracker.create_inference_run(
    train_id=train_id,
    target_id=target_id,
    inference_params=inference_parameters,
    results_path=f"results/"
)

print(f"Logged new inference run. Inference ID: {inference_id}")
print(f"Data saved in: {tracker.inference_dir}/{inference_id}.json")

# %%
# retrieve full run details

print("\n--- Retrieving Full Run Details ---")
full_details = tracker.get_full_run_details(inference_id)

if full_details:
    print("Successfully retrieved all linked information by reading files:")
    pprint(full_details)
else:
    print(f"Could not find details for inference ID: {inference_id}")

run_id = full_details["full_run_id"]

print(f"Saving predicted designs with full run ID: {run_id}")

# target_spectra
# designs_predict
# designs_predict_physical.T

designs_folder = os.path.join("keras","designs", str(run_id))

np.savetxt(os.path.join(designs_folder, run_id+'_suggested_designs.csv'), designs_predict, delimiter=',')
np.savetxt(os.path.join(designs_folder, run_id+'_suggested_designs_physical.csv'), designs_predict_physical.T, delimiter=',')
np.savetxt(os.path.join(designs_folder, run_id+'_target_spectra.csv'), target_spectra , delimiter=',')

