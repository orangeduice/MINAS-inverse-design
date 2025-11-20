# %%

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
def setup_gpu():
    """Configures GPU memory for dynamic growth."""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    return sess 

def save_target(tracker, target_name, target_metadata):
    target_id = tracker.create_inference_target(
            target_name=target_name,
            target_metadata=target_metadata
        )

    print(f"Logged new inference target. Target ID: {target_id}")
    print(f"Data saved in: {tracker.targets_dir}/{target_id}.json")



def save_inference(tracker, train_id, target_id, inference_params, results_path, inference_output, target, wavelengths, condition_name='geo'):
    """Logs the inference run and saves results."""
    inference_id = tracker.create_inference_run(
        train_id=train_id,
        target_id=target_id,
        inference_params=inference_params,
        results_path=results_path
    )
    print(f"\nLogged new inference run. Inference ID: {inference_id}")
    print(f"Data saved in: {tracker.inference_dir}/{inference_id}.json")

    if condition_name == 'geo':

        designs_predict_physical = scaler_geo.inverse_transform(inference_output)

        sep = np.expand_dims(designs_predict_physical[:,3] - np.maximum(designs_predict_physical[:,1], designs_predict_physical[:,2]), axis=1)

        print(sep.shape)

        designs_predict_physical = np.hstack((np.expand_dims(designs_predict_physical[:,0], axis=1), sep, designs_predict_physical[:,[1,2,3]]))

        # np.savetxt(ID+'_suggested_designs.csv', designs_predict_physical.T, delimiter=',')
        print('h_pill (um)', 'sep (um)', 'd_pill (um)', 'w_pill (um)', 'Period Probe')
        print(designs_predict_physical)
        
        target = np.expand_dims(target, axis=1)
        wavelengths_cut = np.expand_dims(wavelengths[-64:], axis=1)
        target_spectra = np.hstack((wavelengths_cut, target))

        print("\n--- Retrieving Full Run Details ---")
        full_details = tracker.get_full_run_details(inference_id)

        if full_details:
            print("Successfully retrieved all linked information by reading files:")
            pprint.pprint(full_details)
        else:
            print(f"Could not find details for inference ID: {inference_id}")

        run_id = full_details["full_run_id"]

        print(f"Saving predicted designs with full run ID: {run_id}")

        run_id_striped = run_id.replace('-', '')

        designs_folder = os.path.join("designs", str(run_id))

        os.makedirs(designs_folder, exist_ok=True)
        print("\n--- Saving predicted designs ---")
        np.savetxt(os.path.join(designs_folder, run_id+'_suggested_designs.csv'), inference_output, delimiter=',')
        np.savetxt(os.path.join(designs_folder, run_id+'_suggested_designs_physical.csv'), designs_predict_physical.T, delimiter=',')
        np.savetxt(os.path.join(designs_folder, run_id+'_target_spectra.csv'), target_spectra , delimiter=',')
        np.savetxt(os.path.join(designs_folder, run_id_striped+'_params.csv'), designs_predict_physical[:, :4], delimiter=',', header='h_pill, sep, d_pill, w_pill', comments='') # used for re simulation

    else: 
        # save target
        for i, output in enumerate(inference_output):
            target_metadata = {
                    "description": "inference output {} from spectra cVAE, inference id {}".format(i, inference_id),
                    "condition_vector": output.tolist()
                }
            save_target(tracker, "spectra cVAE", target_metadata)




def make_infrerence(decoder, z_sample, condition_vector):
    """Generates designs from the decoder given latent samples and condition vectors."""
    condition_vector_tf = tf.expand_dims(tf.convert_to_tensor(condition_vector),1)
    condition_vector_tf = tf.repeat(tf.expand_dims(condition_vector_tf, axis=0), repeats=z_sample.shape[0], axis=0)

    decoder_in = [z_sample, condition_vector_tf]
    designs_predict = tf.squeeze(decoder.predict(decoder_in, verbose=0))
    return designs_predict



# %%

setup_gpu()

data_path = 'datasets/pisa_data.h5'

condition_name = 'dip_masks'
input_name = 'R_norm'

train_id = 2 # ID of the trained model to load

target_id = 3  # ID of the target to use




# %%

# load data and scalers
x_train, x_test, y_train, y_test, wavelengths = helper.load_data_from_h5(
    data_path,
    condition_name = condition_name,
    input_name = input_name,
    save_scalers=False
    )

if input_name == 'geo':
    scaler_geo = pickle.load(
        open('{}_scalers.pkl'.format(os.path.splitext(data_path)[0]), 'rb'))

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# %%

tracker = ExperimentTracker()

keras.backend.clear_session()

train_id = 2 # ID of the trained model to load

model_folder = "pre_models"

decoder_file = os.path.join(model_folder, str(train_id),"pisa_pre_cVAE_dec.h5")

# load trained decoder
decoder = keras.models.load_model(decoder_file)

training_info = tracker.get_training_run(train_id)

latent_dim = training_info["cvae_hyperparameters"]["latent_dim"]

print("loaded decoder from ", decoder_file)

# %%
target_info = tracker.get_inference_target(target_id)

print(target_info)

target_spectra = target_info["target_metadata"]["condition_vector"]


print(target_spectra)
# %%
# evaluate 3 positions in latent space
num_pred = 3

z_sample =  np.random.randn(num_pred, latent_dim)*0.5 

decoder_out = make_infrerence(decoder, z_sample, target_spectra)

if input_name == 'geo':
    decoder_out_physical = scaler_geo.inverse_transform(decoder_out)

    sep = np.expand_dims(decoder_out_physical[:,3] - np.maximum(decoder_out_physical[:,1], decoder_out_physical[:,2]), axis=1)

    decoder_out_physical = np.hstack((np.expand_dims(decoder_out_physical[:,0], axis=1), sep, decoder_out_physical[:,[1,2,3]]))

    print('h_pill (um)', 'sep (um)', 'd_pill (um)', 'w_pill (um)', 'Period Probe')
    print(decoder_out_physical)
else:
    decoder_out_physical = decoder_out


# %%
# save everything

inference_parameters = {
    "latent_sampling_details": "np.random.randn(num_pred, 2) ",
    "num_predictions": int(num_pred),
    "z_samples": z_sample.tolist(),
}

save_inference(
    tracker,
    train_id,
    target_id,
    inference_parameters,
    results_path="to be removed",
    inference_output=decoder_out,
    target=target_spectra,
    wavelengths=target_info["target_metadata"]["wavelengths"],
    condition_name="geo"
)

# %%
