import os
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from experiment_tracker import ExperimentTracker
import helper
import cvae_model
from custom_callbacks import KLAnnealingCallback # Import the new callback

def setup_gpu():
    """Configures GPU memory for dynamic growth."""
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    return sess

def plot_history(history, save_path=None):
    """Plots the training and validation loss."""
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    if 'beta' in history.history:
        plt.plot(history.history['beta'], label='beta')
    plt.yscale('log')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss / Beta')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def save_experiment(tracker, cvae, encoder, decoder, hyperparameters, wavelengths, history, model_folder="models"):
    """Logs the experiment and saves model files."""
    train_id = tracker.create_training_run(
        model_type="cVAE",
        hyperparameters=hyperparameters,
        wavelengths=wavelengths.tolist(),
        test_loss=history.history['loss'],
        val_loss=history.history['val_loss'],
    )

    run_folder = os.path.join(model_folder, str(train_id))
    os.makedirs(run_folder, exist_ok=True)

    full_model_file = os.path.join(run_folder, "pisa_cVAE_full.h5")
    encoder_file = os.path.join(run_folder, "pisa_cVAE_enc.h5")
    decoder_file = os.path.join(run_folder, "pisa_cVAE_dec.h5")

    cvae.save(full_model_file)
    encoder.save(encoder_file)
    decoder.save(decoder_file)

    print(f"\nLogged new training run. Train ID: {train_id}")
    print(f"Experiment data saved in: {tracker.training_dir}/{train_id}.json")
    print(f"Models saved in: {run_folder}")


def main():
    """Main function to run the CVAE training pipeline."""
    # --- Configuration ---
    CVAE_HYPERPARAMS = {
        "architecture_type": "mlp", # 'conv' or 'mlp'
        "latent_dim": 2,
        "n_blocks": 3,
        "dataset": "pisa_data"
    }

    TRAINING_PARAMS = {
        "learning_rate": 0.0002,
        "epochs": 100,
        "batch_size": 32,
        "reduce_lr_factor": 1/3,
        "reduce_lr_patience": 5,
        "early_stopping_patience": 10,
        "kl_annealing": {
            "start_beta": 1e-6,
            "end_beta": 0.001,      # The original beta value
            "annealing_epochs": 25, # Number of epochs to ramp up beta
            "shape": "sigmoid"       # 'linear' or 'sigmoid'
        }
    }

    DATA_PATH = 'datasets/pisa_data.h5'
    
    # --- Setup ---
    setup_gpu()
    keras.backend.clear_session()
    
    # --- Load Data ---
    print("Loading data...")
    x_train, _, y_train, _, wavelengths = helper.load_reflection_spectra_data(
        DATA_PATH, save_scalers=True
    )
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    # --- Build Models ---
    print("Building models...")
    design_shape = x_train.shape[1:]
    condition_shape = y_train.shape[1:]

    encoder = cvae_model.build_encoder(
        design_shape, condition_shape, CVAE_HYPERPARAMS['latent_dim'], CVAE_HYPERPARAMS['n_blocks'],
        architecture_type=CVAE_HYPERPARAMS['architecture_type']
    )
    decoder = cvae_model.build_decoder(
        CVAE_HYPERPARAMS['latent_dim'], condition_shape, design_shape, CVAE_HYPERPARAMS['n_blocks'],
        architecture_type=CVAE_HYPERPARAMS['architecture_type']
    )
    cvae = cvae_model.build_cvae(
        encoder, decoder, design_shape, condition_shape,
        initial_beta=TRAINING_PARAMS['kl_annealing']['start_beta']
    )
    encoder.summary()
    decoder.summary()
    cvae.summary()
    
    # --- Compile Model ---
    optimizer = keras.optimizers.AdamW(learning_rate=TRAINING_PARAMS['learning_rate'])
    cvae.compile(optimizer=optimizer)

    # --- Train Model ---
    print("Starting training...")
    kl_params = TRAINING_PARAMS['kl_annealing']
    callbacks = [
        KLAnnealingCallback(
            start_beta=kl_params['start_beta'],
            end_beta=kl_params['end_beta'],
            annealing_epochs=kl_params['annealing_epochs'],
            annealing_shape=kl_params['shape']
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=TRAINING_PARAMS['reduce_lr_factor'],
            patience=TRAINING_PARAMS['reduce_lr_patience'],
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=TRAINING_PARAMS['early_stopping_patience'],
            verbose=1,
            restore_best_weights=True
        )
    ]

    history = cvae.fit(
        x=[x_train, y_train],
        y=x_train,
        validation_split=.2,
        batch_size=TRAINING_PARAMS['batch_size'],
        epochs=TRAINING_PARAMS['epochs'],
        callbacks=callbacks
    )

    # --- Plot and Save ---
    plot_history(history, save_path="loss_curve_with_annealing.png")

    tracker = ExperimentTracker()
    all_hyperparams = {**CVAE_HYPERPARAMS, **TRAINING_PARAMS}
    
    save_experiment(tracker, cvae, encoder, decoder, all_hyperparams, wavelengths, history)


if __name__ == '__main__':
    main()