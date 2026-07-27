# %%
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# Assuming helper is in your working directory
import helper

# 1. Load Data
data_path = 'datasets/2026_04_15_full_pisa_data.h5'

x_train, x_test, y_train, y_test, wavelengths = helper.load_data_from_h5(
    data_path,
    condition_name='R',
    input_name='geo',
    save_scalers=False)

# Slicing the spectrum data to match CVAE
y_train = y_train[:, 35:110]
y_test = y_test[:, 35:110]

print('Geometry (X) shape:', x_train.shape)
print('Spectrum (Y) shape:', y_train.shape)

# %%
# 2. Build the Surrogate Architecture
# Maps Geometry (x) -> Spectrum (y)
def build_forward_surrogate(input_shape, output_shape):
    inputs = keras.Input(shape=input_shape, name="geometry_in")
    
    # Flatten the geometry for Dense layers
    x = keras.layers.Flatten()(inputs)
    
    # Powerful MLP to map parameters to spectra
    x = keras.layers.Dense(512, activation='leaky_relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(512, activation='leaky_relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(256, activation='leaky_relu')(x)
    
    # Output layer matches the spectrum length (75)
    outputs = keras.layers.Dense(output_shape[0], activation='linear', name="spectrum_out")(x)
    
    return keras.Model(inputs, outputs, name="forward_surrogate")

input_shape = x_train.shape[1:]
output_shape = y_train.shape[1:]

surrogate_model = build_forward_surrogate(input_shape, output_shape)
surrogate_model.summary()

# %%
# 3. Compile and Train
surrogate_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1),
    EarlyStopping(monitor='val_loss', patience=25, verbose=1, restore_best_weights=True)
]

print("\n--- Training Forward Surrogate ---")
history = surrogate_model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=256,
    epochs=200,
    callbacks=callbacks
)
# %%
# 4. Plot Training Results
print("\n--- Plotting Results ---")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss Curve
axes[0].plot(history.history['loss'], label='Train Loss', color='royalblue')
axes[0].plot(history.history['val_loss'], label='Val Loss', color='darkorange')
axes[0].set_title('Surrogate Training Loss (MSE)')
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].set_yscale('log')  # Log scale for better visibility of loss

# Sample Predictions
axes[1].set_title('True vs Predicted Spectra (Test Set)')
axes[1].set_xlabel('Wavelength Index')
axes[1].set_ylabel('Intensity')

# Pick 3 random test samples
random_indices = np.random.choice(len(x_test), 3, replace=False)
y_pred = surrogate_model.predict(x_test[random_indices])

colors = ['green', 'purple', 'red']
for i, (idx, color) in enumerate(zip(random_indices, colors)):
    axes[1].plot(y_test[idx], label=f'True {i+1}', linestyle='--', color=color)
    axes[1].plot(y_pred[i], label=f'Pred {i+1}', alpha=0.6, color=color)

axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('surrogate_training_results.png')
plt.show()

# %%

# 5. Save the model for the CVAE script to use
save_dir = "surrogate_model"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
surrogate_model.save(os.path.join(save_dir, "surrogate_geo_to_spec.keras"))
print("Surrogate model saved successfully!")
# %%
