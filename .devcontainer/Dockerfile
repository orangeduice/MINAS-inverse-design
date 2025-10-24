# Start from the official TensorFlow 2.15.0 GPU-enabled image.
# This image includes Python 3.11, TensorFlow, and all CUDA/cuDNN dependencies.
FROM tensorflow/tensorflow:2.15.0-gpu

# Set the working directory
WORKDIR /app

RUN apt-get update && \
    apt-get install -y git

# TensorFlow 2.15.0 is already installed in the base image,
# so we don't need to run 'pip install' for it.
# You can add other 'pip install' commands here for your project's dependencies, e.g.:
RUN pip install --no-cache-dir pandas scikit-learn numpy matplotlib tqdm h5py ipykernel umap-learn

# You can copy your Python scripts into the container
# For example, if you have a script named 'app.py':
# COPY app.py .

# Set the default command to start a bash shell
CMD ["bash"]