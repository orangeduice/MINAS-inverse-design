
import argparse
import sys

def check_pytorch():
    """
    Checks for CUDA support and device information using PyTorch.
    """
    print("--- Checking with PyTorch ---")
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Please install it to run this check.")
        print("pip install torch torchvision torchaudio")
        return

    # Setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        gpu_id = torch.cuda.current_device()
        print(f'Device Name: {torch.cuda.get_device_name(gpu_id)}')
        print('Memory Usage:')
        # Convert bytes to Gigabytes (1024**3)
        allocated_mem = round(torch.cuda.memory_allocated(gpu_id) / (1024**3), 2)
        reserved_mem = round(torch.cuda.memory_reserved(gpu_id) / (1024**3), 2)
        print(f'Allocated: {allocated_mem} GB')
        print(f'Reserved:  {reserved_mem} GB')

def check_tensorflow():
    """
    Checks for CUDA support and device information using TensorFlow.
    """
    print("--- Checking with TensorFlow ---")
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow is not installed. Please install it to run this check.")
        print("pip install tensorflow")
        return

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # If GPUs are found, print info for the first one
        print(f'Using device: {gpus[0].name}')
        print()
        try:
            # tf.test.gpu_device_name() provides a more user-friendly name
            print(f'Device Name: {tf.test.gpu_device_name(gpus[0].name)}')
            
            # Note: TensorFlow's memory reporting is different from PyTorch's.
            # It often pre-allocates a large portion of memory.
            # get_memory_info gives current usage, not a "cached/reserved" value.
            mem_info = tf.config.experimental.get_memory_info('GPU:0')
            current_usage = round(mem_info['current'] / (1024**3), 2)
            print('Memory Usage:')
            print(f"Current Usage: {current_usage} GB")

        except Exception as e:
            print(f"Could not retrieve detailed GPU information: {e}")

    else:
        print('Using device: CPU')


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Check for CUDA support using either PyTorch or TensorFlow.',
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )
    
    # Add the 'framework' argument
    parser.add_argument(
        'framework',
        type=str,
        choices=['pytorch', 'tensorflow'],
        help='The framework to use for the CUDA check.\n'
             'Choices: "pytorch" or "tensorflow"'
    )

    # If no arguments are provided, print help and exit
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Call the appropriate function based on the user's choice
    if args.framework.lower() == 'pytorch':
        check_pytorch()
    elif args.framework.lower() == 'tensorflow':
        check_tensorflow()
