import tensorflow as tf
import numpy as np
import os
import sys
import scipy.io as sio
import h5py

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.icann.load_model_utils import (
    load_training_data, 
    create_and_compile_model,
    stress_mse_loss,
    dummy_loss
)
from src.models.icann.icann_3d_asymmetry import piCANN

def load_model_with_fallback(model_path, n=[1, 1, 1, 1, 1], seed=1234):
    """
    Load model with fallback to manual weight loading if standard loading fails.
    """
    # Check if model_path is a directory or file
    if os.path.isdir(model_path):
        full_model_path = os.path.join(model_path, 'model.h5')
        weights_path = os.path.join(model_path, 'icann_model_final.weights.h5')
        
        # If weights file doesn't exist, look for best weights
        if not os.path.exists(weights_path):
            weights_path = os.path.join(model_path, 'icann_best.weights.h5')
    else:
        full_model_path = model_path if model_path.endswith('.h5') else None
        weights_path = model_path if model_path.endswith('.weights.h5') else None
    
    # Try loading the full model first
    if full_model_path and os.path.exists(full_model_path):
        try:
            print(f"Loading full model from {full_model_path}...")
            model = tf.keras.models.load_model(
                full_model_path,
                custom_objects={
                    'piCANN': piCANN,
                    'stress_mse_loss': stress_mse_loss,
                    'dummy_loss': dummy_loss
                }
            )
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading full model: {str(e)}")
            print("Will try loading weights instead.")
    
    # If full model loading failed or model file doesn't exist, try loading just the weights
    if not weights_path or not os.path.exists(weights_path):
        print(f"No weights file found at {weights_path}")
        return None
    
    # Create a fresh model with correct architecture
    print(f"Creating new model for loading weights...")
    model = create_and_compile_model(n, 1e-3, seed=seed)
    
    # Make a dummy forward pass to ensure all layers are built
    batch_size = 1
    time_steps = 1
    dummy_input = {
        'C11': tf.ones((batch_size, time_steps, 1)),
        'C22': tf.ones((batch_size, time_steps, 1)),
        'C33': tf.ones((batch_size, time_steps, 1)),
        'dt': tf.ones((batch_size, time_steps, 1)) * 0.01
    }
    model(dummy_input)
    print("Model built with dummy forward pass")
    
    # Try standard weight loading
    try:
        print(f"Loading weights from: {weights_path}")
        model.load_weights(weights_path)
        print("Weights loaded successfully using standard loading")
        return model
    except Exception as e:
        print(f"Error with standard weight loading: {str(e)}")
        print("Attempting layer-by-layer weight loading...")
        
        # Try to load saved weights and apply them manually
        try:
            saved_weights = {}
            with h5py.File(weights_path, 'r') as f:
                for key in f.keys():
                    # Skip non-weight keys
                    if key in ['optimizer_weights', 'model_config']:
                        continue
                    
                    # Extract weights and add to dictionary
                    saved_weights[key] = f[key][:]
            
            # Find matching weights and assign them
            matched_count = 0
            total_count = len(model.weights)
            
            for weight in model.weights:
                name = weight.name
                if name in saved_weights:
                    weight.assign(saved_weights[name])
                    matched_count += 1
                    print(f"Assigned weight: {name}")
            
            print(f"Completed manual loading: {matched_count}/{total_count} weights assigned")
            
            if matched_count > 0:
                return model
            else:
                print("No weights were successfully assigned")
                return None
                
        except Exception as e:
            print(f"Error during manual weight loading: {str(e)}")
            return None

def save_to_matlab(model_path, data_path, output_dir, variant_range=[0, 2], n=[1, 1, 1, 1, 1]):
    """
    Save model training data and predictions to MATLAB .mat files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model with fallback to manual loading if needed
    model = load_model_with_fallback(model_path, n=n)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Load the training data
    print(f"Loading training data from {data_path}...")
    train_inputs, train_outputs, x_train, y_train = load_training_data(
        data_path=data_path, variant_range=variant_range
    )
    
    # Generate model predictions
    print("Generating model predictions...")
    predictions = model.predict(train_inputs)
    
    # Save training inputs to .mat file
    print("Saving training inputs...")
    input_mat = {
        'C11': train_inputs['C11'],
        'C22': train_inputs['C22'],
        'C33': train_inputs['C33'],
        'dt': train_inputs['dt']
    }
    sio.savemat(os.path.join(output_dir, 'train_inputs.mat'), input_mat)
    
    # Save training outputs to .mat file
    print("Saving training outputs...")
    output_mat = {
        'S11': train_outputs[0],
        'S22': train_outputs[1],
        'S33': train_outputs[2],
        'S12': train_outputs[3],
        'S13': train_outputs[4],
        'S23': train_outputs[5]
    }
    sio.savemat(os.path.join(output_dir, 'train_outputs.mat'), output_mat)
    
    # Save model predictions to .mat file
    print("Saving model predictions...")
    pred_mat = {
        'S11_pred': predictions[0],
        'S22_pred': predictions[1],
        'S33_pred': predictions[2],
        'S12_pred': predictions[3],
        'S13_pred': predictions[4],
        'S23_pred': predictions[5]
    }
    sio.savemat(os.path.join(output_dir, 'predictions.mat'), pred_mat)
    
    # Save a combined file with all data
    print("Saving combined data file...")
    combined_data = {}
    combined_data.update(input_mat)
    # Add outputs with prefix to avoid name conflicts
    combined_data.update({'GT_' + k: v for k, v in output_mat.items()})
    # Add predictions (already have _pred suffix)
    combined_data.update(pred_mat)
    sio.savemat(os.path.join(output_dir, 'combined_data.mat'), combined_data)
    
    print(f"All data saved to {output_dir}")
    print("Files:")
    print(f"  - {os.path.join(output_dir, 'train_inputs.mat')}")
    print(f"  - {os.path.join(output_dir, 'train_outputs.mat')}")
    print(f"  - {os.path.join(output_dir, 'predictions.mat')}")
    print(f"  - {os.path.join(output_dir, 'combined_data.mat')}")

if __name__ == "__main__":
    # Direct parameter settings - modify these values as needed
    model_path = "outputs/models/icann_staged_refactored"
    data_path = "data/simple_paths"
    output_dir = "outputs/matlab_exports"
    variant_range = [0, 2]  # Load variants 0 through 2
    
    save_to_matlab(
        model_path=model_path, 
        data_path=data_path, 
        output_dir=output_dir,
        variant_range=variant_range
    )