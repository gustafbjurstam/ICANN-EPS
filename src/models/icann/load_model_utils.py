import tensorflow as tf
from tensorflow import keras
import os
import sys
import json
import copy
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LinearSegmentedColormap

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import the model creation function and related classes
from src.models.icann.icann_3d_asymmetry import (
    create_rnn_model, piCANN, Invariants, ConvexPolynomialLayer, gNet,
    HelmholtzIsoNet, HelmholtzVolNet, HelmholtzNet
)

# Centralized loss functions
def stress_mse_loss(y_true, y_pred):
    """Custom MSE loss for stress components."""
    return tf.reduce_mean(tf.square(y_true - y_pred) )#/ (tf.pow(tf.abs(y_true),1) + 5e-1))

@tf.function
def dummy_loss(y_true, y_pred):
    """Dummy loss function that always returns 0."""
    return tf.zeros(tf.shape(y_pred)[0:1], dtype=y_pred.dtype)

def create_and_compile_model(n, learning_rate=2e-4, seed=2):
    """
    Create a new iCANN model and compile it with appropriate loss functions.
    
    Args:
        n: Network sizes [n_e, n_i, n_g1, n_g2, n_g3]
        learning_rate: Learning rate for the optimizer
    
    Returns:
        Compiled model ready for training
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print(f"Creating new model with network sizes: {n}")
    model = create_rnn_model(n)
    
    # Compile with standard loss configuration
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=[stress_mse_loss, stress_mse_loss, stress_mse_loss, 
                dummy_loss, dummy_loss, dummy_loss],
        loss_weights=[5.0, 3.0, 0.0, 0.0, 0.0, 0.0]
    )
    
    return model

def create_and_load_model(model_path, n=None):
    """
    Creates a fresh model with the same architecture and loads the weights.
    This bypasses the deserialization issues with custom layers.
    
    Args:
        model_path: Path to the saved model
        n: Network sizes [n_e, n_i, n_g1, n_g2, n_g3], defaults to [1, 1, 1, 1, 1]
    
    Returns:
        A new model with weights loaded from the saved model
    """
    if n is None:
        # Default network sizes
        n = [1, 1, 1, 1, 1]
    
    # Create a fresh model with the same architecture
    print(f"Creating new model with network sizes: {n}")
    new_model = create_rnn_model(n)
    
    # Enable unsafe deserialization for Lambda layers
    keras.config.enable_unsafe_deserialization()
    
    try:
        # Load just the weights from the saved model
        print(f"Loading weights from: {model_path}")
        new_model.load_weights(model_path)
        print("Weights loaded successfully!")
        return new_model
    except Exception as e:
        print(f"Error loading weights: {e}")
        
        # Alternative approach - try to extract weights from h5 file directly
        try:
            print("Trying alternative loading approach...")
            import h5py
            
            with h5py.File(model_path, 'r') as f:
                # Print the weight structure to help debug
                print("Weights in file:")
                for key in f.keys():
                    print(f"  {key}")
            
            # If we reached here, the file was readable but we couldn't load weights
            print("H5 file is readable but weights couldn't be loaded.")
            return None
        except Exception as e2:
            print(f"Error with alternative loading: {e2}")
            return None
        
def create_and_load_model_from_weights(model_path, n=None):
    """
    Creates a fresh model and loads just the weights.
    This avoids serialization issues with custom layers.
    
    Args:
        model_path: Path to the model directory or weights file
        n: Network sizes [n_e, n_i, n_g1, n_g2, n_g3], defaults to [1, 1, 1, 1, 1]
    
    Returns:
        A new model with weights loaded
    """
    if n is None:
        # Default network sizes
        n = [1, 1, 1, 1, 1]
        
        # Try to load from params file if available
        try:
            # If model_path is a directory, look for params file directly
            if os.path.isdir(model_path):
                params_path = os.path.join(model_path, 'model_params.json')
            else:
                # Otherwise, look in the parent directory
                params_path = os.path.join(os.path.dirname(model_path), 'model_params.json')
                
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params = json.load(f)
                    if 'n' in params:
                        n = params['n']
                        print(f"Loaded network sizes from params file: {n}")
        except Exception as e:
            print(f"Could not load params file: {e}")
    
    # Create a fresh model with the specified architecture
    model = create_and_compile_model(n)
    
    # Determine the weights file path
    weights_path = None
    
    # Case 1: model_path is a directory
    if os.path.isdir(model_path):
        potential_paths = [
            os.path.join(model_path, 'icann_model_final.weights.h5'),
            os.path.join(model_path, 'icann_best.weights.h5'),
            # Add backward compatibility for old naming
            os.path.join(model_path, 'icann_model_final_weights.h5'),
            os.path.join(model_path, 'icann_best_weights.h5')
        ]
        for path in potential_paths:
            if os.path.exists(path):
                weights_path = path
                break
    # Case 2: model_path is a .keras file
    elif model_path.endswith('.keras'):
        weights_path = model_path.replace('.keras', '.weights.h5')
        if not os.path.exists(weights_path):
            # Try older naming convention
            weights_path = model_path.replace('.keras', '_weights.h5')
        if not os.path.exists(weights_path):
            # Try in the same directory with standard name
            weights_path = os.path.join(os.path.dirname(model_path), 'icann_model_final.weights.h5')
    # Case 3: model_path is already a weights file
    elif model_path.endswith('.weights.h5') or model_path.endswith('_weights.h5'):
        weights_path = model_path
    
    # Final check if weights file exists
    if weights_path is None or not os.path.exists(weights_path):
        print(f"WARNING: Could not find a valid weights file for {model_path}!")
        # Look for any .h5 file in the directory
        if os.path.isdir(model_path):
            h5_files = [f for f in os.listdir(model_path) if f.endswith('.h5')]
            if h5_files:
                weights_path = os.path.join(model_path, h5_files[0])
                print(f"Trying to use {weights_path} as a fallback...")
            else:
                return model  # Return uninitialized model
        else:
            return model  # Return uninitialized model
    
    # Load the weights
    try:
        print(f"Loading weights from: {weights_path}")
        model.load_weights(weights_path)
        print("✓ Weights loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading weights: {e}")
        
        # Try custom layer-by-layer loading as a fallback
        try:
            print("Attempting layer-by-layer weight loading...")
            import h5py
            
            with h5py.File(weights_path, 'r') as f:
                # Load weights for each layer manually
                for layer in model.layers:
                    if layer.name in f and hasattr(layer, 'set_weights'):
                        weights = []
                        for weight_name in f[layer.name]:
                            weights.append(np.array(f[layer.name][weight_name]))
                        try:
                            layer.set_weights(weights)
                            print(f"✓ Loaded weights for layer: {layer.name}")
                        except Exception as e2:
                            print(f"Failed to load weights for layer {layer.name}: {e2}")
            
            print("Completed layer-by-layer loading attempt")
            return model
            
        except Exception as e2:
            print(f"Layer-by-layer loading failed: {e2}")
            return model  # Return uninitialized model

def extract_model_parameters(model):
    """Extract the network parameters from the model's iCANN layer."""
    # Find the iCANN layer in the model
    for layer in model.layers:
        if isinstance(layer, keras.layers.RNN):
            # Get the cell which should be an iCANN instance
            icann_cell = layer.cell
            if hasattr(icann_cell, 'n'):
                return {
                    'n': icann_cell.n,
                    'state_size': icann_cell.state_size
                }
    
    # Fallback if no iCANN layer found
    return {'n': [1, 1, 1, 1, 1]}

def save_model(model, save_path, history=None, learning_rate=None):
    """
    Save model weights and parameters in a standardized way.
    
    Args:
        model: The trained model to save
        save_path: Directory to save the model in
        history: Training history object (optional)
        learning_rate: Learning rate used for training (optional)
    """
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save weights with correct extension (.weights.h5)
    weights_path = os.path.join(save_path, 'icann_model_final.weights.h5')
    model.save_weights(weights_path)
    print(f"Model weights saved to {weights_path}")
    
    # Save best model weights separately if available
    best_model_path = os.path.join(save_path, 'icann_best.keras') 
    if os.path.exists(best_model_path):
        best_weights_path = os.path.join(save_path, 'icann_best.weights.h5')
        try:
            # Try to get the best model's weights
            best_model = create_and_load_model_from_weights(best_model_path)
            if best_model is not None:
                best_model.save_weights(best_weights_path)
                print(f"Best model weights saved to {best_weights_path}")
        except Exception as e:
            print(f"Could not save best weights separately: {e}")
    
    # Extract and save model parameters
    model_params = extract_model_parameters(model)
    
    # Add training metadata if available
    if history is not None:
        model_params['epochs_trained'] = len(history.history['loss'])
        model_params['final_loss'] = float(history.history['loss'][-1])
    
    if learning_rate is not None:
        model_params['learning_rate'] = learning_rate
        
    # Save parameters as JSON
    params_path = os.path.join(save_path, 'model_params.json')
    with open(params_path, 'w') as f:
        json.dump(model_params, f)

    # ----- NEW: Save weights in human-readable formats -----
    
    # 1. Create plain text hierarchical tree with full parameter values
    txt_path = os.path.join(save_path, 'weights_tree.txt')
    save_weights_as_text(model, txt_path)
    print(f"Weight tree saved as text to {txt_path}")
    
    # 2. Save individual weight arrays as .npy files
    npy_dir = save_weights_as_npy(model, save_path)
    print(f"Individual weight arrays saved to {npy_dir}")
    
    # # 3. Still create the CSV table for easy spreadsheet analysis
    # csv_path = os.path.join(save_path, 'weights_table.csv')
    # save_weights_as_csv(model, csv_path)
    # print(f"Weight table saved as CSV to {csv_path}")
    
    return weights_path, params_path

def save_weights_as_text(model, filepath):
    """
    Save model weights as a hierarchical text tree including actual parameter values.
    
    Args:
        model: The model to extract weights from
        filepath: Path to save the text file
    """
    with open(filepath, 'w') as f:
        f.write("ICANN MODEL WEIGHTS HIERARCHY\n")
        f.write("============================\n\n")
        
        # Find the RNN layer which contains our piCANN cell
        rnn_layer = None
        for layer in model.layers:
            if isinstance(layer, keras.layers.RNN):
                rnn_layer = layer
                break
        
        if rnn_layer is None:
            f.write("ERROR: Could not find RNN layer with piCANN cell!\n")
            return
        
        # Get the piCANN cell
        icann_cell = rnn_layer.cell
        
        # Print model summary information
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Cell Type: {type(icann_cell).__name__}\n")
        if hasattr(icann_cell, 'n'):
            f.write(f"Network sizes (n): {icann_cell.n}\n")
        f.write("\n")
        
        # Helper function to write weight details with indentation
        def write_weights(obj, name, indent=0):
            indent_str = "  " * indent
            f.write(f"{indent_str}{name}:\n")
            
            if not hasattr(obj, 'weights') or not obj.weights:
                f.write(f"{indent_str}  (No trainable weights)\n")
                return
            
            for weight in obj.weights:
                # Get shape and some sample values
                shape_str = str(weight.shape)
                
                # Get statistics about the weight values
                weight_np = weight.numpy()
                stats = {
                    'min': np.min(weight_np),
                    'max': np.max(weight_np),
                    'mean': np.mean(weight_np),
                    'std': np.std(weight_np)
                }
                
                stats_str = f"min={stats['min']:.5f}, max={stats['max']:.5f}, mean={stats['mean']:.5f}, std={stats['std']:.5f}"
                
                # Check if this weight has regularization
                reg_str = ""
                if hasattr(weight, '_regularizer') and weight._regularizer is not None:
                    reg_str = " (has regularization)"
                
                # Check if this weight has constraints
                constr_str = ""
                if hasattr(weight, 'constraint') and weight.constraint is not None:
                    if isinstance(weight.constraint, tf.keras.constraints.NonNeg):
                        constr_str = " (constrained: NonNeg)"
                    else:
                        constr_str = " (has constraints)"
                
                # Write the weight info
                f.write(f"{indent_str}  {weight.name}: {shape_str}{reg_str}{constr_str}\n")
                f.write(f"{indent_str}    {stats_str}\n")
                
                # Write the actual weight values
                f.write(f"{indent_str}    Values:\n")
                
                # Format the values based on dimensionality
                if len(weight_np.shape) == 1:  # 1D array
                    values_str = np.array2string(weight_np, precision=6, threshold=np.inf, max_line_width=120)
                    f.write(f"{indent_str}      {values_str}\n")
                elif len(weight_np.shape) == 2:  # 2D array
                    for i, row in enumerate(weight_np):
                        values_str = np.array2string(row, precision=6, threshold=np.inf, max_line_width=120)
                        f.write(f"{indent_str}      [{i}] {values_str}\n")
                else:  # Higher dimensional arrays
                    flat_weights = weight_np.flatten()
                    if flat_weights.size > 100:
                        # For large weights, just show first and last few values
                        first_values = np.array2string(flat_weights[:50], precision=6, threshold=np.inf)
                        last_values = np.array2string(flat_weights[-50:], precision=6, threshold=np.inf)
                        f.write(f"{indent_str}      First 50: {first_values}\n")
                        f.write(f"{indent_str}      Last 50: {last_values}\n")
                    else:
                        # For smaller weights, show all flattened values
                        values_str = np.array2string(flat_weights, precision=6, threshold=np.inf)
                        f.write(f"{indent_str}      {values_str}\n")
                
                f.write("\n")
        
        # Write main layers
        f.write("MAIN LAYERS:\n")
        for layer in model.layers:
            write_weights(layer, layer.name, indent=1)
        
        # Write icann cell components
        f.write("\nICANN CELL COMPONENTS:\n")
        
        # Helmholtz Networks
        for hnet_name in ['hNet_e', 'hNet_i']:
            if hasattr(icann_cell, hnet_name):
                hnet = getattr(icann_cell, hnet_name)
                write_weights(hnet, hnet_name, indent=1)
                
                # Write sublayers of Helmholtz networks
                for subnet_name in ['iso_net1', 'iso_net2', 'vol_net']:
                    if hasattr(hnet, subnet_name):
                        subnet = getattr(hnet, subnet_name)
                        write_weights(subnet, f"{hnet_name}.{subnet_name}", indent=2)
                        
                        # For vol_net, also write power_expansion
                        if subnet_name == 'vol_net' and hasattr(subnet, 'power_expansion'):
                            write_weights(subnet.power_expansion, f"{hnet_name}.{subnet_name}.power_expansion", indent=3)
        
        # Flow potential networks
        for gnet_idx in range(1, 4):
            gnet_name = f'gNet{gnet_idx}'
            if hasattr(icann_cell, gnet_name):
                gnet = getattr(icann_cell, gnet_name)
                write_weights(gnet, gnet_name, indent=1)
                
                # Write polynomial layer
                if hasattr(gnet, 'poly'):
                    write_weights(gnet.poly, f"{gnet_name}.poly", indent=2)
        
        # Other components
        for comp_name in ['uniform_yield_weight']:
            if hasattr(icann_cell, comp_name):
                comp = getattr(icann_cell, comp_name)
                if hasattr(comp, 'numpy'):  # It's a weight
                    val_str = np.array2string(comp.numpy(), precision=6)
                    f.write(f"  {comp_name}: {comp.shape} = {val_str}\n")

def save_weights_as_npy(model, save_path):
    """
    Save weight values as individual .npy files for easy loading and analysis.
    
    Args:
        model: The model to extract weights from
        save_path: Base path to save the weight files
    """
    # Create weights directory
    weights_dir = os.path.join(save_path, 'weight_values')
    os.makedirs(weights_dir, exist_ok=True)
    
    # Find the RNN layer which contains our piCANN cell
    rnn_layer = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.RNN):
            rnn_layer = layer
            break
    
    if rnn_layer is None:
        print("ERROR: Could not find RNN layer with piCANN cell!")
        return
    
    # Get the piCANN cell
    icann_cell = rnn_layer.cell
    
    # Helper function to save weights
    def save_component_weights(obj, prefix):
        if not hasattr(obj, 'weights') or not obj.weights:
            return
        
        for weight in obj.weights:
            # Create a clean filename
            clean_name = weight.name.replace('/', '_').replace(':', '_')
            filename = f"{prefix}_{clean_name}.npy"
            filepath = os.path.join(weights_dir, filename)
            
            # Save as numpy array
            np.save(filepath, weight.numpy())
            # print(f"Saved {weight.name} to {filename}")
    
    # Save main layer weights
    for layer in model.layers:
        save_component_weights(layer, "main")
    
    # Save Helmholtz networks
    for hnet_name in ['hNet_e', 'hNet_i']:
        if hasattr(icann_cell, hnet_name):
            hnet = getattr(icann_cell, hnet_name)
            save_component_weights(hnet, hnet_name)
            
            # Save sublayers
            for subnet_name in ['iso_net1', 'iso_net2', 'vol_net']:
                if hasattr(hnet, subnet_name):
                    subnet = getattr(hnet, subnet_name)
                    save_component_weights(subnet, f"{hnet_name}_{subnet_name}")
                    
                    # For vol_net, also save power_expansion
                    if subnet_name == 'vol_net' and hasattr(subnet, 'power_expansion'):
                        save_component_weights(subnet.power_expansion, 
                                               f"{hnet_name}_{subnet_name}_power_expansion")
    
    # Save flow potential networks
    for gnet_idx in range(1, 4):
        gnet_name = f'gNet{gnet_idx}'
        if hasattr(icann_cell, gnet_name):
            gnet = getattr(icann_cell, gnet_name)
            save_component_weights(gnet, gnet_name)
            
            # Save polynomial layer
            if hasattr(gnet, 'poly'):
                save_component_weights(gnet.poly, f"{gnet_name}_poly")
    
    # Save other components
    for comp_name in ['uniform_yield_weight']:
        if hasattr(icann_cell, comp_name):
            comp = getattr(icann_cell, comp_name)
            if hasattr(comp, 'numpy'):  # It's a weight
                filename = f"{comp_name}.npy"
                filepath = os.path.join(weights_dir, filename)
                np.save(filepath, comp.numpy())
    
    return weights_dir

def create_callbacks(save_path, patience=50, reduce_lr_patience=10, min_lr=1e-10):
    """
    Create a standard set of training callbacks.

    Args:
        save_path: Base path for saving model checkpoints and logs
        patience: Early stopping patience
        reduce_lr_patience: Patience for learning rate reduction
        
    Returns:
        List of callbacks for model training
    """
    # Create subdirectories
    logs_dir = os.path.join(save_path, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    checkpoints_dir = os.path.join(save_path, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    callbacks = [
        # keras.callbacks.ModelCheckpoint(
        #     filepath=os.path.join(checkpoints_dir, 'icann_epoch_{epoch:03d}.keras'),
        #     save_best_only=True,
        #     monitor='loss',
        #     verbose=1
        # ),
        # keras.callbacks.ModelCheckpoint(
        #     filepath=os.path.join(save_path, 'icann_best.keras'),
        #     save_best_only=True,
        #     monitor='loss',
        #     verbose=1
        # ),
        keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        SmartReduceLROnPlateau(  # Use our enhanced callback instead
            monitor='loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
            verbose=1
        ),
        keras.callbacks.CSVLogger(
            os.path.join(logs_dir, 'training_history.csv')
        ),
        keras.callbacks.TensorBoard(
            log_dir=logs_dir,
            histogram_freq=1,
            update_freq='epoch'
        ),
        TenEpochLogger(print_interval=10)
    ]
    
    return callbacks, logs_dir

def plot_training_history(history, logs_dir, title="Training Loss Over Time"):
    """
    Plot and save training history.
    
    Args:
        history: Training history object
        logs_dir: Directory to save the plot
        title: Title for the plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Loss')
    
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(logs_dir, 'training_loss.png'))
    plt.close()

def load_training_data(data_path="data/simple_paths", variant_range=None):
    """
    Load training data from npz files and prepare for iCANN model.
    
    Args:
        data_path: Path to the directory containing path_*.npz files
        variant_range: Range of variants to load (default: all available)
    
    Returns:
        tuple: (train_inputs, train_outputs, raw_inputs, raw_outputs)
            - train_inputs: Dictionary with concatenated tensors ready for model.fit
            - train_outputs: List of tensors (S11, S22, S33, dummy...) ready for model.fit
            - raw_inputs: Dictionary with individual tensors per variant
            - raw_outputs: List of individual output tensors per variant
    """
    # Initialize data containers
    x_train = {
        'C11': [],
        'C22': [],
        'C33': [],
        'dt': []
    }
    
    y_train = []  # List to hold target stresses [S11, S22, S33] for each time series
    
    # Default to loading all variants if not specified
    if variant_range is None:
        variant_range = range(5)  # Default: load variants 0-4
    
    # Load each variant
    for i in variant_range:
        file_path = os.path.join(data_path, f"path_{i}.npz")
        if os.path.exists(file_path):
            data = np.load(file_path)
            strains = data['strains']  # Shape: (time_steps, 6)
            stresses = data['stresses'] / 4  # Shape: (time_steps, 6)
            
            time_steps = strains.shape[0]
            
            # Extract components and reshape for RNN
            C11 = strains[:, 0].reshape(1, time_steps, 1)**2  # Add batch dimension
            C22 = strains[:, 1].reshape(1, time_steps, 1)**2
            C33 = strains[:, 2].reshape(1, time_steps, 1)**2
            
            S11 = stresses[:, 0].reshape(1, time_steps, 1)
            S22 = stresses[:, 1].reshape(1, time_steps, 1)
            S33 = stresses[:, 2].reshape(1, time_steps, 1)
            
            dt = 1e0 * np.ones((1, time_steps, 1)) # For some reason, the training is more stable with higher dt
            
            # Add to our dataset
            x_train['C11'].append(C11)
            x_train['C22'].append(C22)
            x_train['C33'].append(C33)
            x_train['dt'].append(dt)
            y_train.append([S11, S22, S33])
            
            print(f"Loaded data for variant {i}: {time_steps} time steps")
        else:
            print(f"Warning: File {file_path} not found!")
    
    if not x_train['C11']:
        raise ValueError(f"No data files were found in {data_path}!")
    
    # Create combined inputs for model training
    train_inputs = {
        'C11': tf.concat([tensor for tensor in x_train['C11']], axis=0),
        'C22': tf.concat([tensor for tensor in x_train['C22']], axis=0),
        'C33': tf.concat([tensor for tensor in x_train['C33']], axis=0),
        'dt': tf.concat([tensor for tensor in x_train['dt']], axis=0)
    }

    # Create combined targets
    S11_combined = tf.concat([tensor for tensor in [y[0] for y in y_train]], axis=0)
    S22_combined = tf.concat([tensor for tensor in [y[1] for y in y_train]], axis=0)
    S33_combined = tf.concat([tensor for tensor in [y[2] for y in y_train]], axis=0)

    # Create dummy targets with appropriate shape
    batch_size = S11_combined.shape[0]
    time_steps = S11_combined.shape[1]
    dummy_target = tf.zeros((batch_size, time_steps, 1))
    
    # Prepare outputs in the format expected by model.fit()
    train_outputs = [S11_combined, S22_combined, S33_combined, 
                    dummy_target, dummy_target, dummy_target]
    
    return train_inputs, train_outputs, x_train, y_train

def visualize_strain_stress(model, x_data, y_data, save_path):
    """
    Create strain-stress plots with time evolution indicated by color.
    """
    # Create a custom colormap for time evolution
    colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # Blue -> Green -> Red
    time_cmap = LinearSegmentedColormap.from_list('time_evolution', colors, N=100)
    
    # Create visualization directory
    viz_dir = os.path.join(save_path, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Process each time series
    for i in range(len(y_data)):
        # Prepare inputs for prediction
        inputs = {
            'C11': x_data['C11'][i],
            'C22': x_data['C22'][i],
            'C33': x_data['C33'][i],
            'dt': x_data['dt'][i]
        }
        
        # Get model predictions
        predictions = model.predict(inputs)
        
        # Extract stress components and remove batch dimension
        pred_S11 = predictions[0][0]
        pred_S22 = predictions[1][0]
        pred_S33 = predictions[2][0]
        
        # Ground truth stresses
        true_S11 = y_data[i][0][0]
        true_S22 = y_data[i][1][0]
        true_S33 = y_data[i][2][0]
        
        # Extract strain components
        C11 = x_data['C11'][i][0]
        C22 = x_data['C22'][i][0]
        C33 = x_data['C33'][i][0]
        
        # Time steps for color mapping
        time_steps = C11.shape[0]
        time_normalized = np.linspace(0, 1, time_steps)
        
        # Create plots for each stress component vs each strain component
        components = [
            ('C11', 'S11', C11, pred_S11, true_S11),
            ('C22', 'S22', C22, pred_S22, true_S22), 
            ('C33', 'S33', C33, pred_S33, true_S33),
            ('C11', 'S22', C11, pred_S22, true_S22),  # Cross-component plots
            ('C22', 'S11', C22, pred_S11, true_S11)
        ]
        
        for strain_name, stress_name, strain, pred_stress, true_stress in components:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Scatter plot with color representing time evolution - store the scatter object
            scatter_pred = ax.scatter(strain, pred_stress, c=time_normalized, 
                        cmap=time_cmap, s=20, alpha=0.7, marker='o', label='Predicted')
                
            scatter_true = ax.scatter(strain, true_stress, c=time_normalized, 
                        cmap=time_cmap, marker='x', s=30, alpha=0.5, label='True')
            
            # Add a colorbar using the scatter object directly
            cbar = fig.colorbar(scatter_pred, ax=ax)
            cbar.set_label('Time Evolution')
            
            ax.set_xlabel(f'Strain ({strain_name})')
            ax.set_ylabel(f'Stress ({stress_name})')
            ax.set_title(f'{stress_name} vs {strain_name} - Variant {i}')
            ax.grid(True)
            ax.legend(handles=[
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label='Predicted'),
                plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='b', markersize=8, label='True')
            ])
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f'variant_{i}_{stress_name}_vs_{strain_name}.png'))
            plt.close(fig)
        
        # Create a combined strain path visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(C11, C22, c=time_normalized, cmap=time_cmap, s=30, alpha=0.7)
        
        ax.set_xlabel('C11')
        ax.set_ylabel('C22')
        ax.set_title(f'C11 vs C22 Strain Path - Variant {i}')
        ax.grid(True)
        
        # Add colorbar directly to the scatter plot
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Time Evolution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'variant_{i}_strain_path.png'))
        plt.close(fig)

class NaNDetectionCallback(tf.keras.callbacks.Callback):
    def __init__(self, max_restarts=10, initial_seed=42, restart_threshold=10, model_args=None):
        super().__init__()
        self.max_restarts = max_restarts
        self.initial_seed = initial_seed
        self.restart_threshold = restart_threshold
        self.model_args = model_args or {}
        self.nan_detected = False
        self.nan_epoch = None  # Track when NaN first occurs
        self.history = []
        self.restart_info = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        self.history.append(loss)
        
        if np.isnan(loss) or np.isinf(loss):
            if not self.nan_detected:  # Only record the first NaN occurrence
                # self.nan_detected = True
                self.nan_epoch = epoch + 1  # +1 because epochs are 0-indexed
                
                # Only restart if within threshold
                if self.nan_epoch <= self.restart_threshold:
                    print(f"\n⚠️ NaN/Inf detected at epoch {self.nan_epoch} (within restart threshold).")
                    self.nan_detected = True
                    self.model.stop_training = True
                # else:
                #     print(f"\n⚠️ NaN/Inf detected at epoch {self.nan_epoch} (beyond restart threshold).")


class SmartReduceLROnPlateau(keras.callbacks.Callback):
    """
    Custom callback that combines learning rate reduction with weight restoration.
    
    When training plateaus, this will:
    1. Restore the model to its best weights
    2. Then reduce the learning rate
    
    Args:
        monitor: Quantity to monitor
        factor: Factor by which to reduce learning rate
        patience: Number of epochs with no improvement after which LR will be reduced
        min_lr: Lower bound on learning rate
        verbose: Whether to print messages
    """
    def __init__(self, monitor='loss', factor=0.5, patience=10, 
                    min_lr=1e-7, verbose=1):
        super().__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best_weights = None
        self.best = float('inf')
        self.wait = 0
        self.best_epoch = 0
        self.prev_weight = None  # Track previous weights

    def on_train_begin(self, logs=None):
        self.best = float('inf')
        self.wait = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        # Store the current weights before any changes
        self.prev_weight = self.model.get_weights()
        
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if epoch <= 1:
            self.best_weights = self.model.get_weights()

        def get_lr(optimizer):
            """Get learning rate from optimizer safely"""
            try:
                # Adding safety conversion for TensorFlow 2.x
                from keras import backend as K
                return float(K.get_value(optimizer.lr))
            except:
                try:
                    # Direct .numpy() approach
                    if hasattr(optimizer.learning_rate, "numpy"):
                        return float(optimizer.learning_rate.numpy())
                    else:
                        return float(optimizer.learning_rate)
                except:
                    # Last resort
                    print("Could not extract learning rate from optimizer!")
                    return 0.0001
        
        # If this is a better result, save the weights and reset patience
        if current < self.best:
            self.best = current
            self.wait = 0
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(self.prev_weight)  # Store the best weights
            if self.verbose == 1 and epoch % 10 == 0:
                print(f"\nEpoch {epoch}: {self.monitor} improved to {current:.5f}, saving best weights, current LR={get_lr(self.model.optimizer):.2e}")
        else:
            self.wait += 1
            if self.verbose == 1 and epoch % 10 == 0:
                print(f"\nEpoch {epoch}: {self.monitor} did not improve from {self.best:.5f}")
                
            if self.wait >= self.patience:
                old_lr = get_lr(self.model.optimizer)
                    
                if old_lr > self.min_lr:
                    # Calculate new learning rate
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    
                    if self.verbose:
                        print(f"\n⚙️ Epoch {epoch}: Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")
                        print(f"⚙️ Restoring best weights from epoch {self.best_epoch} with {self.monitor}={self.best:.5f}")
                    
                    # Restore best weights first
                    self.model.set_weights(self.best_weights)
                    
                    # Then update learning rate
                    self.model.optimizer.learning_rate = new_lr
                    
                    # Reset patience counter
                    self.wait = 0
            elif (np.isinf(current) or np.isnan(current)) and epoch > 1:
                # Reduce lr and restore best weights if NaN/Inf detected
                if self.verbose:
                    print(f"\n⚠️ Epoch {epoch}: NaN/Inf detected. Reducing learning rate and restoring best weights.")
                self.model.set_weights(self.best_weights)
                old_lr = get_lr(self.model.optimizer)
                new_lr = max(old_lr * self.factor, self.min_lr)
                self.model.optimizer.learning_rate = new_lr
                if self.verbose:
                    print(f"⚙️ Learning rate reduced to {new_lr:.2e}")
                self.wait = 0

class WarmupLRSchedule(tf.keras.callbacks.Callback):
    """
    Custom callback that implements a warmup learning rate schedule.
    
    Args:
        initial_lr: Starting learning rate (small value)
        target_lr: Target learning rate to reach after warmup
        warmup_epochs: Number of epochs for warmup phase
        verbose: Whether to print messages
    """
    def __init__(self, initial_lr=1e-7, target_lr=2e-3, warmup_epochs=10, verbose=1):
        super().__init__()
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Exponential warmup from initial_lr to target_lr
            # Calculate learning rate based on epoch
            lr = self.initial_lr * (self.target_lr / self.initial_lr) ** (epoch / self.warmup_epochs)
            # lr = self.initial_lr + (self.target_lr - self.initial_lr) * (epoch / self.warmup_epochs)
            self.model.optimizer.learning_rate = lr
            if self.verbose:
                print(f"\n🔥 Warmup phase: Learning rate set to {lr:.2e}")
        elif epoch == self.warmup_epochs:
            # Set to target learning rate
            self.model.optimizer.learning_rate = self.target_lr
            if self.verbose:
                print(f"\n🔥 Warmup complete: Learning rate set to {self.target_lr:.2e}")

import time
class TenEpochLogger(keras.callbacks.Callback):
    """
    Custom callback that logs training progress every 10 epochs.
    Shows time elapsed since the last print interval with Keras-style formatting.
    """
    def __init__(self, print_interval=10):
        super().__init__()
        self.print_interval = print_interval
        self.last_print_time = None
        
    def on_train_begin(self, logs=None):
        self.last_print_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_interval == 0 or epoch == 0:
            # Calculate time for this interval
            current_time = time.time()
            interval_time = current_time - self.last_print_time
            self.last_print_time = current_time
            
            # Calculate progress for the progress bar
            progress = float(epoch + 1) / self.params['epochs']
            bar_length = 30
            
            # Create progress bar similar to Keras style
            bar = '=' * int(bar_length * progress)
            if progress < 1:
                bar += '>'
            bar += '.' * (bar_length - len(bar))
            
            # Format the logs similar to Keras output
            log_items = []
            for k, v in logs.items():
                log_items.append(f'{k}: {v:.4f}')
            
            # Add timing information for the interval
            log_items.append(f'{self.print_interval}-epoch time: {interval_time:.2f}s')
            
            # Join everything
            log_str = ' - '.join(log_items)
            
            # Print with progress bar
            print(f"\r{epoch + 1}/{self.params['epochs']} [{bar}] - {log_str}")

from scipy.io import loadmat
def load_matlab_data(file_paths,  dt_value=1.0, normalize=True):
    """
    Load training data from MATLAB .mat files for the iCANN model.
    
    Args:
        file_paths: List of paths to .mat files containing time series data
            Each file should contain 'C' and 'PK2' variables
        scale_factor: Scaling factor for stresses (default: 1/8.0 to match load_training_data)
        dt_value: Value to use for the dt input (default: 1.0)
        normalize: Whether to normalize stress values by max absolute value (default: True)
    
    Returns:
        tuple: (train_inputs, train_outputs, x_train, y_train)
            - train_inputs: Dictionary with concatenated tensors ready for model.fit
            - train_outputs: List of tensors (S11, S22, S33, dummy...) ready for model.fit
            - x_train: Dictionary with individual tensors per variant
            - y_train: List of individual output tensors per variant
    """
    # Initialize data containers
    x_train = {
        'C11': [],
        'C22': [],
        'C33': [],
        'dt': []
    }
    
    # For normalization, first collect all stress data
    all_stresses = []
    raw_data = []  # Store all loaded data for later processing
    
    # Load each provided .mat file
    for i, file_path in enumerate(file_paths):
        if os.path.exists(file_path):
            print(f"Loading data from {file_path}...")
            
            # Load the .mat file
            mat_data = loadmat(file_path)
            
            # Extract strain and stress data
            if 'C' in mat_data and 'PK2' in mat_data:
                strains = mat_data['C']     # Shape: (time_steps, 3)
                stresses = mat_data['PK2']  # Shape: (time_steps, 3)
                
                # Store raw data for later processing
                raw_data.append((strains, stresses, i))
                
                # Collect all stress values for normalization
                all_stresses.append(stresses)
                
                print(f"Loaded time series {i} with {strains.shape[0]} steps (PK2)")
            elif 'C' in mat_data and 'cauchy' in mat_data:
                strains = mat_data['C']     # Shape: (time_steps, 3)
                stresses = mat_data['cauchy']  # Shape: (time_steps, 3)
                
                # Store raw data for later processing
                raw_data.append((strains, stresses, i))
                
                # Collect all stress values for normalization
                all_stresses.append(stresses)
                
                print(f"Loaded time series {i} with {strains.shape[0]} steps (cauchy)")
            else:
                print(f"Warning: Required variables 'C' and 'PK2/cauchy' not found in {file_path}!")
                print(f"Available variables: {list(mat_data.keys())}")
        else:
            print(f"Warning: File {file_path} not found!")
    
    if not raw_data:
        raise ValueError(f"No valid data was loaded from the provided .mat files!")
    
    # Normalization factor - max absolute value across all stress components
    if normalize and all_stresses:
        all_stresses_array = np.vstack(all_stresses)
        max_abs_stress = np.max(np.abs(all_stresses_array))
        print(f"Normalizing all stresses by maximum absolute value: {max_abs_stress:.6f}")
        norm_factor = 0.5 * max_abs_stress
    else:
        norm_factor = 1.0
    
    # Process the loaded data
    y_train = []  # List to hold target stresses
    
    for strains, stresses, i in raw_data:
        time_steps = strains.shape[0]
        
        # Extract components and reshape for RNN
        # Note: C is already the right Cauchy-Green tensor, no need to square
        C11 = strains[:, 0].reshape(1, time_steps, 1)  # Add batch dimension
        C22 = strains[:, 1].reshape(1, time_steps, 1)
        C33 = strains[:, 2].reshape(1, time_steps, 1)
        
        # Normalize stresses if requested
        if normalize:
            S11 = (stresses[:, 0] / norm_factor).reshape(1, time_steps, 1)
            S22 = (stresses[:, 1] / norm_factor).reshape(1, time_steps, 1)
            S33 = (stresses[:, 2] / norm_factor).reshape(1, time_steps, 1)
        else:
            S11 = stresses[:, 0].reshape(1, time_steps, 1)
            S22 = stresses[:, 1].reshape(1, time_steps, 1)
            S33 = stresses[:, 2].reshape(1, time_steps, 1)
        
        dt = dt_value * np.ones((1, time_steps, 1))
        
        # Add to our dataset
        x_train['C11'].append(C11)
        x_train['C22'].append(C22)
        x_train['C33'].append(C33)
        x_train['dt'].append(dt)
        y_train.append([S11, S22, S33])
        
        print(f"Processed time series {i} with {time_steps} steps")
    
    # Create combined inputs for model training
    train_inputs = {
        'C11': tf.concat([tensor for tensor in x_train['C11']], axis=0),
        'C22': tf.concat([tensor for tensor in x_train['C22']], axis=0),
        'C33': tf.concat([tensor for tensor in x_train['C33']], axis=0),
        'dt': tf.concat([tensor for tensor in x_train['dt']], axis=0)
    }

    # Create combined targets
    S11_combined = tf.concat([tensor for tensor in [y[0] for y in y_train]], axis=0)
    S22_combined = tf.concat([tensor for tensor in [y[1] for y in y_train]], axis=0)
    S33_combined = tf.concat([tensor for tensor in [y[2] for y in y_train]], axis=0)

    # Create dummy targets with appropriate shape
    batch_size = S11_combined.shape[0]
    time_steps = S11_combined.shape[1]
    dummy_target = tf.zeros((batch_size, time_steps, 1))
    
    # Prepare outputs in the format expected by model.fit()
    train_outputs = [S11_combined, S22_combined, S33_combined, 
                    dummy_target, dummy_target, dummy_target]
    
    # Save normalization factor for later use during inference
    if normalize:
        # Return normalization factor as metadata
        return train_inputs, train_outputs, x_train, y_train#, {'stress_norm_factor': float(norm_factor)}
    else:
        return train_inputs, train_outputs, x_train, y_train

def load_checkpoint(save_path, stage, n=[1, 1, 1, 1, 1], seed=1234):
    """
    Load a model checkpoint from a completed stage with fallback to manual loading.
    
    Args:
        save_path: Base directory where checkpoints are saved
        stage: Stage number to load (1-based indexing)
        n: Network sizes for creating a fresh model if needed
        seed: Random seed for reproducibility
        
    Returns:
        model: Loaded model or None if loading failed
    """
    stage_dir = os.path.join(save_path, f'stage{stage}')
    model_path = os.path.join(stage_dir, 'model.h5')
    weights_path = os.path.join(stage_dir, 'icann_model_final.weights.h5')
    
    if not os.path.exists(stage_dir):
        print(f"No checkpoint directory found for stage {stage} at {stage_dir}")
        return None
    
    # Try loading the full model first
    if os.path.exists(model_path):
        try:
            print(f"Loading full model from stage {stage}...")
            model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'piCANN': piCANN,
                    'stress_mse_loss': stress_mse_loss,
                    'dummy_loss': dummy_loss
                }
            )
            print(f"Successfully loaded full model from stage {stage}")
            return model
        except Exception as e:
            print(f"Error loading full model: {str(e)}")
            print("Will try loading weights instead.")
    
    # If full model loading failed or model file doesn't exist, try loading just the weights
    if not os.path.exists(weights_path):
        # Check for best weights
        weights_path = os.path.join(stage_dir, 'icann_best.weights.h5')
        if not os.path.exists(weights_path):
            print(f"No weights file found for stage {stage}")
            return None
    
    # Create a fresh model with correct architecture
    print(f"Creating new model for loading weights from stage {stage}...")
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
        print("Weights loaded successfully")
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
        

def prepare_truncated_data(train_inputs, train_outputs, percentage=10):
    """
    Prepare data for staged training by taking only the first X% of time steps.
    """
    # Calculate number of time steps to keep
    time_steps = train_inputs['C11'].shape[1]
    keep_steps = max(1, int(time_steps * percentage / 100))
    
    print(f"Preparing truncated data: keeping first {keep_steps}/{time_steps} time steps ({percentage}%)")
    
    # Create truncated inputs
    truncated_inputs = {
        'C11': train_inputs['C11'][:, :keep_steps, :],
        'C22': train_inputs['C22'][:, :keep_steps, :],
        'C33': train_inputs['C33'][:, :keep_steps, :],
        'dt': train_inputs['dt'][:, :keep_steps, :]
    }
    
    # Create truncated outputs
    truncated_outputs = [
        train_outputs[0][:, :keep_steps, :],  # S11
        train_outputs[1][:, :keep_steps, :],  # S22
        train_outputs[2][:, :keep_steps, :],  # S33
        train_outputs[3][:, :keep_steps, :],  # dummy
        train_outputs[4][:, :keep_steps, :],  # dummy
        train_outputs[5][:, :keep_steps, :]   # dummy
    ]
    
    return truncated_inputs, truncated_outputs

def copy_weights_from_hNet_e_to_hNet_i(model, scale_factor=0.25):
    """
    Copy weights from hNet_e to hNet_i, scaled by the specified factor.
    
    Args:
        model: The model containing both hNet_e and hNet_i
        scale_factor: Factor to multiply hNet_e weights by when copying to hNet_i
        
    Returns:
        The model with updated hNet_i weights
    """
    # Find the RNN layer which contains our piCANN cell
    rnn_layer = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.RNN):
            rnn_layer = layer
            break
    
    if rnn_layer is None:
        print("WARNING: Could not find RNN layer to copy weights!")
        return model
    
    # Get the piCANN cell
    icann_cell = rnn_layer.cell
    
    # Check if both hNet_e and hNet_i exist
    if not hasattr(icann_cell, 'hNet_e') or not hasattr(icann_cell, 'hNet_i'):
        print("WARNING: Model does not have both hNet_e and hNet_i!")
        return model
    
    print(f"Copying weights from hNet_e to hNet_i with scale factor {scale_factor}...")
    
    # Function to copy weights between layers, respecting constraints
    def copy_weights_between_layers(source_layer, target_layer, scale):
        if len(source_layer.weights) != len(target_layer.weights):
            print(f"WARNING: {source_layer.name} and {target_layer.name} have different number of weights!")
            return
        
        for src_w, tgt_w in zip(source_layer.weights, target_layer.weights):
            # Check if shapes match
            if src_w.shape != tgt_w.shape:
                print(f"WARNING: Weight shapes don't match: {src_w.shape} vs {tgt_w.shape} - skipping")
                continue
            
            # Create scaled copy
            scaled_weight = src_w * scale
            
            # Handle constraints for specific weight types
            if ('cp1' in tgt_w.name or 'cp2' in tgt_w.name or 'cp3' in tgt_w.name or 
                'coefficients' in tgt_w.name):
                # Ensure non-negative for constrained weights
                scaled_weight = tf.maximum(scaled_weight, 1e-12)
            
            tgt_w.assign(scaled_weight)
            print(f"Copied and scaled weight: {src_w.name} → {tgt_w.name}")
    
    # Handle each sublayer if it exists in both networks
    for subnet_name in ['iso_net1', 'iso_net2', 'vol_net']:
        if hasattr(icann_cell.hNet_e, subnet_name) and hasattr(icann_cell.hNet_i, subnet_name):
            src_subnet = getattr(icann_cell.hNet_e, subnet_name)
            tgt_subnet = getattr(icann_cell.hNet_i, subnet_name)
            copy_weights_between_layers(src_subnet, tgt_subnet, scale_factor)
            
            # Handle nested layers like vol_net.power_expansion if needed
            if subnet_name == 'vol_net' and hasattr(src_subnet, 'power_expansion') and hasattr(tgt_subnet, 'power_expansion'):
                copy_weights_between_layers(src_subnet.power_expansion, tgt_subnet.power_expansion, scale_factor)
    
    return model

def check_for_nans(model, inputs):
    """
    Check if model predictions contain any NaN values.
    
    Args:
        model: The model to check
        inputs: Input data dictionary
        
    Returns:
        bool: True if NaNs were found, False otherwise
    """
    try:
        # Make predictions on inputs
        predictions = model.predict(inputs)
        
        # Check each output for NaNs
        has_nans = False
        for i, pred in enumerate(predictions):
            if np.isnan(pred).any():
                print(f"NaN detected in output {i}")
                has_nans = True
                
        return has_nans
    except Exception as e:
        print(f"Error during NaN check: {e}")
        # If we got an error during prediction, it likely means there are NaNs
        return True

# Additional utility functions

def freeze_nets(model, frozen_nets=[], unfrozen_nets=[]):
    """
    Freeze the weights of specified neural networks and unfreeze others in the model.
    
    Args:
        model: The model containing the neural networks
        frozen_nets: List of network names to freeze (e.g. ['hNet_e', 'gNet1'])
        unfrozen_nets: List of network names to explicitly unfreeze
        
    Returns:
        The model with updated trainable status for specified networks
    """
    # Find the RNN layer which contains our piCANN cell
    rnn_layer = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.RNN):
            rnn_layer = layer
            break
    
    if rnn_layer is None:
        print("WARNING: Could not find RNN layer to modify network trainability!")
        return model
    
    # Get the piCANN cell
    icann_cell = rnn_layer.cell
    
    # Process networks to freeze
    for net_name in frozen_nets:
        if hasattr(icann_cell, net_name):
            print(f"Freezing {net_name}...")
            getattr(icann_cell, net_name).trainable = False
        else:
            print(f"WARNING: Model does not have {net_name} to freeze!")
    
    # Process networks to unfreeze
    for net_name in unfrozen_nets:
        if hasattr(icann_cell, net_name):
            print(f"Unfreezing {net_name}...")
            getattr(icann_cell, net_name).trainable = True
        else:
            print(f"WARNING: Model does not have {net_name} to unfreeze!")
    
    # Print trainable parameters to verify
    trainable_count = sum([tf.size(w).numpy() for w in icann_cell.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in icann_cell.non_trainable_weights])
    print(f"Trainable params: {trainable_count}, Non-trainable params: {non_trainable_count}")
    
    return model

# Additional utility functions

def extract_time_series(data, series):
    """
    Extract a specific time series from the training data by index.
    
    Args:
        data: Dictionary of training inputs or list of training outputs
        series: Index of the time series to extract (0-based)
    
    Returns:
        The extracted time series data
    """
    if isinstance(data, dict):
        # For train_inputs (dictionary format)
        return {k: v[series:series+1, :, :] for k, v in data.items()}
    elif isinstance(data, list):
        # For train_outputs (list format)
        return [output[series:series+1, :, :] for output in data]
    else:
        # For numpy array format
        return data[series:series+1, :, :]