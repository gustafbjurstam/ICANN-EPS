import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import h5py

# Add project root to Python path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary model classes and utilities
from src.models.icann.load_model_utils import create_and_load_model_from_weights
from src.models.icann.icann_3d_asymmetry import HelmholtzIsoNet, HelmholtzVolNet
def extract_weights_to_fortran(model_path, output_path, module_name="piCANN_weights"):
    """
    Extract weights from a piCANN model and write them as Fortran constant arrays.
    
    Args:
        model_path: Path to the model weights directory or file
        output_path: Path to write the Fortran module file
        module_name: Name of the Fortran module to create
    """
    # Get network sizes from params file if available
    params_file = os.path.join(model_path, 'model_params.json')
    n = [1, 1, 1, 1, 1]  # Default sizes
    if os.path.exists(params_file):
        import json
        with open(params_file, 'r') as f:
            params = json.load(f)
            if 'n' in params:
                n = params['n']
                print(f"Loaded network sizes from params file: {n}")
    
    # Create a fresh model with correct architecture
    from src.models.icann.icann_3d_asymmetry import create_rnn_model
    model = create_rnn_model(n)
    
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
    
    # Determine weights path
    weights_path = None
    if os.path.isdir(model_path):
        # Try standard weight file locations
        for filename in ['icann_model_final.weights.h5', 'icann_best.weights.h5']:
            path = os.path.join(model_path, filename)
            if os.path.exists(path):
                weights_path = path
                break
    elif os.path.isfile(model_path):
        weights_path = model_path
    
    if weights_path is None:
        print(f"No weights file found at {model_path}")
        return
    
    # Load weights
    print(f"Loading weights from: {weights_path}")
    try:
        model.load_weights(weights_path)
        print("Weights loaded successfully")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Attempting layer-by-layer weight loading...")
        
        # Try to load saved weights and apply them manually
        saved_weights = {}
        with h5py.File(weights_path, 'r') as f:
            for key in f.keys():
                # Skip non-weight keys
                if key in ['optimizer_weights', 'model_config']:
                    continue
                
                # Extract weights and add to dictionary
                saved_weights[key] = f[key][:]
        
        # Find matching weights and assign them
        for weight in model.weights:
            name = weight.name
            if name in saved_weights:
                weight.assign(saved_weights[name])
                print(f"Assigned {name}")
        
        print("Completed layer-by-layer loading attempt")
    
    # Now continue with the rest of the function to extract weights to Fortran
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
    
    # Open the output file
    with open(output_path, 'w') as f:
        # Write module header
        f.write("! Auto-generated Fortran module for piCANN weights\n")
        f.write(f"! Generated from model: {model_path}\n\n")
        f.write(f"module {module_name}\n")
        f.write("  use iso_fortran_env, only: real64\n")
        f.write("  implicit none\n\n")
        
        # Extract network sizes
        if hasattr(icann_cell, 'n'):
            n = icann_cell.n
            f.write("  ! Network sizes\n")
            f.write(f"  integer, parameter :: n_e = {n[0]}\n")
            f.write(f"  integer, parameter :: n_i = {n[1]}\n")
            f.write(f"  integer, parameter :: n_g1 = {n[2]}\n")
            f.write(f"  integer, parameter :: n_g2 = {n[3]}\n")
            f.write(f"  integer, parameter :: n_g3 = {n[4]}\n\n")
        
        # Add other configuration parameters
        f.write("  ! Configuration parameters\n")
        if hasattr(icann_cell, 'tolerance'):
            f.write(f"  real(real64), parameter :: tolerance = {float(icann_cell.tolerance.numpy()):.16e}_real64\n")
        if hasattr(icann_cell, 'lambda_dot'):
            f.write(f"  real(real64), parameter :: lambda_dot = {float(icann_cell.lambda_dot.numpy()):.16e}_real64\n\n")
        
        # Helper function to write weights - modified to avoid duplicates
        def write_weight_array(obj, name, path="", processed_weights=None):
            if processed_weights is None:
                processed_weights = set()
                
            if not hasattr(obj, 'weights') or not obj.weights:
                return processed_weights
            
            full_path = path + "." + name if path else name
            
            # Skip direct HelmholtzNet weights since they're redundant with subnet weights
            from src.models.icann.icann_3d_asymmetry import HelmholtzNet
            is_helmholtz_net = isinstance(obj, HelmholtzNet)
            
            # Process weights
            for weight in obj.weights:
                weight_name = weight.name.split('/')[-1].split(':')[0]
                weight_data = weight.numpy()
                
                # Skip direct HelmholtzNet weights that match subnet weight names
                # These are typically the cp1, cp2, cp3 weights
                if is_helmholtz_net and weight_name in ['cp1', 'cp2', 'cp3', 'coefficients', 'biases']:
                    continue
                
                # Create Fortran array name with unique identifier
                if isinstance(obj, HelmholtzIsoNet):
                    # Extract a unique identifier from the layer's name
                    layer_id = obj.name.split('_')[-1]  # This will get "iso1", "iso2"
                    fortran_name = f"{full_path}_{layer_id}_{weight_name}".replace('.', '_')
                elif isinstance(obj, HelmholtzVolNet):
                    # For volume network
                    layer_id = obj.name.split('_')[-1]  # This will get "vol"
                    fortran_name = f"{full_path}_{layer_id}_{weight_name}".replace('.', '_')
                else:
                    fortran_name = f"{full_path}_{weight_name}".replace('.', '_')
                
                # Check if we've already processed a weight with this name
                if fortran_name in processed_weights:
                    print(f"Skipping duplicate weight: {fortran_name}")
                    continue
                
                processed_weights.add(fortran_name)
                
                # Get shape for declaration
                shape_str = ", ".join([str(s) for s in weight_data.shape])
                
                # Write declaration with any constraints as comments
                f.write(f"  ! {weight.name}\n")
                if hasattr(weight, 'constraint') and weight.constraint is not None:
                    if isinstance(weight.constraint, tf.keras.constraints.NonNeg):
                        f.write(f"  ! This weight has non-negative constraint\n")
                    else:
                        f.write(f"  ! This weight has constraints\n")
                
                # Special case for empty shape (scalar)
                if not weight_data.shape:
                    f.write(f"  real(real64), parameter :: {fortran_name} = {float(weight_data):.16e}_real64\n\n")
                    continue
                
                f.write(f"  real(real64), parameter :: {fortran_name}({shape_str}) = &\n")
                
                # Format the data according to dimensionality
                MAX_LINE_LENGTH = 100  # Safe limit for Fortran free-form
                
                if len(weight_data.shape) == 1:  # 1D array
                    f.write("    [")
                    values_per_line = 3  # Limit values per line for readability
                    total_values = len(weight_data)
                    
                    for i in range(0, total_values, values_per_line):
                        # Get slice of values for this line
                        end_idx = min(i + values_per_line, total_values)
                        slice_values = weight_data[i:end_idx]
                        
                        # Format line with proper indentation
                        if i > 0:
                            f.write("     ")  # Indent continuation lines
                        
                        # Create the formatted line with all values in this slice
                        line = []
                        for val in slice_values:
                            line.append(f"{float(val):.16e}_real64")
                        line_str = ", ".join(line)
                        f.write(line_str)
                        
                        # Add line continuation if not the last line
                        if end_idx < total_values:
                            f.write(", &\n")
                    
                    f.write("]\n\n")
                
                elif len(weight_data.shape) == 2:  # 2D array
                    f.write("    reshape([\n")
                    line_length = 5  # Initial line length for each line
                    for i, row in enumerate(weight_data):
                        f.write("    ")
                        line_length = 4  # Reset for each row
                        for j, val in enumerate(row):
                            # Format the value
                            value_str = f"{float(val):.16e}_real64"
                            # Add comma if not first element
                            if i > 0 or j > 0:
                                value_str = ", " + value_str
                                line_length += 2  # Length of ", "
                            
                            # Check if adding this value would exceed line length
                            if line_length + len(value_str) > MAX_LINE_LENGTH and (i > 0 or j > 0):
                                f.write(" &\n     ")  # Line continuation and indent
                                line_length = 5  # Reset line length
                                # Remove leading comma and space from value_str if it exists
                                if value_str.startswith(", "):
                                    value_str = value_str[2:]
                            
                            f.write(value_str)
                            line_length += len(value_str)
                        
                        # Add line continuation at end of each row (except last)
                        if i < len(weight_data) - 1:
                            f.write(" &\n")
                        else:
                            f.write("\n")
                    
                    f.write("    ], [" + shape_str + "])\n\n")
                
                else:  # Higher dimensional arrays - flatten and use reshape
                    f.write("    reshape([\n")
                    flat_data = weight_data.flatten()
                    line_length = 5  # Initial line length for each line
                    
                    for i, val in enumerate(flat_data):
                        # Start a new line with indentation for each chunk
                        if i % 3 == 0:  # Reduced from 4 to 3 values per line for safety
                            if i > 0:
                                f.write(" &\n    ")
                            else:
                                f.write("    ")
                            line_length = 4  # Reset for new line
                        
                        # Format the value
                        value_str = f"{float(val):.16e}_real64"
                        # Add comma if not first element
                        if i > 0:
                            value_str = ", " + value_str
                            line_length += 2  # Length of ", "
                        
                        f.write(value_str)
                        line_length += len(value_str)
                    
                    f.write("\n    ], [" + shape_str + "])\n\n")
            
            return processed_weights
        
        # Track processed weights to avoid duplicates
        processed_weights = set()
        
        # Extract Helmholtz Networks
        for hnet_name in ['hNet_e', 'hNet_i']:
            if hasattr(icann_cell, hnet_name):
                hnet = getattr(icann_cell, hnet_name)
                
                # Extract sublayers first - these are the ones we want to keep
                for subnet_name in ['iso_net1', 'iso_net2', 'vol_net']:
                    if hasattr(hnet, subnet_name):
                        subnet = getattr(hnet, subnet_name)
                        processed_weights = write_weight_array(subnet, subnet_name, hnet_name, processed_weights)
                        
                        # For vol_net, also extract power_expansion
                        if subnet_name == 'vol_net' and hasattr(subnet, 'power_expansion'):
                            processed_weights = write_weight_array(
                                subnet.power_expansion, 'power_expansion', 
                                f"{hnet_name}.{subnet_name}", processed_weights
                            )
                
                # Now process the HelmholtzNet itself, which will skip the duplicate weights
                processed_weights = write_weight_array(hnet, hnet_name, "", processed_weights)
        
        # Extract Flow Potential Networks
        for gnet_idx in range(1, 4):
            gnet_name = f'gNet{gnet_idx}'
            if hasattr(icann_cell, gnet_name):
                gnet = getattr(icann_cell, gnet_name)
                processed_weights = write_weight_array(gnet, gnet_name, "", processed_weights)
                
                # Extract polynomial layer
                if hasattr(gnet, 'poly'):
                    processed_weights = write_weight_array(gnet.poly, 'poly', gnet_name, processed_weights)
        
        # Extract other weights like uniform_yield_weight
        for comp_name in ['uniform_yield_weight']:
            if hasattr(icann_cell, comp_name):
                comp = getattr(icann_cell, comp_name)
                if hasattr(comp, 'numpy'):  # It's a weight
                    val = comp.numpy()
                    f.write(f"  ! {comp_name}\n")
                    f.write(f"  real(real64), parameter :: {comp_name} = {float(val[0]):.16e}_real64\n\n")
        
        # Close the module
        f.write("end module " + module_name + "\n")
    
    print(f"Weights extracted to {output_path}")
    
    # Count the total number of parameters
    param_count = 0
    for layer in model.layers:
        if hasattr(layer, 'count_params'):
            param_count += layer.count_params()
    
    print(f"Total parameters: {param_count}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Extract piCANN weights to Fortran')
    # parser.add_argument('model_path', help='Path to the model weights directory or file')
    # parser.add_argument('--output', '-o', default='piCANN_weights.f90', help='Output Fortran file path')
    # parser.add_argument('--module', '-m', default='piCANN_weights', help='Fortran module name')
    
    # args = parser.parse_args()
    
    # Extract weights to Fortran
    model_path = "outputs/models/icann_staged_refactored"
    output_path = "src/fortran_ports/piCANN_weights.f90"
    module_name = "piCANN_weights"
    extract_weights_to_fortran(model_path, output_path, module_name)
    
    # print("\nExample usage in Fortran:")
    # print(f"  use {args.module}")
    # print("  ! Now you can access weights like:")
    # print("  ! hNet_e_iso_net1_cp1, gNet1_weights_g, uniform_yield_weight, etc.")