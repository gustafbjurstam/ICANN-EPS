import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt


# Add project root to Python path for imports from any location
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.icann.icann_3d_asymmetry import create_rnn_model, piCANN

from src.models.icann.load_model_utils import (
    create_and_compile_model, 
    save_model, 
    create_callbacks, 
    plot_training_history,
    load_training_data,
    visualize_strain_stress,
    NaNDetectionCallback,
    SmartReduceLROnPlateau,
    load_matlab_data,
    stress_mse_loss, 
    dummy_loss,
    prepare_truncated_data,
    copy_weights_from_hNet_e_to_hNet_i,
    check_for_nans,
    freeze_nets,
    extract_time_series,
    load_checkpoint
)

def train_staged_icann_refactored(resume_from_stage=None, save_path="outputs/models/icann_staged_refactored"):
    """
    Train the piCANN model in stages using a configuration-based approach.
    
    Args:
        resume_from_stage: Stage to resume training from (1-based indexing). If None, start from beginning.
        save_path: Directory to save model and results
        
    Returns:
        Trained model or None if training failed
    """
    # Set parameters
    data_path = "data/simple_paths"  # Or use matlab files
    n = [1, 1, 1, 1, 1]  # Network sizes
    seed = 1234
    
    # Define stage configuration
    # This config for normal yield surface
    stage_configs = [
        {
            "name": "Initial elastic stage with hNet_e",
            "epochs": 1000,
            "lr": 5e-3,
            "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "data_percentage": 8,
            "data_series": -1,  # Use all data
            "frozen_nets": ['hNet_i', 'gNet1', 'gNet2', 'gNet3'],
            "unfrozen_nets": ['hNet_e'],
            "patience": 40,
            "reduce_lr_patience": 15,
            "copy_weights": True,
            "copy_scale_factor": 0.0002
        },
        {
            "name": "Initial elastic stage with hNet_e",
            "epochs": 3000,
            "lr": 5e-3,
            "loss_weights": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "data_percentage": 8,
            "data_series": -1,  # Use all data
            "frozen_nets": ['hNet_i', 'gNet1', 'gNet2', 'gNet3'],
            "unfrozen_nets": ['hNet_e'],
            "patience": 40,
            "reduce_lr_patience": 15,
            "copy_weights": True,
            "copy_scale_factor": 0.0001
        },
        {
            "name": "Yield weight calibration",
            "epochs": 1000,
            "lr": 1e-2,
            "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "data_percentage": 20,
            "data_series": 0,  # Uniaxial data
            "frozen_nets": ['hNet_e', 'hNet_i', 'gNet1', 'gNet2', 'gNet3'],
            "unfrozen_nets": ['uniform_yield_weight'],
            "patience": 20,
            "reduce_lr_patience": 8,
            "copy_weights": False
        },
        {
            "name": "gNets training",
            "epochs": 1000,
            "lr": 1e-4,
            "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "data_percentage": 30,
            "data_series": 0,  # Uniaxial data
            "frozen_nets": ['hNet_e', 'hNet_i'],
            "unfrozen_nets": ['gNet1', 'gNet2', 'gNet3'],
            "patience": 40, 
            "reduce_lr_patience": 15,
            "copy_weights": False
        },
        {
            "name": "hNet_i and hNet_e training",
            "epochs": 2000,
            "lr": 2e-3,
            "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "data_percentage": 94,
            "data_series": 0,  # Uniaxial data
            "frozen_nets": ['gNet1', 'gNet2', 'gNet3'],
            "unfrozen_nets": ['hNet_i', 'hNet_e'],
            "patience": 60,
            "reduce_lr_patience": 20,
            "copy_weights": False
        },
        {
            "name": "hNet_i and hNet_e training",
            "epochs": 500,
            "lr": 1e-2,
            "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "data_percentage": 94,
            "data_series": 0,  # Uniaxial data
            "frozen_nets": ['gNet1', 'gNet2', 'gNet3'],
            "unfrozen_nets": ['hNet_i', 'hNet_e'],
            "patience": 60,
            "reduce_lr_patience": 20,
            "copy_weights": False
        },
        {
            "name": "Initial elastic stage with hNet_e",
            "epochs": 1000,
            "lr": 5e-3,
            "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "data_percentage": 8,
            "data_series": -1,  # Use all data
            "frozen_nets": ['hNet_i', 'gNet1', 'gNet2', 'gNet3'],
            "unfrozen_nets": ['hNet_e'],
            "patience": 40,
            "reduce_lr_patience": 15,
            "copy_weights": False
        },
        {
            "name": "Initial elastic stage with hNet_e",
            "epochs": 1000,
            "lr": 5e-3,
            "loss_weights": [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            "data_percentage": 8,
            "data_series": 0,  # Uniaxial data
            "frozen_nets": ['hNet_i', 'gNet1', 'gNet2', 'gNet3'],
            "unfrozen_nets": ['hNet_e'],
            "patience": 40,
            "reduce_lr_patience": 15,
            "copy_weights": False
        },
        {
            "name": "Initial elastic stage with hNet_e",
            "epochs": 700,
            "lr": 5e-3,
            "loss_weights": [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            "data_percentage": 8,
            "data_series": -1,  # All data
            "frozen_nets": ['hNet_i', 'gNet1', 'gNet2', 'gNet3'],
            "unfrozen_nets": ['hNet_e'],
            "patience": 40,
            "reduce_lr_patience": 15,
            "copy_weights": False
        },
        {
            "name": "Full training",
            "epochs": 1000,
            "lr": 1e-4,
            "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "data_percentage": 94,
            "data_series": 0,  # Uniaxial data
            "frozen_nets": [],
            "unfrozen_nets": ['hNet_e', 'hNet_i', 'gNet1', 'gNet2', 'gNet3'],
            "patience": 80,
            "reduce_lr_patience": 25,
            "copy_weights": False
        }
    ]
    # This config for cubic yield surface
    # stage_configs = [
    #     {
    #         "name": "Initial elastic stage with hNet_e",
    #         "epochs": 1000,
    #         "lr": 5e-3,
    #         "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         "data_percentage": 8,
    #         "data_series": -1,  # Use all data
    #         "frozen_nets": ['hNet_i', 'gNet1', 'gNet2', 'gNet3','uniform_yield_weight'],
    #         "unfrozen_nets": ['hNet_e'],
    #         "patience": 40,
    #         "reduce_lr_patience": 15,
    #         "copy_weights": True,
    #         "copy_scale_factor": 0.0002
    #     },
    #     {
    #         "name": "Initial elastic stage with hNet_e",
    #         "epochs": 3000,
    #         "lr": 1e-2,
    #         "loss_weights": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    #         "data_percentage": 8,
    #         "data_series": -1,  # Use all data
    #         "frozen_nets": ['hNet_i', 'gNet1', 'gNet2', 'gNet3','uniform_yield_weight'],
    #         "unfrozen_nets": ['hNet_e'],
    #         "patience": 40,
    #         "reduce_lr_patience": 15,
    #         "copy_weights": True,
    #         "copy_scale_factor": 0.0001
    #     },
    #     {
    #         "name": "Yield weight calibration",
    #         "epochs": 1000,
    #         "lr": 1e-2,
    #         "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         "data_percentage": 20,
    #         "data_series": 0,  # Uniaxial data
    #         "frozen_nets": ['hNet_e', 'hNet_i', 'gNet1', 'gNet2', 'gNet3'],
    #         "unfrozen_nets": ['uniform_yield_weight'],
    #         "patience": 20,
    #         "reduce_lr_patience": 8,
    #         "copy_weights": False
    #     },
    #     {
    #         "name": "hNet_i and hNet_e training",
    #         "epochs": 400,
    #         "lr": 1e-2,
    #         "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         "data_percentage": 94,
    #         "data_series": 0,  # Uniaxial data
    #         "frozen_nets": ['gNet1', 'gNet2', 'gNet3'],
    #         "unfrozen_nets": ['hNet_i', 'hNet_e'],
    #         "patience": 60,
    #         "reduce_lr_patience": 20,
    #         "copy_weights": False
    #     },
    #     {
    #         "name": "hNet_i and hNet_e training",
    #         "epochs": 500,
    #         "lr": 1e-2,
    #         "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         "data_percentage": 94,
    #         "data_series": 0,  # Uniaxial data
    #         "frozen_nets": ['gNet1', 'gNet2', 'gNet3'],
    #         "unfrozen_nets": ['hNet_i', 'hNet_e'],
    #         "patience": 60,
    #         "reduce_lr_patience": 20,
    #         "copy_weights": False
    #     },
    #     {
    #         "name": "Initial elastic stage with hNet_e",
    #         "epochs": 1000,
    #         "lr": 5e-3,
    #         "loss_weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         "data_percentage": 8,
    #         "data_series": -1,  # Use all data
    #         "frozen_nets": ['hNet_i', 'gNet1', 'gNet2', 'gNet3'],
    #         "unfrozen_nets": ['hNet_e'],
    #         "patience": 40,
    #         "reduce_lr_patience": 15,
    #         "copy_weights": False
    #     },
    #     {
    #         "name": "Initial elastic stage with hNet_e",
    #         "epochs": 1000,
    #         "lr": 5e-3,
    #         "loss_weights": [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    #         "data_percentage": 8,
    #         "data_series": 0,  # Uniaxial data
    #         "frozen_nets": ['hNet_i', 'gNet1', 'gNet2', 'gNet3'],
    #         "unfrozen_nets": ['hNet_e'],
    #         "patience": 40,
    #         "reduce_lr_patience": 15,
    #         "copy_weights": False
    #     },
    #     {
    #         "name": "Initial elastic stage with hNet_e",
    #         "epochs": 700,
    #         "lr": 5e-3,
    #         "loss_weights": [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
    #         "data_percentage": 8,
    #         "data_series": -1,  # All data
    #         "frozen_nets": ['hNet_i', 'gNet1', 'gNet2', 'gNet3'],
    #         "unfrozen_nets": ['hNet_e'],
    #         "patience": 40,
    #         "reduce_lr_patience": 15,
    #         "copy_weights": False
    #     },
    #     {
    #         "name": "Full training",
    #         "epochs": 1000,
    #         "lr": 1e-4,
    #         "loss_weights": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    #         "data_percentage": 94,
    #         "data_series": 0,  # Uniaxial data
    #         "frozen_nets": [],
    #         "unfrozen_nets": ['hNet_e', 'hNet_i', 'gNet1', 'gNet2', 'gNet3'],
    #         "patience": 80,
    #         "reduce_lr_patience": 25,
    #         "copy_weights": False
    #     },
    #     {
    #         "name": "Full training",
    #         "epochs": 500,
    #         "lr": 1e-3,
    #         "loss_weights": [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    #         "data_percentage": 100,
    #         "data_series": 0,  # Uniaxial data
    #         "frozen_nets": [],
    #         "unfrozen_nets": ['hNet_e', 'hNet_i', 'gNet1', 'gNet2', 'gNet3'],
    #         "patience": 80,
    #         "reduce_lr_patience": 25,
    #         "copy_weights": False
    #     }
    # ]
    
    all_nets = ['hNet_e', 'hNet_i', 'gNet1', 'gNet2', 'gNet3']
    
    # Create save directory structure
    os.makedirs(save_path, exist_ok=True)
    stage_dirs = []
    for i in range(len(stage_configs)):
        stage_dir = os.path.join(save_path, f'stage{i+1}')
        stage_dirs.append(stage_dir)
        os.makedirs(stage_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    # Load training data
    print("Loading training data...")
    # Option 1: Load from NPZ files
    variant_range = [0, 2]
    # variant_range = [0]
    train_inputs, train_outputs, x_train, y_train = load_training_data(
        data_path=data_path, variant_range=variant_range
    )
    
    # Option 2: Load from MATLAB files (uncomment if needed)
    # matlab_files = [
    #     "data/matlab_data/new_uniaxial_C_PK2.mat",
    #     "data/matlab_data/new_biaxial_C_PK2.mat",
    #     "data/matlab_data/new_pure_shear_C_PK2.mat"
    # ]
    # train_inputs, train_outputs, x_train, y_train = load_matlab_data(
    #     matlab_files, dt_value=1.0, normalize=True
    # )
    
    # Determine starting stage and initialize model
    start_stage_idx = 0
    model = None
    
    if resume_from_stage is not None:
        # We want to resume after this stage, so start from the next stage
        completed_stage = resume_from_stage
        start_stage_idx = completed_stage  # Start from the next stage
        
        if start_stage_idx <= 0 or start_stage_idx >= len(stage_configs):
            print(f"Invalid resume stage: {resume_from_stage}. Must be between 1 and {len(stage_configs)-1}")
            return None
        
        # Try to load the model from the completed stage
        model = load_checkpoint(save_path, completed_stage, n=n, seed=seed)
        
        if model is None:
            print(f"Failed to load model from stage {completed_stage}. Starting from beginning.")
            start_stage_idx = 0
        else:
            print(f"Successfully loaded model from stage {completed_stage}. Starting with stage {start_stage_idx+1}")
    
    # If not resuming or resume failed, create a new model
    if model is None:
        model = create_and_compile_model(n, stage_configs[0]["lr"], seed=seed)
        print("Initializing model weights with a dummy forward pass...")
        dummy_input_shape = {k: v[:1, :1, :] for k, v in train_inputs.items()}
        _ = model(dummy_input_shape, training=False)
        print("Model weights initialized successfully!")
    
    # Store histories for combined plot
    stage_histories = []
    
    # Train through all stages, starting from the specified stage
    for stage_idx in range(start_stage_idx, len(stage_configs)):
        config = stage_configs[stage_idx]
        stage_num = stage_idx + 1
        print("\n" + "="*80)
        print(f"STAGE {stage_num}: {config['name']}")
        print("="*80)
        
        # Freeze/unfreeze networks as specified in config
        model = freeze_nets(model, 
                           frozen_nets=config["frozen_nets"], 
                           unfrozen_nets=config["unfrozen_nets"])
        
        # Compile model with stage-specific parameters
        optimizer = keras.optimizers.SGD(learning_rate=config["lr"], clipvalue=1.0)
        model.compile(
            optimizer=optimizer,
            loss=[stress_mse_loss, stress_mse_loss, stress_mse_loss, 
                  dummy_loss, dummy_loss, dummy_loss],
            loss_weights=config["loss_weights"]
        )
        
        # Prepare data for this stage
        if config["data_series"] >= 0:
            # Extract specific time series
            stage_inputs_raw = extract_time_series(train_inputs, config["data_series"])
            stage_outputs_raw = extract_time_series(train_outputs, config["data_series"])
        else:
            # Use all data
            stage_inputs_raw = train_inputs
            stage_outputs_raw = train_outputs
        
        # Truncate data according to percentage
        stage_inputs, stage_outputs = prepare_truncated_data(
            stage_inputs_raw, stage_outputs_raw, percentage=config["data_percentage"]
        )
        
        # Create callbacks
        stage_callbacks, stage_logs_dir = create_callbacks(
            stage_dirs[stage_idx], 
            patience=config["patience"], 
            reduce_lr_patience=config["reduce_lr_patience"]
        )
        
        # Train the model
        print(f"\nTraining Stage {stage_num} with learning rate {config['lr']}")
        stage_history = model.fit(
            x=stage_inputs,
            y=stage_outputs,
            epochs=config["epochs"],
            callbacks=stage_callbacks,
            batch_size=1,
            verbose=0
        )
        
        # Store history for combined plot
        stage_histories.append(stage_history)
        
        # Unfreeze all networks for saving and visualization
        model = freeze_nets(model, unfrozen_nets=all_nets)
        
        # Copy weights if specified
        if config.get("copy_weights", False):
            model = copy_weights_from_hNet_e_to_hNet_i(
                model, scale_factor=config.get("copy_scale_factor", 0.0)
            )
        
        # Save model and visualize results
        save_model(model, stage_dirs[stage_idx], stage_history, config["lr"])
        plot_training_history(
            stage_history, 
            stage_logs_dir, 
            title=f"Stage {stage_num}: {config['name']}"
        )
        visualize_strain_stress(model, x_train, y_train, stage_dirs[stage_idx])
        
        # Check for NaNs after each stage 
        print("\nChecking for NaNs in predictions on current dataset...")
        has_nans = check_for_nans(model, stage_inputs)
        if has_nans:
            print("NaNs detected, terminating training to reinitialize weights.")
            return None
        
        # Additional validation for specific stages
        if stage_num == 2 and stage_history.history['loss'][-1] > 0.1:
            print(f"Loss after Stage 2 is too high: {stage_history.history['loss'][-1]:.6f} > 0.1")
            print("Aborting training as the yield weight wasn't properly calibrated.")
            return None
        
        if stage_num == 3 and stage_history.history['loss'][-1] > 0.1:
            print(f"Loss after Stage 3 is too high: {stage_history.history['loss'][-1]:.6f} > 0.1")
            print("Aborting training as the gNets weren't properly trained.")
            return None
    
    # Save final model
    final_model = model
    save_model(final_model, save_path, stage_histories[-1], config["lr"])
    visualize_strain_stress(final_model, x_train, y_train, save_path)
    
    # Create combined training history plot with all stages
    plt.figure(figsize=(12, 8))
    
    # Plot loss for all stages with proper offset on x-axis
    offset = 0
    colors = ['b', 'm', 'g', 'r', 'c']
    
    for i, history in enumerate(stage_histories):
        stage_index = i + start_stage_idx
        plt.semilogy(range(offset, offset + len(history.history['loss'])),
                    history.history['loss'], f'{colors[i % len(colors)]}-', 
                    label=f'Stage {stage_index+1}: {stage_configs[stage_index]["name"]}')
        offset += len(history.history['loss'])
    
    plt.title('Combined Training Loss Across All Stages')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'combined_training_loss.png'))
    
    print("\n" + "="*80)
    print("Staged training complete!")
    print(f"Final model saved to {save_path}")
    print("="*80)
    
    # Save parameters for future reference
    with open(os.path.join(save_path, 'model_params.json'), 'w') as f:
        json.dump({
            'n': n,
            'seed': seed
        }, f, indent=2)
    
    return final_model

if __name__ == "__main__":
    trained_model = None
    attempts = 0
    max_attempts = 15
    resume_stage = None # Last completed stage
    while trained_model is None:
        attempts += 1
        print(f"Attempt {attempts}/{max_attempts} to train staged ICANN model...")
        
        # Resume only applies to the first attempt - subsequent attempts start from scratch
        # resume_stage = args.resume if attempts == 1 else None
        if resume_stage:
            print(f"Resuming training after stage {resume_stage}")
        
        trained_model = train_staged_icann_refactored(
            resume_from_stage=resume_stage,
            save_path="outputs/models/icann_staged_refactored_test_new_file"
        )
        
        if attempts >= max_attempts and trained_model is None:
            # If we reach max attempts and still have no model, exit
            print("Max attempts reached. Exiting.")
            exit()
    
    print("Training completed successfully on attempt:", attempts)