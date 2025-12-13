#!/usr/bin/env python3
"""
Comprehensive robustness evaluation script for ProtoViT on CUB-200-C.
Evaluates model performance across specified corruption types at severity 5.
Saves results iteratively to JSON for resumability.

Usage:
    python evaluate_robustness.py --model ./saved_models/best_model.pth \
                                   --data_dir ./datasets/cub200_c/ \
                                   --output ./robustness_results_sev5.json
"""

import os
import sys
import argparse
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import logging
from tqdm import tqdm

# Import ProtoViT modules
import model  # Necessary for torch.load
import train_and_test as tnt
from settings import img_size, test_dir, test_batch_size, k, sum_cls
from preprocess import mean, std
from noise_utils import get_corrupted_transform
import tent
import proto_entropy
import loss_adapt
import eata_adapt
import sar_adapt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

class Cfg:
    """Configuration for test-time adaptation methods."""
    def __init__(self):
        self.OPTIM = self.Optim()
        self.MODEL = self.Model()

    class Optim:
        def __init__(self):
            self.METHOD = 'Adam'
            self.LR = 0.001
            self.BETA = 0.9
            self.WD = 0.000
            self.STEPS = 1

    class Model:
        def __init__(self):
            self.EPISODIC = False

cfg = Cfg()


def setup_optimizer(params):
    """Set up optimizer for adaptation."""
    if cfg.OPTIM.METHOD == 'Adam':
        return torch.optim.Adam(params, lr=cfg.OPTIM.LR,
                               betas=(cfg.OPTIM.BETA, 0.999),
                               weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError


def setup_tent(model):
    """Set up Tent adaptation."""
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    return tent.Tent(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)


def setup_proto_entropy(model, use_importance=False, use_confidence=False, 
                        reset_mode=None, reset_frequency=10, 
                        confidence_threshold=0.7, ema_alpha=0.999,
                        use_geometric_filter=False, geo_filter_threshold=0.3,
                        consensus_strategy='max', consensus_ratio=0.5,
                        adaptation_mode='layernorm_only',
                        use_ensemble_entropy=False,
                        source_proto_stats=None, alpha_source_kl=0.0,
                        adapt_all_prototypes=False):
    """Set up Prototype Entropy adaptation (without threshold).
    
    Args:
        use_importance: Use prototype importance weighting
        use_confidence: Use confidence weighting
        reset_mode: 'episodic', 'periodic', 'confidence', 'hybrid', 'ema', 'none'
        reset_frequency: How often to reset in 'periodic'/'hybrid' modes (in BATCHES)
        confidence_threshold: Confidence threshold for 'confidence'/'hybrid' modes
        ema_alpha: EMA decay for 'ema' mode
        use_geometric_filter: Use geometric similarity to filter unreliable samples
        geo_filter_threshold: Minimum similarity to ANY prototype to be considered reliable
        consensus_strategy: How to aggregate sub-prototypes
        consensus_ratio: For 'top_k_mean', fraction of sub-prototypes to use
        adaptation_mode: What parameters to adapt
        use_ensemble_entropy: Treat sub-prototypes as ensemble
        source_proto_stats: Pre-computed source prototype statistics
        alpha_source_kl: Weight for source KL regularization
        adapt_all_prototypes: If True, adapt all prototypes (not just target)
    """
    model = proto_entropy.configure_model(model, adaptation_mode=adaptation_mode)
    params, param_names = proto_entropy.collect_params(model, adaptation_mode=adaptation_mode)
    
    optimizer = setup_optimizer(params)
    proto_model = proto_entropy.ProtoEntropy(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        use_prototype_importance=use_importance,
        use_confidence_weighting=use_confidence,
        reset_mode=reset_mode,
        reset_frequency=reset_frequency,
        confidence_threshold=confidence_threshold,
        ema_alpha=ema_alpha,
        use_geometric_filter=use_geometric_filter,
        geo_filter_threshold=geo_filter_threshold,
        consensus_strategy=consensus_strategy,
        consensus_ratio=consensus_ratio,
        use_ensemble_entropy=use_ensemble_entropy,
        source_proto_stats=source_proto_stats,
        alpha_source_kl=alpha_source_kl,
        adapt_all_prototypes=adapt_all_prototypes
    )
    return proto_model


def setup_loss_adapt(model):
    """Set up Loss-based adaptation."""
    model = loss_adapt.configure_model(model)
    params, param_names = loss_adapt.collect_params(model)
    optimizer = setup_optimizer(params)
    return loss_adapt.LossAdapt(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)


def setup_eata(model, fishers):
    """Set up EATA adaptation."""
    model = eata_adapt.configure_model(model)
    params, param_names = eata_adapt.collect_params(model)
    optimizer = setup_optimizer(params)
    return eata_adapt.EATA(model, optimizer, fishers=fishers, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)


def setup_sar(model):
    """Set up SAR adaptation."""
    model = sar_adapt.configure_model(model)
    params, param_names = sar_adapt.collect_params(model)
    # SAR uses SAM optimizer with SGD base
    base_optimizer = torch.optim.SGD
    optimizer = sar_adapt.SAM(params, base_optimizer, lr=cfg.OPTIM.LR, momentum=0.9)
    return sar_adapt.SAR(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)


def evaluate_model(model, loader, description="Inference", verbose=True):
    """Run evaluation and return accuracy."""
    if verbose:
        print(f'\n{description}...')
    
    class_specific = True
    accu, test_loss_dict = tnt.test(
        model=model, 
        dataloader=loader,
        class_specific=class_specific,
        log=print if verbose else lambda x: None,
        clst_k=k,
        sum_cls=sum_cls
    )
    
    if verbose:
        print(f'Accuracy: {accu*100:.2f}%')
    
    return accu


def load_corrupted_dataset(data_dir, corruption_type, severity, batch_size=None):
    """Load a pre-generated corrupted dataset."""
    if batch_size is None:
        batch_size = test_batch_size
    
    corruption_path = Path(data_dir) / corruption_type / str(severity)
    
    if not corruption_path.exists():
        raise FileNotFoundError(f"Corrupted dataset not found at {corruption_path}")
    
    # Simple transform: just resize, to tensor, and normalize
    # (corruption already applied in the saved images)
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    dataset = datasets.ImageFolder(corruption_path, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )
    
    return loader


def load_dataset_with_corruption(clean_data_dir, corruption_type, severity, batch_size=None):
    """Load clean dataset and apply corruption on-the-fly."""
    if batch_size is None:
        batch_size = test_batch_size
    
    transform = get_corrupted_transform(img_size, mean, std, corruption_type, severity)
    
    dataset = datasets.ImageFolder(clean_data_dir, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=False
    )
    
    return loader


def load_results_json(output_file):
    """Load existing results from JSON file."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Could not load existing results from {output_file}: {e}")
            logger.warning("Starting fresh...")
    return None


def save_results_json(output_file, results_dict, metadata=None):
    """Save results to JSON file."""
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {},
        'results': results_dict
    }
    
    # Create backup before writing
    if os.path.exists(output_file):
        backup_file = output_file + '.backup'
        try:
            import shutil
            shutil.copy2(output_file, backup_file)
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")
    
    # Write to temporary file first, then rename (atomic write)
    temp_file = output_file + '.tmp'
    try:
        with open(temp_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        os.replace(temp_file, output_file)
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise


def evaluate_single_combination(model_path, corruption_type, severity, data_dir, 
                               clean_data_dir, on_the_fly, mode_name, mode_config, 
                               device, batch_size, fishers=None):
    """Evaluate a single mode on a single corruption-severity combination.
    
    Args:
        mode_name: Name of the mode (e.g., 'normal', 'tent', 'proto_imp_conf_v1')
        mode_config: Dict with setup parameters for this mode
    """
    try:
        # Load data
        if on_the_fly:
            loader = load_dataset_with_corruption(clean_data_dir, corruption_type, severity, batch_size)
        else:
            loader = load_corrupted_dataset(data_dir, corruption_type, severity, batch_size)
    except Exception as e:
        logger.error(f"Failed to load data for {corruption_type}-{severity}: {e}")
        return None
    
    try:
        # Load fresh model
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        base_model.eval()
        
        # Setup adaptation based on mode
        if mode_name == 'normal':
            eval_model = base_model
        elif mode_name == 'tent':
            eval_model = setup_tent(base_model)
        elif mode_name.startswith('proto_imp_conf'):
            # ProtoEntropy with Importance+Confidence
            eval_model = setup_proto_entropy(
                base_model,
                use_importance=True,
                use_confidence=True,
                reset_mode=mode_config.get('reset_mode', None),
                reset_frequency=mode_config.get('reset_frequency', 10),
                confidence_threshold=mode_config.get('confidence_threshold', 0.7),
                ema_alpha=mode_config.get('ema_alpha', 0.999),
                use_geometric_filter=mode_config.get('use_geometric_filter', False),
                geo_filter_threshold=mode_config.get('geo_filter_threshold', 0.3),
                consensus_strategy=mode_config.get('consensus_strategy', 'max'),
                consensus_ratio=mode_config.get('consensus_ratio', 0.5),
                adaptation_mode=mode_config.get('adaptation_mode', 'layernorm_only'),
                use_ensemble_entropy=mode_config.get('use_ensemble_entropy', False),
                source_proto_stats=mode_config.get('source_proto_stats', None),
                alpha_source_kl=mode_config.get('alpha_source_kl', 0.0)
            )
            
            # Reset geo filter stats if applicable
            if mode_config.get('use_geometric_filter', False) and hasattr(eval_model, 'reset_geo_filter_stats'):
                eval_model.reset_geo_filter_stats()
        elif mode_name == 'loss':
            eval_model = setup_loss_adapt(base_model)
        elif mode_name == 'eata':
            if fishers is None:
                # Compute Fishers on test data (source-free)
                fisher_model = torch.load(model_path, weights_only=False)
                fisher_model = fisher_model.to(device)
                fisher_model = eata_adapt.configure_model(fisher_model)
                current_fishers = eata_adapt.compute_fishers(fisher_model, loader, device, num_samples=500)
                del fisher_model
                torch.cuda.empty_cache()
            else:
                current_fishers = fishers
            
            if current_fishers is None:
                raise ValueError("Fishers could not be computed for EATA")
            eval_model = setup_eata(base_model, current_fishers)
        elif mode_name == 'sar':
            eval_model = setup_sar(base_model)
        else:
            logger.warning(f"Unknown mode: {mode_name}")
            return None
        
        # Evaluate
        acc = evaluate_model(eval_model, loader, 
                           description=f"{mode_name} on {corruption_type}-{severity}",
                           verbose=False)
        
        # Cleanup
        del eval_model
        del base_model
        torch.cuda.empty_cache()
        
        return float(acc)
        
    except Exception as e:
        logger.error(f"Failed to evaluate {mode_name} on {corruption_type}-{severity}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive robustness evaluation on CUB-200-C (Severity 5)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Model and data paths
    parser.add_argument('--model', type=str, 
                       default='./saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth',
                       help='Path to saved model')
    parser.add_argument('--data_dir', type=str,
                       default='./datasets/cub200_c/',
                       help='Path to pre-generated CUB-C dataset')
    parser.add_argument('--clean_data_dir', type=str,
                       default=None,
                       help='Path to clean test data (for on-the-fly). Default: uses test_dir from settings.')
    
    # Data loading
    parser.add_argument('--on_the_fly', action='store_true',
                       help='Generate corruptions on-the-fly instead of loading pre-generated')
    parser.add_argument('--batch_size', type=int, default=None,
                       help=f'Batch size for evaluation (default: {test_batch_size} from settings)')
    
    # Output
    parser.add_argument('--output', type=str, default='./robustness_results_sev5.json',
                       help='Path to save results JSON')
    
    # Hardware
    parser.add_argument('--gpuid', type=str, default='0',
                       help='GPU ID to use')
    
    # EATA settings
    parser.add_argument('--use_clean_fisher', action='store_true', default=False,
                       help='Use clean data to compute Fisher Information Matrix for EATA. Default is False (use test data).')
    
    args = parser.parse_args()
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Use clean_data_dir from args or fallback to settings
    clean_data_dir = args.clean_data_dir if args.clean_data_dir else test_dir
    batch_size = args.batch_size if args.batch_size else test_batch_size
    
    # Verify model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    # Define corruption types (severity 5 only)
    corruption_types = [
        'gaussian_noise', 'fog', 'gaussian_blur', 'elastic_transform', 
        'brightness', 'jpeg_compression', 'contrast', 'defocus_blur', 
        'frost', 'impulse_noise', 'pixelate', 'saturate', 'shot_noise', 
        'spatter', 'speckle_noise'
    ]
    severity = 5
    
    # Define modes with their configurations
    # proto_imp_conf has 3 variations
    modes = {
        'normal': {},
        'tent': {},
        'proto_imp_conf_v1': {  # Full config
            'use_geometric_filter': True,
            'geo_filter_threshold': 0.92,
            'consensus_strategy': 'top_k_mean',
            'adaptation_mode': 'layernorm_attn_bias',
            'use_ensemble_entropy': True,
            'reset_mode': None,
            'reset_frequency': 10,
            'confidence_threshold': 0.7,
            'ema_alpha': 0.999,
            'consensus_ratio': 0.5,
        },
        'proto_imp_conf_v2': {  # Without layernorm_attn_bias (default layernorm_only)
            'use_geometric_filter': True,
            'geo_filter_threshold': 0.92,
            'consensus_strategy': 'top_k_mean',
            'adaptation_mode': 'layernorm_only',  # Default
            'use_ensemble_entropy': True,
            'reset_mode': None,
            'reset_frequency': 10,
            'confidence_threshold': 0.7,
            'ema_alpha': 0.999,
            'consensus_ratio': 0.5,
        },
        'proto_imp_conf_v3': {  # Without use_ensemble_entropy
            'use_geometric_filter': True,
            'geo_filter_threshold': 0.92,
            'consensus_strategy': 'top_k_mean',
            'adaptation_mode': 'layernorm_attn_bias',
            'use_ensemble_entropy': False,  # Disabled
            'reset_mode': None,
            'reset_frequency': 10,
            'confidence_threshold': 0.7,
            'ema_alpha': 0.999,
            'consensus_ratio': 0.5,
        },
        'loss': {},
        'eata': {},
        'sar': {},
    }
    
    # Compute Fishers GLOBAL if requested (Clean Data Access)
    fishers = None
    if 'eata' in modes and args.use_clean_fisher:
        print("\n" + "="*80)
        print("Computing Fisher Information Matrix on CLEAN data for EATA")
        print("="*80)
        
        # Load model
        base_model = torch.load(args.model, weights_only=False)
        base_model = base_model.to(device)
        
        # Load clean data (subset)
        transform_clean = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        clean_dataset = datasets.ImageFolder(clean_data_dir, transform_clean)
        
        num_fisher_samples = 2000
        if len(clean_dataset) > num_fisher_samples:
            fisher_indices = torch.randperm(len(clean_dataset))[:num_fisher_samples]
            fisher_subset = torch.utils.data.Subset(clean_dataset, fisher_indices)
        else:
            fisher_subset = clean_dataset
            
        fisher_loader = torch.utils.data.DataLoader(
            fisher_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
        )
        
        # Configure and compute
        base_model = eata_adapt.configure_model(base_model)
        fishers = eata_adapt.compute_fishers(base_model, fisher_loader, device)
        print("Fisher information computed on CLEAN data.")
        
        del base_model
        torch.cuda.empty_cache()
    
    # Load existing results if available
    existing_results = load_results_json(args.output)
    if existing_results:
        results_dict = existing_results.get('results', {})
        print(f"Loaded existing results from {args.output}")
        print(f"  Found {len(results_dict)} modes with partial results")
    else:
        results_dict = {}
    
    # Initialize results structure
    for mode_name in modes.keys():
        if mode_name not in results_dict:
            results_dict[mode_name] = {}
        for corruption_type in corruption_types:
            if corruption_type not in results_dict[mode_name]:
                results_dict[mode_name][corruption_type] = {}
    
    # Print configuration
    print("="*80)
    print("ROBUSTNESS EVALUATION CONFIGURATION")
    print("="*80)
    print(f"Model:           {args.model}")
    print(f"Data directory:  {args.data_dir if not args.on_the_fly else 'On-the-fly generation'}")
    print(f"Clean data dir:  {clean_data_dir}")
    print(f"Corruptions:     {len(corruption_types)} types")
    print(f"Severity:        {severity}")
    print(f"Modes:           {', '.join(modes.keys())}")
    print(f"Batch size:      {batch_size}")
    print(f"Output file:     {args.output}")
    print("="*80)
    
    # Create list of all combinations to evaluate
    all_combinations = []
    for mode_name in modes.keys():
        for corruption_type in corruption_types:
            # Check if already completed
            if (corruption_type in results_dict[mode_name] and 
                severity in results_dict[mode_name][corruption_type] and
                results_dict[mode_name][corruption_type][severity] is not None):
                continue  # Skip already completed
            all_combinations.append((mode_name, corruption_type))
    
    total_combinations = len(all_combinations)
    print(f"\nTotal combinations to evaluate: {total_combinations}")
    if total_combinations == 0:
        print("All combinations already completed!")
        return
    
    # Evaluate each combination with progress bar
    start_time = time.time()
    
    pbar = tqdm(all_combinations, desc="Evaluating", unit="comb")
    for mode_name, corruption_type in pbar:
        pbar.set_description(f"Evaluating {mode_name} on {corruption_type}")
        
        acc = evaluate_single_combination(
            args.model,
            corruption_type,
            severity,
            args.data_dir,
            clean_data_dir,
            args.on_the_fly,
            mode_name,
            modes[mode_name],
            device,
            batch_size,
            fishers=fishers
        )
        
        # Update results
        results_dict[mode_name][corruption_type][severity] = acc
        
        # Save after each combination (iterative saving)
        try:
            metadata = {
                'model_path': args.model,
                'data_dir': args.data_dir,
                'clean_data_dir': clean_data_dir,
                'on_the_fly': args.on_the_fly,
                'batch_size': batch_size,
                'severity': severity,
                'corruption_types': corruption_types,
                'modes': list(modes.keys()),
                'mode_configs': modes
            }
            save_results_json(args.output, results_dict, metadata)
            
            # Update progress bar with accuracy
            if acc is not None:
                pbar.set_postfix({'acc': f'{acc*100:.2f}%', 'saved': '✓'})
            else:
                pbar.set_postfix({'acc': 'FAILED', 'saved': '✓'})
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            pbar.set_postfix({'acc': 'ERROR', 'saved': '✗'})
    
    pbar.close()
    
    # Final summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Results saved to: {args.output}")
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Header
    header = f"{'Corruption':<25s}"
    for mode_name in modes.keys():
        header += f" {mode_name[:15]:>15s}"
    print(header)
    print("-" * (25 + 16 * len(modes)))
    
    # Results for each corruption
    for corruption_type in corruption_types:
        line = f"{corruption_type:<25s}"
        for mode_name in modes.keys():
            if (corruption_type in results_dict[mode_name] and 
                severity in results_dict[mode_name][corruption_type]):
                acc = results_dict[mode_name][corruption_type][severity]
                if acc is not None:
                    line += f" {acc*100:14.2f}%"
                else:
                    line += f" {'N/A':>15s}"
            else:
                line += f" {'N/A':>15s}"
        print(line)
    
    print("="*80)


if __name__ == '__main__':
    main()
