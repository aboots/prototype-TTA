#!/usr/bin/env python3
"""
Comprehensive robustness evaluation script for ProtoViT on CUB-200-C.
Evaluates model performance across all corruption types and severity levels.

Usage:
    # Evaluate on pre-generated CUB-C dataset
    python evaluate_robustness.py --model ./saved_models/best_model.pth \
                                   --data_dir ./datasets/cub200_c/ \
                                   --mode all
    
    # Evaluate specific corruptions
    python evaluate_robustness.py --model ./saved_models/best_model.pth \
                                   --corruptions gaussian_noise shot_noise \
                                   --mode normal,tent
    
    # Generate corruptions on-the-fly (slower but saves disk space)
    python evaluate_robustness.py --model ./saved_models/best_model.pth \
                                   --on_the_fly \
                                   --clean_data_dir ./datasets/cub200_cropped/test_cropped/
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

# Import ProtoViT modules
import model  # Necessary for torch.load
import train_and_test as tnt
from settings import img_size, k, sum_cls
from preprocess import mean, std
from noise_utils import get_all_corruption_types, get_corrupted_transform
import tent
import proto_entropy
import loss_adapt
import fisher_proto
import eata_adapt
import memo_adapt
import sar_adapt


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def setup_proto_entropy(model, alpha_target=1.0, alpha_separation=0.5, alpha_coherence=0.3,
                       use_prototype_importance=False):
    """Set up ProtoEntropy adaptation (without threshold)."""
    model = proto_entropy.configure_model(model)
    params, param_names = proto_entropy.collect_params(model)
    optimizer = setup_optimizer(params)
    return proto_entropy.ProtoEntropy(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC,
                                     alpha_target=alpha_target, alpha_separation=alpha_separation, 
                                     alpha_coherence=alpha_coherence,
                                     use_prototype_importance=use_prototype_importance)


def setup_proto_entropy_eata(model, entropy_threshold=0.4):
    """Set up ProtoEntropy adaptation with EATA-style thresholding."""
    model = proto_entropy.configure_model(model)
    params, param_names = proto_entropy.collect_params(model)
    optimizer = setup_optimizer(params)
    return proto_entropy.ProtoEntropyEATA(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC, entropy_threshold=entropy_threshold)


def setup_loss_adapt(model):
    """Set up Loss-based adaptation."""
    model = loss_adapt.configure_model(model)
    params, param_names = loss_adapt.collect_params(model)
    optimizer = setup_optimizer(params)
    return loss_adapt.LossAdapt(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)


def setup_fisher_proto(model):
    """Set up Fisher-guided prototype adaptation."""
    model = fisher_proto.configure_model(model)
    params, param_names = fisher_proto.collect_params(model)
    optimizer = setup_optimizer(params)
    fisher_model = fisher_proto.FisherProto(model, optimizer, steps=cfg.OPTIM.STEPS, episodic=cfg.MODEL.EPISODIC)
    device = next(model.parameters()).device
    return fisher_model.to(device)


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


def setup_memo(model, lr=0.00025, batch_size=64, steps=1):
    """Set up MEMO adaptation.
    
    MEMO (Test Time Robustness via Adaptation and Augmentation) adapts
    the model to each test sample by minimizing the entropy of the marginal
    distribution over multiple augmented views.
    
    Args:
        model: The model to adapt
        lr: Learning rate for adaptation (default: 0.00025)
        batch_size: Number of augmented views per step (default: 64)
        steps: Number of adaptation steps per sample (default: 1)
    
    Returns:
        MEMO-wrapped model
    """
    model = memo_adapt.configure_model(model)
    params, param_names = memo_adapt.collect_params(model)
    
    # MEMO uses SGD optimizer with momentum
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)
    
    return memo_adapt.MEMO(model, optimizer, steps=steps, 
                          batch_size=batch_size, episodic=True)



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


def load_corrupted_dataset(data_dir, corruption_type, severity, batch_size=128):
    """Load a pre-generated corrupted dataset."""
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
        num_workers=4,
        pin_memory=True
    )
    
    return loader


def load_dataset_with_corruption(clean_data_dir, corruption_type, severity, batch_size=128):
    """Load clean dataset and apply corruption on-the-fly."""
    transform = get_corrupted_transform(img_size, mean, std, corruption_type, severity)
    
    dataset = datasets.ImageFolder(clean_data_dir, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return loader


def evaluate_corruption(model_path, corruption_type, severity, data_dir, 
                       clean_data_dir, on_the_fly, modes, device, batch_size, fishers=None, proto_threshold=None):
    """Evaluate model on a single corruption type and severity."""
    results = {}
    
    # Load data
    try:
        if on_the_fly:
            loader = load_dataset_with_corruption(clean_data_dir, corruption_type, severity, batch_size)
        else:
            loader = load_corrupted_dataset(data_dir, corruption_type, severity, batch_size)
    except Exception as e:
        logger.error(f"Failed to load data for {corruption_type}-{severity}: {e}")
        return None

    # If EATA is requested, compute Fishers on this test data (source-free setting)
    # We compute it once per corruption-severity setting using the current loader.
    if 'eata' in modes and fishers is None:
        try:
            # Load fresh model for Fisher computation
            fisher_model = torch.load(model_path, weights_only=False)
            fisher_model = fisher_model.to(device)
            fisher_model = eata_adapt.configure_model(fisher_model)
            
            # Compute Fishers on first 2000 samples of the current loader
            # Note: We pass the loader directly; compute_fishers handles early stopping
            current_fishers = eata_adapt.compute_fishers(fisher_model, loader, device, num_samples=2000)
            
            del fisher_model
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to compute Fishers for {corruption_type}-{severity}: {e}")
            current_fishers = None
    else:
        current_fishers = fishers
    
    # Evaluate each mode
    for mode in modes:
        try:
            # Load fresh model
            base_model = torch.load(model_path, weights_only=False)
            base_model = base_model.to(device)
            base_model.eval()
            
            # Setup adaptation if needed
            if mode == 'normal':
                eval_model = base_model
            elif mode == 'tent':
                eval_model = setup_tent(base_model)
            elif mode == 'proto':
                eval_model = setup_proto_entropy(base_model)
            elif mode == 'proto_eata':
                eval_model = setup_proto_entropy_eata(base_model, entropy_threshold=proto_threshold)
            elif mode == 'loss':
                eval_model = setup_loss_adapt(base_model)
            elif mode == 'fisher':
                eval_model = setup_fisher_proto(base_model)
            elif mode == 'eata':
                if current_fishers is None:
                    raise ValueError("Fishers could not be computed for EATA")
                eval_model = setup_eata(base_model, current_fishers)
            elif mode == 'sar':
                eval_model = setup_sar(base_model)
            elif mode == 'memo':
                eval_model = setup_memo(base_model, lr=0.00025, batch_size=64, steps=1)
            else:
                logger.warning(f"Unknown mode: {mode}")
                continue
            
            # Evaluate
            acc = evaluate_model(eval_model, loader, 
                               description=f"{mode.capitalize()} on {corruption_type}-{severity}",
                               verbose=False)
            results[mode] = float(acc)
            
            # Cleanup
            del eval_model
            del base_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            logger.error(f"Failed to evaluate {mode} on {corruption_type}-{severity}: {e}")
            results[mode] = None
    
    return results



def compute_metrics(results_dict):
    """
    Compute mCE (mean Corruption Error) and other aggregate metrics.
    
    Note: True mCE requires baseline model errors. Here we compute mean accuracy.
    """
    metrics = {}
    
    for mode in ['normal', 'tent', 'proto', 'proto_eata', 'loss', 'fisher', 'eata', 'sar', 'memo']:
        if mode not in results_dict:
            continue
        
        all_accuracies = []
        corruption_means = {}
        
        for corruption_type, severities in results_dict[mode].items():
            if corruption_type == 'clean':
                continue
            
            valid_accs = [acc for acc in severities.values() if acc is not None]
            if valid_accs:
                corruption_means[corruption_type] = np.mean(valid_accs)
                all_accuracies.extend(valid_accs)
        
        if all_accuracies:
            metrics[mode] = {
                'mean_accuracy': np.mean(all_accuracies),
                'std_accuracy': np.std(all_accuracies),
                'min_accuracy': np.min(all_accuracies),
                'max_accuracy': np.max(all_accuracies),
                'corruption_means': corruption_means
            }
    
    return metrics


def print_results_table(results_dict, metrics, clean_results=None):
    """Print formatted results table."""
    print("\n" + "="*80)
    print("ROBUSTNESS EVALUATION RESULTS")
    print("="*80)
    
    # Print clean accuracy if available
    if clean_results:
        print("\nClean Data Performance:")
        print("-" * 40)
        for mode, acc in clean_results.items():
            if acc is not None:
                print(f"  {mode.capitalize():15s}: {acc*100:6.2f}%")
    
    # Print per-corruption results
    print("\n" + "="*80)
    print("Per-Corruption Results (all severities)")
    print("="*80)
    
    modes = [m for m in ['normal', 'tent', 'proto', 'proto_eata', 'loss', 'fisher', 'eata', 'sar', 'memo'] if m in results_dict]
    
    if not modes:
        print("No results to display.")
        return
    
    # Header
    header = f"{'Corruption':<20s}"
    for mode in modes:
        header += f" {mode.capitalize():>12s}"
    print(header)
    print("-" * 80)
    
    # Get all corruption types
    corruption_types = list(results_dict[modes[0]].keys())
    corruption_types = [c for c in corruption_types if c != 'clean']
    corruption_types.sort()
    
    # Print results for each corruption
    for corruption in corruption_types:
        line = f"{corruption:<20s}"
        for mode in modes:
            if corruption in results_dict[mode]:
                severities = results_dict[mode][corruption]
                valid_accs = [acc for acc in severities.values() if acc is not None]
                if valid_accs:
                    mean_acc = np.mean(valid_accs)
                    line += f" {mean_acc*100:11.2f}%"
                else:
                    line += f" {'N/A':>12s}"
            else:
                line += f" {'N/A':>12s}"
        print(line)
    
    # Print summary metrics
    print("\n" + "="*80)
    print("Summary Metrics")
    print("="*80)
    
    for mode in modes:
        if mode in metrics:
            m = metrics[mode]
            print(f"\n{mode.capitalize()}:")
            print(f"  Mean Accuracy:  {m['mean_accuracy']*100:.2f}%")
            print(f"  Std Accuracy:   {m['std_accuracy']*100:.2f}%")
            print(f"  Min Accuracy:   {m['min_accuracy']*100:.2f}%")
            print(f"  Max Accuracy:   {m['max_accuracy']*100:.2f}%")


def save_results(results_dict, metrics, output_file, clean_results=None):
    """Save results to JSON file."""
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'clean_results': clean_results,
        'corruption_results': results_dict,
        'metrics': metrics
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive robustness evaluation on CUB-200-C',
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
                       default='./datasets/cub200_cropped/test_cropped/',
                       help='Path to clean test data (for on-the-fly or clean evaluation)')
    
    # Evaluation settings
    parser.add_argument('--corruptions', nargs='+', default=['all'],
                       help='Corruption types to evaluate. Use "all" for all types.')
    parser.add_argument('--severities', nargs='+', type=int, default=[2, 3, 4, 5],
                       help='Severity levels to evaluate (1-5)')
    parser.add_argument('--mode', type=str, default='all',
                       help='Evaluation modes: normal, tent, proto, proto_eata, loss, fisher, eata, sar, memo, or "all" (comma-separated)')
    
    # Data loading
    parser.add_argument('--on_the_fly', action='store_true',
                       help='Generate corruptions on-the-fly instead of loading pre-generated')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    # Output
    parser.add_argument('--output', type=str, default='./robustness_results.json',
                       help='Path to save results JSON')
    parser.add_argument('--eval_clean', action='store_true',
                       help='Also evaluate on clean (uncorrupted) test data')
    
    # Hardware
    parser.add_argument('--gpuid', type=str, default='0',
                       help='GPU ID to use')
    
    # EATA settings
    parser.add_argument('--use_clean_fisher', action='store_true', default=False,
                       help='Use clean data to compute Fisher Information Matrix for EATA. Default is False (use test data).')

    parser.add_argument('--proto_threshold', type=float, default=0.4,
                       help='Entropy threshold for ProtoEntropy+EATA adaptation (default: 0.4).')

    args = parser.parse_args()
    
    # Setup device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Determine corruption types
    if 'all' in args.corruptions:
        corruption_types = get_all_corruption_types()
    else:
        corruption_types = args.corruptions
    
    # Determine modes
    if args.mode.lower() == 'all':
        modes = ['normal', 'tent', 'proto', 'proto_eata', 'loss', 'fisher', 'eata', 'sar', 'memo']
    else:
        modes = [m.strip().lower() for m in args.mode.split(',')]
    
    # Verify model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    
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
        clean_dataset = datasets.ImageFolder(args.clean_data_dir, transform_clean)
        
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
    
    # Print configuration
    print("="*80)
    print("ROBUSTNESS EVALUATION CONFIGURATION")
    print("="*80)
    print(f"Model:           {args.model}")
    print(f"Data directory:  {args.data_dir if not args.on_the_fly else 'On-the-fly generation'}")
    print(f"Corruptions:     {len(corruption_types)} types")
    print(f"Severities:      {args.severities}")
    print(f"Modes:           {', '.join(modes)}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Output file:     {args.output}")
    print("="*80)
    
    # Evaluate clean data if requested
    clean_results = {}
    if args.eval_clean:
        print("\n" + "="*80)
        print("Evaluating on Clean Data")
        print("="*80)
        
        transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        clean_dataset = datasets.ImageFolder(args.clean_data_dir, transform)
        clean_loader = torch.utils.data.DataLoader(
            clean_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        for mode in modes:
            base_model = torch.load(args.model, weights_only=False)
            base_model = base_model.to(device)
            base_model.eval()
            
            if mode == 'normal':
                eval_model = base_model
            elif mode == 'tent':
                eval_model = setup_tent(base_model)
            elif mode == 'proto':
                eval_model = setup_proto_entropy(base_model)
            elif mode == 'proto_eata':
                eval_model = setup_proto_entropy_eata(base_model, entropy_threshold=args.proto_threshold)
            elif mode == 'loss':
                eval_model = setup_loss_adapt(base_model)
            elif mode == 'fisher':
                eval_model = setup_fisher_proto(base_model)
            elif mode == 'eata':
                # For clean data eval, compute Fisher on clean data
                if fishers is None:
                    # Compute on clean loader if not provided
                    print("Computing Fisher on clean data for clean eval...")
                    fisher_model = torch.load(args.model, weights_only=False)
                    fisher_model = fisher_model.to(device)
                    fisher_model = eata_adapt.configure_model(fisher_model)
                    fishers = eata_adapt.compute_fishers(fisher_model, clean_loader, device, num_samples=2000)
                    del fisher_model
                
                eval_model = setup_eata(base_model, fishers)
            elif mode == 'sar':
                eval_model = setup_sar(base_model)
            elif mode == 'memo':
                eval_model = setup_memo(base_model, lr=0.00025, batch_size=64, steps=1)
            
            acc = evaluate_model(eval_model, clean_loader, 
                               description=f"Clean data - {mode.capitalize()}")
            clean_results[mode] = float(acc)
            
            del eval_model
            del base_model
            torch.cuda.empty_cache()
    
    # Initialize results dictionary
    results_dict = {mode: {} for mode in modes}
    
    # Evaluate each corruption and severity
    print("\n" + "="*80)
    print("Evaluating Corruptions")
    print("="*80)
    
    total_evaluations = len(corruption_types) * len(args.severities) * len(modes)
    current_eval = 0
    
    start_time = time.time()
    
    for corruption_type in corruption_types:
        print(f"\n{'='*80}")
        print(f"Corruption: {corruption_type}")
        print(f"{'='*80}")
        
        for mode in modes:
            results_dict[mode][corruption_type] = {}
        
        for severity in args.severities:
            print(f"\n  Severity {severity}:")
            
            # Evaluate this corruption-severity combination
            results = evaluate_corruption(
                args.model,
                corruption_type,
                severity,
                args.data_dir,
                args.clean_data_dir,
                args.on_the_fly,
                modes,
                device,
                args.batch_size,
                fishers=fishers,  # Pass globally computed fishers (if any)
                proto_threshold=args.proto_threshold
            )
            
            if results:
                for mode in modes:
                    if mode in results:
                        results_dict[mode][corruption_type][severity] = results[mode]
                        if results[mode] is not None:
                            print(f"    {mode.capitalize():12s}: {results[mode]*100:.2f}%")
                        current_eval += 1
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time = elapsed / max(current_eval, 1)
            remaining = (total_evaluations - current_eval) * avg_time
            print(f"  Progress: {current_eval}/{total_evaluations} "
                  f"({current_eval/total_evaluations*100:.1f}%) "
                  f"- Est. remaining: {remaining/60:.1f} min")
    
    # Compute aggregate metrics
    metrics = compute_metrics(results_dict)
    
    # Print results
    print_results_table(results_dict, metrics, clean_results)
    
    # Save results
    save_results(results_dict, metrics, args.output, clean_results)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print("="*80)


if __name__ == '__main__':
    main()

