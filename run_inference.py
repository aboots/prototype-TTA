import os
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import torch.optim as optim
import logging
import train_and_test as tnt
import model # Necessary for torch.load to find the class definition
import push_greedy # Importing just in case the model object depends on it
from settings import img_size, test_dir, test_batch_size, k, sum_cls
from preprocess import mean, std
from noise_utils import get_corrupted_transform
import tent
import proto_entropy
import loss_adapt
import fisher_proto
import eata_adapt
import memo_adapt
import sar_adapt
from pathlib import Path
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)

class Cfg:
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
    """Set up optimizer for tent adaptation.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError

def setup_tent(model):
    """Set up tent adaptation.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_proto_entropy(model, use_importance=False, use_confidence=False, 
                        reset_mode=None, reset_frequency=10, 
                        confidence_threshold=0.7, ema_alpha=0.999,
                        use_geometric_filter=False, geo_filter_threshold=0.3,
                        consensus_strategy='max', consensus_ratio=0.5,
                        adaptation_mode='layernorm_only',
                        use_ensemble_entropy=False):
    """Set up Prototype Entropy adaptation (without threshold).
    
    Args:
        reset_mode: 'episodic', 'periodic', 'confidence', 'hybrid', 'ema', 'none'
                   If None, infers from cfg.MODEL.EPISODIC
        reset_frequency: How often to reset in 'periodic'/'hybrid' modes (in BATCHES, not samples)
        confidence_threshold: Confidence threshold for 'confidence'/'hybrid' modes
        ema_alpha: EMA decay for 'ema' mode
        use_geometric_filter: Use geometric similarity to filter unreliable samples
        geo_filter_threshold: Minimum similarity to ANY prototype to be considered reliable
        consensus_strategy: How to aggregate sub-prototypes ('max', 'mean', 'median', 'top_k_mean', 'weighted_mean')
        consensus_ratio: For 'top_k_mean', fraction of sub-prototypes to use
        adaptation_mode: What parameters to adapt ('layernorm_only', 'layernorm_proto', etc.)
        use_ensemble_entropy: Treat sub-prototypes as ensemble (compute entropy per sub-proto, then average)
    """
    model = proto_entropy.configure_model(model, adaptation_mode=adaptation_mode)
    params, param_names = proto_entropy.collect_params(model, adaptation_mode=adaptation_mode)
    
    print(f"Adapting {len(params)} parameter groups: {', '.join(param_names[:5])}...")
    
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
        use_ensemble_entropy=use_ensemble_entropy
    )
    return proto_model


def setup_proto_entropy_eata(model, entropy_threshold=0.4):
    """Set up Prototype Entropy adaptation with EATA-style thresholding."""
    model = proto_entropy.configure_model(model)
    params, param_names = proto_entropy.collect_params(model)
    optimizer = setup_optimizer(params)
    proto_model = proto_entropy.ProtoEntropyEATA(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        entropy_threshold=entropy_threshold
    )
    return proto_model


def setup_loss_adapt(model):
    """Set up Loss-based adaptation."""
    model = loss_adapt.configure_model(model)
    params, param_names = loss_adapt.collect_params(model)
    optimizer = setup_optimizer(params)
    loss_model = loss_adapt.LossAdapt(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC
    )
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
    return loss_model


def setup_fisher_proto(model):
    """Set up Fisher-guided prototype adaptation."""
    model = fisher_proto.configure_model(model)
    params, param_names = fisher_proto.collect_params(model)
    optimizer = setup_optimizer(params)
    fisher_model = fisher_proto.FisherProto(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
    )
    # Ensure wrapper (and its buffers such as fisher_scores) are on the
    # same device as the underlying model parameters.
    device = next(model.parameters()).device
    fisher_model = fisher_model.to(device)
    # logger.info(f"model for adaptation (Fisher): %s", model)
    # logger.info(f"params for adaptation (Fisher): %s", param_names)
    # logger.info(f"optimizer for adaptation (Fisher): %s", optimizer)
    return fisher_model


def setup_eata(model, fishers):
    """Set up EATA adaptation."""
    model = eata_adapt.configure_model(model)
    params, param_names = eata_adapt.collect_params(model)
    optimizer = setup_optimizer(params)
    eata_model = eata_adapt.EATA(
        model,
        optimizer,
        fishers=fishers,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC
    )
    return eata_model


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
    """
    model = memo_adapt.configure_model(model)
    params, param_names = memo_adapt.collect_params(model)
    
    # MEMO uses SGD optimizer with momentum
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)
    
    memo_model = memo_adapt.MEMO(
        model,
        optimizer,
        steps=steps,
        batch_size=batch_size,
        episodic=True
    )
    return memo_model


def create_performance_plots(results_dict, output_dir, corruption_info=""):
    """Create bar plots showing per-batch performance for each method.
    
    Args:
        results_dict: Dict mapping method_name -> {'accuracy': float, 'batch_accuracies': list}
        output_dir: Directory to save plots
        corruption_info: String describing corruption (for title)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    methods = []
    batch_accs_list = []
    final_accs = []
    
    for method_name, data in results_dict.items():
        if 'batch_accuracies' in data and data['batch_accuracies'] is not None:
            methods.append(method_name)
            batch_accs_list.append(data['batch_accuracies'])
            final_accs.append(data['accuracy'] * 100)
    
    if not methods:
        print("No per-batch data to plot.")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for grouped bar plot
    num_batches = max(len(batch_accs) for batch_accs in batch_accs_list)
    x = np.arange(num_batches)
    width = 0.8 / len(methods)
    
    # Plot bars for each method
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    for i, (method, batch_accs, final_acc, color) in enumerate(zip(methods, batch_accs_list, final_accs, colors)):
        # Pad with NaN if needed
        padded_accs = batch_accs + [np.nan] * (num_batches - len(batch_accs))
        offset = (i - len(methods)/2 + 0.5) * width
        ax.bar(x + offset, [acc * 100 if not np.isnan(acc) else 0 for acc in padded_accs], 
               width, label=f'{method} (Final: {final_acc:.2f}%)', 
               color=color, alpha=0.7)
    
    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    title = f'Per-Batch Performance Comparison'
    if corruption_info:
        title += f'\n{corruption_info}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x[::max(1, num_batches//20)])  # Show every Nth batch label
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'per_batch_performance.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-batch performance plot to: {plot_path}")
    
    # Also create a line plot for easier trend visualization
    fig, ax = plt.subplots(figsize=(14, 8))
    for method, batch_accs, final_acc, color in zip(methods, batch_accs_list, final_accs, colors):
        batch_indices = np.arange(len(batch_accs))
        ax.plot(batch_indices, [acc * 100 for acc in batch_accs], 
               label=f'{method} (Final: {final_acc:.2f}%)', 
               color=color, linewidth=2, marker='o', markersize=3, alpha=0.7)
    
    ax.set_xlabel('Batch Index', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    title = f'Per-Batch Performance Trends'
    if corruption_info:
        title += f'\n{corruption_info}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'per_batch_trends.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-batch trends plot to: {plot_path}")


def parse_modes(mode_arg: str):
    """
    Parse a comma-separated list of modes.

    If empty or contains 'all', all modes are enabled.
    Valid individual modes: normal, tent, proto, proto_importance, proto_confidence, proto_importance_confidence, proto_eata, loss, fisher, eata, sar, memo.
    """
    if not mode_arg:
        return {"normal", "tent", "proto", "proto_importance", "proto_confidence", "proto_importance_confidence", "proto_eata", "loss", "fisher", "eata", "sar", "memo"}

    raw = [m.strip().lower() for m in mode_arg.split(",") if m.strip()]
    modes = set(raw)
    if "all" in modes:
        return {"normal", "tent", "proto", "proto_importance", "proto_confidence", "proto_importance_confidence", "proto_eata", "loss", "fisher", "eata", "sar", "memo"}

    valid = {"normal", "tent", "proto", "proto_importance", "proto_confidence", "proto_importance_confidence", "proto_eata", "loss", "fisher", "eata", "sar", "memo"}
    selected = modes & valid
    return selected or valid

def evaluate_model(model, loader, description="Inference", track_per_batch=False):
    """Helper to run evaluation loop.
    
    Args:
        track_per_batch: If True, returns per-batch accuracies for visualization
    """
    print(f'\nStarting {description}...')
    class_specific = True 
    
    if track_per_batch:
        # Custom evaluation with per-batch tracking
        model.eval()
        batch_accuracies = []
        n_examples = 0
        n_correct = 0
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()
                
                # Forward pass
                outputs = model(images)
                
                # Get predictions - handle both tuple and single output
                if isinstance(outputs, tuple):
                    logits = outputs[0]  # First element is logits
                else:
                    logits = outputs
                
                _, predicted = logits.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                batch_size = labels.size(0)
                batch_acc = batch_correct / batch_size
                batch_accuracies.append(batch_acc)
                
                n_correct += batch_correct
                n_examples += batch_size
        
        accu = n_correct / n_examples
        print('-' * 20)
        print(f'{description} Complete.')
        print(f'Final Accuracy: {accu*100:.2f}%')
        print('-' * 20)
        return accu, batch_accuracies
    else:
        accu, test_loss_dict = tnt.test(model=model, dataloader=loader,
                                        class_specific=class_specific, log=print, 
                                        clst_k=k, sum_cls=sum_cls)
        print('-' * 20)
        print(f'{description} Complete.')
        print(f'Final Accuracy: {accu*100:.2f}%')
        print('-' * 20)
        return accu


def run_unified_inference(model_path, gpu_id='0', corruption=None, severity=1, mode='all', 
                         use_pre_generated=True, use_clean_fisher=False, proto_threshold=None,
                         reset_mode=None, reset_frequency=10, confidence_threshold=0.7, ema_alpha=0.999,
                         use_geometric_filter=False, geo_filter_threshold=0.3, output_dir='./plots',
                         consensus_strategy='max', consensus_ratio=0.5,
                         adaptation_mode='layernorm_only',
                         use_ensemble_entropy=False):
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(f'Using GPU: {os.environ["CUDA_VISIBLE_DEVICES"]}')

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Data loading
    if corruption:
        # Check if pre-generated corrupted dataset exists
        corrupted_data_dir = Path('./datasets/cub200_c')
        corruption_path = corrupted_data_dir / corruption / str(severity)
        
        if use_pre_generated and corruption_path.exists():
            print(f'Using PRE-GENERATED corrupted images from: {corruption_path}')
            # Load pre-generated corrupted dataset (corruption already applied)
            transform = transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            test_dataset = datasets.ImageFolder(str(corruption_path), transform)
        else:
            if use_pre_generated:
                print(f'Pre-generated corrupted images not found at {corruption_path}')
            print(f'Generating corruption ON-THE-FLY: {corruption} (Severity: {severity})')
            print(f'Loading clean images from: {test_dir}')
            transform = get_corrupted_transform(img_size, mean, std, corruption, severity)
            test_dataset = datasets.ImageFolder(test_dir, transform)
    else:
        print('Applying NO corruption (Clean Data)')
        print(f'Loading test data from: {test_dir}')
        transform = get_corrupted_transform(img_size, mean, std, None, severity)
        test_dataset = datasets.ImageFolder(test_dir, transform)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=8, pin_memory=False)

    print(f'Test set size: {len(test_loader.dataset)}')
    
    results = {}
    results_with_batches = {}  # Store per-batch data for plotting

    # Determine which modes to run (supports comma-separated list).
    selected_modes = parse_modes(mode)
    run_normal = "normal" in selected_modes
    run_tent = "tent" in selected_modes
    run_proto = "proto" in selected_modes
    run_proto_importance = "proto_importance" in selected_modes
    run_proto_confidence = "proto_confidence" in selected_modes
    run_proto_importance_confidence = "proto_importance_confidence" in selected_modes
    run_proto_eata = "proto_eata" in selected_modes
    run_loss = "loss" in selected_modes
    run_fisher = "fisher" in selected_modes
    run_eata = "eata" in selected_modes
    run_sar = "sar" in selected_modes
    run_memo = "memo" in selected_modes

    # --- NORMAL INFERENCE ---
    if run_normal:
        print(f'\n>>> Loading model for NORMAL inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        base_model.eval()

        acc, batch_accs = evaluate_model(base_model, test_loader, 
                                         description="Normal Inference (No Adaptation)",
                                         track_per_batch=True)
        results['Normal'] = acc
        results_with_batches['Normal'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        # Clean up to free memory
        del base_model
        torch.cuda.empty_cache()

    # --- TENT INFERENCE ---
    if run_tent:
        print(f'\n>>> Loading model for TENT inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk (IMPORTANT: start from same state)
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        
        # Setup Tent (this switches mode to train for norm layers)
        print("Setting up Tent adaptation...")
        tent_model = setup_tent(base_model)
        
        acc, batch_accs = evaluate_model(tent_model, test_loader, 
                                         description="Tent Adaptation Inference",
                                         track_per_batch=True)
        results['Tent'] = acc
        results_with_batches['Tent'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        # Clean up
        del tent_model
        torch.cuda.empty_cache()

    # --- PROTO ENTROPY INFERENCE ---
    if run_proto:
        print(f'\n>>> Loading model for PROTO ENTROPY inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup ProtoEntropy (without threshold)
        filter_info = f", geo_filter={use_geometric_filter}" if use_geometric_filter else ""
        consensus_info = f", consensus={consensus_strategy}" if consensus_strategy != 'max' else ""
        adapt_info = f", adapt={adaptation_mode}" if adaptation_mode != 'layernorm_only' else ""
        ensemble_info = ", ensemble_entropy=True" if use_ensemble_entropy else ""
        print(f"Setting up ProtoEntropy (reset_mode={reset_mode}, freq={reset_frequency} batches{filter_info}{consensus_info}{adapt_info}{ensemble_info})...")
        proto_model = setup_proto_entropy(base_model, use_importance=False, use_confidence=False,
                                         reset_mode=reset_mode, reset_frequency=reset_frequency,
                                         confidence_threshold=confidence_threshold, ema_alpha=ema_alpha,
                                         use_geometric_filter=use_geometric_filter, 
                                         geo_filter_threshold=geo_filter_threshold,
                                         consensus_strategy=consensus_strategy, consensus_ratio=consensus_ratio,
                                         adaptation_mode=adaptation_mode,
                                         use_ensemble_entropy=use_ensemble_entropy)

        if hasattr(proto_model, 'reset_geo_filter_stats'):
            proto_model.reset_geo_filter_stats()
        
        acc, batch_accs = evaluate_model(proto_model, test_loader, 
                                         description="ProtoEntropy Adaptation Inference",
                                         track_per_batch=True)
        results['ProtoEntropy'] = acc
        results_with_batches['ProtoEntropy'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        if use_geometric_filter and hasattr(proto_model, 'get_geo_filter_stats'):
            geo_stats = proto_model.get_geo_filter_stats()
            results_with_batches['ProtoEntropy']['geo_stats'] = geo_stats

        # Clean up
        del proto_model
        torch.cuda.empty_cache()

    # --- PROTO ENTROPY with IMPORTANCE WEIGHTING ---
    if run_proto_importance:
        print(f'\n>>> Loading model for PROTO ENTROPY (Importance-Weighted) inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup ProtoEntropy with importance weighting
        filter_info = f", geo_filter={use_geometric_filter}" if use_geometric_filter else ""
        consensus_info = f", consensus={consensus_strategy}" if consensus_strategy != 'max' else ""
        adapt_info = f", adapt={adaptation_mode}" if adaptation_mode != 'layernorm_only' else ""
        ensemble_info = ", ensemble_entropy=True" if use_ensemble_entropy else ""
        print(f"Setting up ProtoEntropy+Importance (reset_mode={reset_mode}, freq={reset_frequency}{filter_info}{consensus_info}{adapt_info}{ensemble_info})...")
        proto_model = setup_proto_entropy(base_model, use_importance=True, use_confidence=False,
                                         reset_mode=reset_mode, reset_frequency=reset_frequency,
                                         confidence_threshold=confidence_threshold, ema_alpha=ema_alpha,
                                         use_geometric_filter=use_geometric_filter, 
                                         geo_filter_threshold=geo_filter_threshold,
                                         consensus_strategy=consensus_strategy, consensus_ratio=consensus_ratio,
                                         adaptation_mode=adaptation_mode,
                                         use_ensemble_entropy=use_ensemble_entropy)

        if hasattr(proto_model, 'reset_geo_filter_stats'):
            proto_model.reset_geo_filter_stats()
        
        acc, batch_accs = evaluate_model(proto_model, test_loader, 
                                         description="ProtoEntropy (Importance-Weighted) Inference",
                                         track_per_batch=True)
        results['ProtoEntropy-Importance'] = acc
        results_with_batches['ProtoEntropy-Importance'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        if use_geometric_filter and hasattr(proto_model, 'get_geo_filter_stats'):
            geo_stats = proto_model.get_geo_filter_stats()
            results_with_batches['ProtoEntropy-Importance']['geo_stats'] = geo_stats

        # Clean up
        del proto_model
        torch.cuda.empty_cache()

    # --- PROTO ENTROPY with CONFIDENCE WEIGHTING ---
    if run_proto_confidence:
        print(f'\n>>> Loading model for PROTO ENTROPY (Confidence-Weighted) inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup ProtoEntropy with confidence weighting
        filter_info = f", geo_filter={use_geometric_filter}" if use_geometric_filter else ""
        consensus_info = f", consensus={consensus_strategy}" if consensus_strategy != 'max' else ""
        adapt_info = f", adapt={adaptation_mode}" if adaptation_mode != 'layernorm_only' else ""
        ensemble_info = ", ensemble_entropy=True" if use_ensemble_entropy else ""
        print(f"Setting up ProtoEntropy+Confidence (reset_mode={reset_mode}, freq={reset_frequency}{filter_info}{consensus_info}{adapt_info}{ensemble_info})...")
        proto_model = setup_proto_entropy(base_model, use_importance=False, use_confidence=True,
                                         reset_mode=reset_mode, reset_frequency=reset_frequency,
                                         confidence_threshold=confidence_threshold, ema_alpha=ema_alpha,
                                         use_geometric_filter=use_geometric_filter, 
                                         geo_filter_threshold=geo_filter_threshold,
                                         consensus_strategy=consensus_strategy, consensus_ratio=consensus_ratio,
                                         adaptation_mode=adaptation_mode,
                                         use_ensemble_entropy=use_ensemble_entropy)

        if hasattr(proto_model, 'reset_geo_filter_stats'):
            proto_model.reset_geo_filter_stats()
        
        acc, batch_accs = evaluate_model(proto_model, test_loader, 
                                         description="ProtoEntropy (Confidence-Weighted) Inference",
                                         track_per_batch=True)
        results['ProtoEntropy-Confidence'] = acc
        results_with_batches['ProtoEntropy-Confidence'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        if use_geometric_filter and hasattr(proto_model, 'get_geo_filter_stats'):
            geo_stats = proto_model.get_geo_filter_stats()
            results_with_batches['ProtoEntropy-Confidence']['geo_stats'] = geo_stats

        # Clean up
        del proto_model
        torch.cuda.empty_cache()

    # --- PROTO ENTROPY with IMPORTANCE+CONFIDENCE WEIGHTING ---
    if run_proto_importance_confidence:
        print(f'\n>>> Loading model for PROTO ENTROPY (Importance+Confidence-Weighted) inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup ProtoEntropy with both importance and confidence weighting
        filter_info = f", geo_filter={use_geometric_filter}" if use_geometric_filter else ""
        consensus_info = f", consensus={consensus_strategy}" if consensus_strategy != 'max' else ""
        adapt_info = f", adapt={adaptation_mode}" if adaptation_mode != 'layernorm_only' else ""
        ensemble_info = ", ensemble_entropy=True" if use_ensemble_entropy else ""
        print(f"Setting up ProtoEntropy Imp+Conf (reset_mode={reset_mode}, freq={reset_frequency}{filter_info}{consensus_info}{adapt_info}{ensemble_info})...")
        proto_model = setup_proto_entropy(base_model, use_importance=True, use_confidence=True,
                                         reset_mode=reset_mode, reset_frequency=reset_frequency,
                                         confidence_threshold=confidence_threshold, ema_alpha=ema_alpha,
                                         use_geometric_filter=use_geometric_filter, 
                                         geo_filter_threshold=geo_filter_threshold,
                                         consensus_strategy=consensus_strategy, consensus_ratio=consensus_ratio,
                                         adaptation_mode=adaptation_mode,
                                         use_ensemble_entropy=use_ensemble_entropy)

        # Reset stats before evaluation
        if hasattr(proto_model, 'reset_geo_filter_stats'):
            proto_model.reset_geo_filter_stats()
        
        acc, batch_accs = evaluate_model(proto_model, test_loader, 
                                         description="ProtoEntropy (Importance+Confidence-Weighted) Inference",
                                         track_per_batch=True)
        results['ProtoEntropy-Imp+Conf'] = acc
        results_with_batches['ProtoEntropy-Imp+Conf'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        # Get geometric filter statistics
        if use_geometric_filter and hasattr(proto_model, 'get_geo_filter_stats'):
            geo_stats = proto_model.get_geo_filter_stats()
            results_with_batches['ProtoEntropy-Imp+Conf']['geo_stats'] = geo_stats
            print(f"\n--- Geometric Filter Statistics (ProtoEntropy-Imp+Conf) ---")
            print(f"Total samples processed: {geo_stats['total_samples']}")
            print(f"Samples filtered: {geo_stats['filtered_samples']} ({geo_stats['filter_rate']*100:.2f}%)")
            if geo_stats['overall_min_sim'] is not None:
                print(f"Minimum similarity across all batches: {geo_stats['overall_min_sim']:.4f}")
                print(f"Maximum similarity across all batches: {geo_stats['overall_max_sim']:.4f}")
                print(f"Average similarity across all batches: {geo_stats['overall_avg_sim']:.4f}")
                print(f"Threshold used: {geo_filter_threshold}")
            print("-" * 50)

        # Clean up
        del proto_model
        torch.cuda.empty_cache()

    # --- PROTO ENTROPY+EATA INFERENCE ---
    if run_proto_eata:
        print(f'\n>>> Loading model for PROTO ENTROPY+EATA inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup ProtoEntropy with EATA threshold
        print(f"Setting up ProtoEntropy+EATA adaptation (threshold={proto_threshold})...")
        proto_eata_model = setup_proto_entropy_eata(base_model, entropy_threshold=proto_threshold)

        acc, batch_accs = evaluate_model(proto_eata_model, test_loader, 
                                         description="ProtoEntropy+EATA Adaptation Inference",
                                         track_per_batch=True)
        results['ProtoEntropy+EATA'] = acc
        results_with_batches['ProtoEntropy+EATA'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }

        # Clean up
        del proto_eata_model
        torch.cuda.empty_cache()

    # --- LOSS ADAPT INFERENCE ---
    if run_loss:
        print(f'\n>>> Loading model for LOSS ADAPT inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")
             
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        
        print("Setting up Loss Adapt adaptation...")
        loss_model = setup_loss_adapt(base_model)
        
        acc, batch_accs = evaluate_model(loss_model, test_loader, 
                                         description="Loss Adapt Inference",
                                         track_per_batch=True)
        results['LossAdapt'] = acc
        results_with_batches['LossAdapt'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }
        
        del loss_model
        torch.cuda.empty_cache()

    # --- FISHER-PROTO INFERENCE ---
    if run_fisher:
        print(f'\n>>> Loading model for FISHER PROTO inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup Fisher-guided prototype adaptation
        print("Setting up Fisher-guided Proto adaptation...")
        fisher_model = setup_fisher_proto(base_model)

        acc, batch_accs = evaluate_model(fisher_model, test_loader, 
                                         description="FisherProto Adaptation Inference",
                                         track_per_batch=True)
        results['FisherProto'] = acc
        results_with_batches['FisherProto'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }

        # Clean up
        del fisher_model
        torch.cuda.empty_cache()

    # --- EATA INFERENCE ---
    if run_eata:
        print(f'\n>>> Loading model for EATA inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        if use_clean_fisher:
            print("Computing Fisher Information Matrix on CLEAN data (using clean images)...")
            # Prepare clean data loader for Fisher computation (subset of 2000 samples)
            transform_clean = get_corrupted_transform(img_size, mean, std, None, 1)
            clean_dataset = datasets.ImageFolder(test_dir, transform_clean)
            
            num_fisher_samples = 2000
            if len(clean_dataset) > num_fisher_samples:
                fisher_indices = torch.randperm(len(clean_dataset))[:num_fisher_samples]
                fisher_subset = torch.utils.data.Subset(clean_dataset, fisher_indices)
            else:
                fisher_subset = clean_dataset
                
            fisher_loader = torch.utils.data.DataLoader(
                fisher_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
            )
            
            base_model = eata_adapt.configure_model(base_model)
            fishers = eata_adapt.compute_fishers(base_model, fisher_loader, device)
        else:
            print("Computing Fisher Information Matrix on TEST data (online estimation)...")
            # Use test loader (corrupted data) to estimate Fishers
            base_model = eata_adapt.configure_model(base_model)
            # Compute Fishers on a subset of test data (e.g., first 2000 samples)
            fishers = eata_adapt.compute_fishers(base_model, test_loader, device, num_samples=500)
        
        print("Fisher information computed.")

        # Setup EATA
        print("Setting up EATA adaptation...")
        eata_model = setup_eata(base_model, fishers)

        acc, batch_accs = evaluate_model(eata_model, test_loader, 
                                         description="EATA Adaptation Inference",
                                         track_per_batch=True)
        results['EATA'] = acc
        results_with_batches['EATA'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }

        # Clean up
        del eata_model
        torch.cuda.empty_cache()

    # --- SAR INFERENCE ---
    if run_sar:
        print(f'\n>>> Loading model for SAR inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        # Setup SAR
        print("Setting up SAR adaptation...")
        sar_model = setup_sar(base_model)

        acc, batch_accs = evaluate_model(sar_model, test_loader, 
                                         description="SAR Adaptation Inference",
                                         track_per_batch=True)
        results['SAR'] = acc
        results_with_batches['SAR'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }

        # Clean up
        del sar_model
        torch.cuda.empty_cache()

    # --- MEMO INFERENCE ---
    if run_memo:
        print(f'\n>>> Loading model for MEMO inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)

        print("Setting up MEMO adaptation...")
        print("MEMO parameters: lr=0.00025, batch_size=32, steps=1")
        memo_model = setup_memo(base_model, lr=0.00025, batch_size=32, steps=1)
        
        # MEMO requires batch_size=1 (processes one image at a time)
        memo_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=4, pin_memory=False)

        acc, batch_accs = evaluate_model(memo_model, memo_loader, 
                                         description="MEMO Adaptation Inference",
                                         track_per_batch=True)
        results['MEMO'] = acc
        results_with_batches['MEMO'] = {
            'accuracy': acc,
            'batch_accuracies': batch_accs
        }

        # Clean up
        del memo_model
        torch.cuda.empty_cache()

    # --- CREATE PLOTS ---
    if results_with_batches:
        corruption_str = f"{corruption}_sev{severity}" if corruption else "clean"
        plot_subdir = os.path.join(output_dir, corruption_str)
        os.makedirs(plot_subdir, exist_ok=True)
        
        # Create per-batch performance plots
        create_performance_plots(results_with_batches, plot_subdir, 
                                corruption_info=f"Corruption: {corruption if corruption else 'None'}, Severity: {severity if corruption else 'N/A'}")
        
        # Print geometric filter statistics for all methods that used it
        print("\n" + "="*50)
        print("GEOMETRIC FILTER STATISTICS")
        print("="*50)
        for method_name, data in results_with_batches.items():
            if 'geo_stats' in data:
                geo_stats = data['geo_stats']
                print(f"\n{method_name}:")
                print(f"  Total samples: {geo_stats['total_samples']}")
                print(f"  Filtered: {geo_stats['filtered_samples']} ({geo_stats['filter_rate']*100:.2f}%)")
                if geo_stats['overall_min_sim'] is not None:
                    print(f"  Min similarity: {geo_stats['overall_min_sim']:.4f}")
                    print(f"  Max similarity: {geo_stats['overall_max_sim']:.4f}")
                    print(f"  Avg similarity: {geo_stats['overall_avg_sim']:.4f}")
                    print(f"  Threshold: {geo_filter_threshold}")
        print("="*50)
    
    # --- FINAL SUMMARY ---
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Dataset Corruption: {corruption if corruption else 'None'}")
    if corruption:
        print(f"Severity: {severity}")
    print("-" * 50)
    
    if 'Normal' in results:
        print(f"Normal                   Accuracy: {results['Normal']*100:.2f}%")
    if 'Tent' in results:
        print(f"Tent                     Accuracy: {results['Tent']*100:.2f}%")
    if 'ProtoEntropy' in results:
        print(f"ProtoEntropy             Accuracy: {results['ProtoEntropy']*100:.2f}%")
    if 'ProtoEntropy-Importance' in results:
        print(f"ProtoEntropy-Importance  Accuracy: {results['ProtoEntropy-Importance']*100:.2f}%")
    if 'ProtoEntropy-Confidence' in results:
        print(f"ProtoEntropy-Confidence  Accuracy: {results['ProtoEntropy-Confidence']*100:.2f}%")
    if 'ProtoEntropy-Imp+Conf' in results:
        print(f"ProtoEntropy-Imp+Conf    Accuracy: {results['ProtoEntropy-Imp+Conf']*100:.2f}%")
    if 'ProtoEntropy+EATA' in results:
        print(f"ProtoEntropy+EATA        Accuracy: {results['ProtoEntropy+EATA']*100:.2f}%")
    if 'LossAdapt' in results:
        print(f"LossAdapt                Accuracy: {results['LossAdapt']*100:.2f}%")
    if 'FisherProto' in results:
        print(f"FisherProto              Accuracy: {results['FisherProto']*100:.2f}%")
    if 'EATA' in results:
        print(f"EATA                     Accuracy: {results['EATA']*100:.2f}%")
    if 'SAR' in results:
        print(f"SAR                      Accuracy: {results['SAR']*100:.2f}%")
    if 'MEMO' in results:
        print(f"MEMO                     Accuracy: {results['MEMO']*100:.2f}%")
    
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Inference for ProtoViT with multiple TTA methods')
    
    default_model_path = './saved_models/deit_small_patch16_224/exp1/14finetuned0.8609.pth'
    
    parser.add_argument('-model', type=str, default=default_model_path, help='Path to the saved model file')
    parser.add_argument('-gpuid', type=str, default='0', help='GPU ID to use')
    parser.add_argument('-corruption', type=str, default='gaussian_noise', 
                       help='Type of corruption to apply (e.g., gaussian_noise). Use None or empty string for clean data (no corruption).')
    parser.add_argument('-severity', type=int, default=3, help='Severity of corruption (1-5)')
    parser.add_argument('--no-corruption', action='store_true', 
                       help='Run inference without any corruption (clean test data)')
    parser.add_argument(
        '-mode',
        type=str,
        default='all',
        help=(
            'Inference mode(s): '
            'use a comma-separated list of any of [normal, tent, proto, proto_importance, proto_confidence, proto_importance_confidence, proto_eata, loss, fisher, eata, sar, memo], '
            'or "all" (default) to run every mode.'
        ),
    )  
    parser.add_argument(
        '--on-the-fly',
        action='store_true',
        default=False,
        help='Force on-the-fly corruption generation (ignore pre-generated images). By default, uses pre-generated images if available.'
    )
    
    parser.add_argument(
        '--use-clean-fisher',
        action='store_true',
        default=False,
        help='Use clean data to compute Fisher Information Matrix for EATA (requires access to clean data). Default is False (use test data).'
    )
    
    parser.add_argument(
        '--proto-threshold',
        type=float,
        default=0.4,
        help='Entropy threshold for ProtoEntropy+EATA adaptation (default: 0.4). Samples with higher entropy are ignored.'
    )
    
    parser.add_argument(
        '--reset-mode',
        type=str,
        default=None,
        choices=['episodic', 'periodic', 'confidence', 'hybrid', 'ema', 'none', None],
        help='Reset strategy for ProtoEntropy: episodic (every sample), periodic (every N samples), '
             'confidence (when conf drops), hybrid (periodic+confidence), ema (smooth updates), '
             'none (no resets). Default: inferred from --episodic flag.'
    )
    
    parser.add_argument(
        '--reset-frequency',
        type=int,
        default=10,
        help='How often to reset in periodic/hybrid modes in BATCHES (default: 10 batches). '
             'With batch_size=128, this means reset every 1280 samples.'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.7,
        help='Confidence threshold for confidence/hybrid reset modes (default: 0.7)'
    )
    
    parser.add_argument(
        '--ema-alpha',
        type=float,
        default=0.999,
        help='EMA decay factor for ema mode (default: 0.999, closer to 1 = slower adaptation)'
    )
    
    parser.add_argument(
        '--use-geometric-filter',
        action='store_true',
        default=False,
        help='Enable geometric filtering: filter out samples with low similarity to ALL prototypes (noisy/unreliable samples)'
    )
    
    parser.add_argument(
        '--geo-filter-threshold',
        type=float,
        default=0.3,
        help='Geometric filter threshold: minimum max similarity to ANY prototype to be considered reliable (default: 0.3). '
             'Higher = more strict filtering, lower = accept more samples. Range: [-1, 1]'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./plots',
        help='Directory to save performance plots (default: ./plots)'
    )
    
    parser.add_argument(
        '--consensus-strategy',
        type=str,
        default='max',
        choices=['max', 'mean', 'median', 'top_k_mean', 'weighted_mean'],
        help='Strategy for aggregating sub-prototypes: '
             'max (best sub-proto, default), mean (all must agree), median (robust), '
             'top_k_mean (soft consensus), weighted_mean (similarity-weighted)'
    )
    
    parser.add_argument(
        '--consensus-ratio',
        type=float,
        default=0.5,
        help='For top_k_mean consensus: fraction of sub-prototypes to use (default: 0.5 = top 50%%)'
    )
    
    parser.add_argument(
        '--adaptation-mode',
        type=str,
        default='layernorm_only',
        choices=['layernorm_only', 'layernorm_proto', 'layernorm_proto_patch', 
                 'layernorm_proto_last', 'layernorm_attn_bias', 'layernorm_last_block', 
                 'full_proto', 'all_adaptive'],
        help='What parameters to adapt during TTA:\n'
             'layernorm_only (default) - Only LayerNorms (safest)\n'
             'layernorm_proto - LayerNorms + Prototype vectors\n'
             'layernorm_proto_patch - LayerNorms + Prototypes + Patch selection\n'
             'layernorm_proto_last - LayerNorms + Prototypes + Last layer\n'
             'layernorm_attn_bias - LayerNorms + Attention biases\n'
             'layernorm_last_block - LayerNorms + Last transformer blocks\n'
             'full_proto - Prototypes + Patch select + Last layer (no backbone)\n'
             'all_adaptive - Everything except backbone features'
    )
    
    parser.add_argument(
        '--use-ensemble-entropy',
        action='store_true',
        default=False,
        help='Use ensemble entropy: compute entropy for each sub-prototype independently, then average. '
             'Prevents single overconfident sub-prototype from dominating (requires sub-prototypes).'
    )
    
    args = parser.parse_args()
    
    # Handle no-corruption flag
    if args.no_corruption:
        corruption = None
    elif args.corruption and args.corruption.lower() in ['none', 'null', '']:
        corruption = None
    else:
        corruption = args.corruption
    
    # Determine whether to use pre-generated images (default: True, unless --on-the-fly is set)
    use_pre_generated = not args.on_the_fly
    
    run_unified_inference(args.model, args.gpuid, corruption, args.severity, args.mode, 
                         use_pre_generated, args.use_clean_fisher, args.proto_threshold,
                         args.reset_mode, args.reset_frequency, args.confidence_threshold, args.ema_alpha,
                         args.use_geometric_filter, args.geo_filter_threshold, args.output_dir,
                         args.consensus_strategy, args.consensus_ratio, args.adaptation_mode,
                         args.use_ensemble_entropy)
