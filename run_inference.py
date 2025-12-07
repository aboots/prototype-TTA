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
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def setup_proto_entropy(model, entropy_threshold=None):
    """Set up Prototype Entropy adaptation."""
    model = proto_entropy.configure_model(model)
    params, param_names = proto_entropy.collect_params(model)
    optimizer = setup_optimizer(params)
    proto_model = proto_entropy.ProtoEntropy(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        entropy_threshold=entropy_threshold
    )
    # logger.info(f"model for adaptation: %s", model)
    # logger.info(f"params for adaptation: %s", param_names)
    # logger.info(f"optimizer for adaptation: %s", optimizer)
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


def parse_modes(mode_arg: str):
    """
    Parse a comma-separated list of modes.

    If empty or contains 'all', all modes are enabled.
    Valid individual modes: normal, tent, proto, loss, fisher.
    """
    if not mode_arg:
        return {"normal", "tent", "proto", "loss", "fisher", "eata"}

    raw = [m.strip().lower() for m in mode_arg.split(",") if m.strip()]
    modes = set(raw)
    if "all" in modes:
        return {"normal", "tent", "proto", "loss", "fisher", "eata"}

    valid = {"normal", "tent", "proto", "loss", "fisher", "eata"}
    selected = modes & valid
    return selected or valid

def evaluate_model(model, loader, description="Inference"):
    """Helper to run evaluation loop."""
    print(f'\nStarting {description}...')
    class_specific = True 
    accu, test_loss_dict = tnt.test(model=model, dataloader=loader,
                                    class_specific=class_specific, log=print, 
                                    clst_k=k, sum_cls=sum_cls)
    print('-' * 20)
    print(f'{description} Complete.')
    print(f'Final Accuracy: {accu*100:.2f}%')
    print('-' * 20)
    return accu


def run_unified_inference(model_path, gpu_id='0', corruption=None, severity=1, mode='all', use_pre_generated=True, use_clean_fisher=False, proto_threshold=None):
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

    # Determine which modes to run (supports comma-separated list).
    selected_modes = parse_modes(mode)
    run_normal = "normal" in selected_modes
    run_tent = "tent" in selected_modes
    run_proto = "proto" in selected_modes
    run_loss = "loss" in selected_modes
    run_fisher = "fisher" in selected_modes
    run_eata = "eata" in selected_modes

    # --- NORMAL INFERENCE ---
    if run_normal:
        print(f'\n>>> Loading model for NORMAL inference from {model_path}')
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found at {model_path}")

        # Load clean model fresh from disk
        base_model = torch.load(model_path, weights_only=False)
        base_model = base_model.to(device)
        base_model.eval()

        acc = evaluate_model(base_model, test_loader, description="Normal Inference (No Adaptation)")
        results['Normal'] = acc
        
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
        
        acc = evaluate_model(tent_model, test_loader, description="Tent Adaptation Inference")
        results['Tent'] = acc
        
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

        # Setup ProtoEntropy
        print(f"Setting up ProtoEntropy adaptation (threshold={proto_threshold})...")
        proto_model = setup_proto_entropy(base_model, entropy_threshold=proto_threshold)

        acc = evaluate_model(proto_model, test_loader, description="ProtoEntropy Adaptation Inference")
        results['ProtoEntropy'] = acc

        # Clean up
        del proto_model
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
        
        acc = evaluate_model(loss_model, test_loader, description="Loss Adapt Inference")
        results['LossAdapt'] = acc
        
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

        acc = evaluate_model(fisher_model, test_loader, description="FisherProto Adaptation Inference")
        results['FisherProto'] = acc

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

        acc = evaluate_model(eata_model, test_loader, description="EATA Adaptation Inference")
        results['EATA'] = acc

        # Clean up
        del eata_model
        torch.cuda.empty_cache()

    # --- FINAL SUMMARY ---
    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Dataset Corruption: {corruption if corruption else 'None'}")
    if corruption:
        print(f"Severity: {severity}")
    print("-" * 50)
    
    if 'Normal' in results:
        print(f"Normal       Accuracy: {results['Normal']*100:.2f}%")
    if 'Tent' in results:
        print(f"Tent         Accuracy: {results['Tent']*100:.2f}%")
    if 'ProtoEntropy' in results:
        print(f"ProtoEntropy Accuracy: {results['ProtoEntropy']*100:.2f}%")
    if 'LossAdapt' in results:
        print(f"LossAdapt    Accuracy: {results['LossAdapt']*100:.2f}%")
    if 'FisherProto' in results:
        print(f"FisherProto  Accuracy: {results['FisherProto']*100:.2f}%")
    if 'EATA' in results:
        print(f"EATA         Accuracy: {results['EATA']*100:.2f}%")
    
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified Inference for ProtoViT (Normal & Tent & ProtoEntropy)')
    
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
            'use a comma-separated list of any of [normal, tent, proto, loss, fisher, eata], '
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
        default=None,
        help='Entropy threshold for ProtoEntropy adaptation (e.g., 0.4). Samples with higher entropy are ignored.'
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
    
    run_unified_inference(args.model, args.gpuid, corruption, args.severity, args.mode, use_pre_generated, args.use_clean_fisher, args.proto_threshold)
