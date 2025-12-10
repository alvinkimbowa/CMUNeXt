#!/usr/bin/env python3
"""
Analyze model parameters and FLOPs
Callable from command line or from run_train.sh
"""
import argparse
import json
import torch

from thop import profile

from network.CMUNeXt import cmunext, cmunext_s, cmunext_l

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_flops(model, input_tensor):
    """Count FLOPs using thop"""
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = macs * 2  # MACs to FLOPS (multiply-add operations)
    return macs, flops, params


def analyze_model(arch, input_channels=1, num_classes=2, input_h=256, input_w=256, 
                  deep_supervision=False, gpu=0, save_path=None):
    """Analyze model parameters and FLOPs"""
    
    # Set device
    if gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*60}")
    print(f"MODEL ANALYSIS")
    print(f"{'='*60}")
    print(f"Architecture: {arch}")
    print(f"Input channels: {input_channels}")
    print(f"Number of classes: {num_classes}")
    print(f"Input size: {input_h}x{input_w}")
    print(f"Deep supervision: {deep_supervision}")
    print(f"Device: {device}")
    print(f"{'='*60}")
    
    # Load model (mirror main.py get_model without importing it to avoid arg parsing there)
    if arch == "CMUNeXt":
        model = cmunext()
    elif arch == "CMUNeXt-S":
        model = cmunext_s()
    elif arch == "CMUNeXt-L":
        model = cmunext_l()
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Create input tensor
    input_tensor = torch.randn(1, input_channels, input_h, input_w).to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    
    # Count FLOPs
    macs, flops, thop_params = count_flops(model, input_tensor)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"PARAMETERS:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    print(f"\nCOMPUTATIONAL COMPLEXITY:")
    print(f"  MACs (Multiply-Accumulates): {macs:,} ({macs/1e9:.2f}G)")
    print(f"  FLOPS (Floating Point Operations): {flops:,} ({flops/1e9:.2f}G)")
    print(f"  thop parameter count: {thop_params:,}")
    
    print(f"{'='*60}\n")
    
    # Prepare metrics dictionary
    metrics = {
        "architecture": arch,
        "input_channels": input_channels,
        "num_classes": num_classes,
        "input_size": [input_h, input_w],
        "deep_supervision": deep_supervision,
        "device": str(device),
        "parameters": {
            "total": int(total_params),
            "trainable": int(trainable_params),
            "non_trainable": int(total_params - trainable_params),
            "total_millions": round(total_params/1e6, 2),
            "trainable_millions": round(trainable_params/1e6, 2)
        }
    }
    
    metrics["macs"] = {
        "total": int(macs),
        "giga": round(macs/1e9, 2)
    }
    metrics["flops"] = {
        "total": int(flops),
        "giga": round(flops/1e9, 2)
    }
    
    # Save metrics to JSON
    if save_path:
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {save_path}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Analyze model parameters and FLOPs')
    parser.add_argument('--arch', '-a', type=str, required=True,
                        choices=["CMUNeXt", "CMUNeXt-S", "CMUNeXt-L"],
                        help='Model architecture name')
    parser.add_argument('--input_channels', type=int, default=1,
                        help='Number of input channels (default: 1)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes (default: 2)')
    parser.add_argument('--input_h', type=int, default=256,
                        help='Input height (default: 256)')
    parser.add_argument('--input_w', type=int, default=256,
                        help='Input width (default: 256)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool,
                        help='Use deep supervision')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (default: 0, use -1 for CPU)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save metrics JSON (optional)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Alias for --save (optional)')
    
    args = parser.parse_args()
    
    save_path = args.save_path or args.save

    analyze_model(
        arch=args.arch,
        input_channels=args.input_channels,
        num_classes=args.num_classes,
        input_h=args.input_h,
        input_w=args.input_w,
        deep_supervision=args.deep_supervision,
        gpu=args.gpu,
        save_path=save_path
    )


if __name__ == '__main__':
    main()
