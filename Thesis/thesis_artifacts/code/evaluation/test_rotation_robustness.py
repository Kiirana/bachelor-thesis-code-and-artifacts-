#!/usr/bin/env python3
"""
Test rotation and shearing robustness of MobileNetV3 texture classifier.
Tests model performance under:
- Rotations: 0°, 90°, 180°, 270°
- Shearing: ±10°, ±20°
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode
from torchvision.models import mobilenet_v3_small
from torch.utils.data import DataLoader
from pathlib import Path
import json

def load_model(checkpoint_path: Path, num_classes: int = 4, device: str = "mps"):
    """Load trained MobileNetV3-Small model from checkpoint."""
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    
    return model, ckpt.get("classes", ["asphalt", "cobblestone", "gravel", "sand"])


def create_transform_with_perturbation(perturbation_type: str, angle: float = 0):
    """
    Create transform with specific perturbation.
    
    Args:
        perturbation_type: 'rotation', 'shear', or 'none'
        angle: angle in degrees for rotation or shear
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    base_transforms = [
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
    ]
    
    if perturbation_type == "rotation":
        # Apply rotation before resize/crop
        base_transforms.insert(0, transforms.RandomRotation(
            degrees=(angle, angle), 
            interpolation=InterpolationMode.BILINEAR
        ))
    elif perturbation_type == "shear":
        # Apply affine shearing
        base_transforms.insert(0, transforms.RandomAffine(
            degrees=0,
            shear=(angle, angle, 0, 0),  # horizontal shear only
            interpolation=InterpolationMode.BILINEAR
        ))
    
    base_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return transforms.Compose(base_transforms)


@torch.no_grad()
def evaluate_with_transform(model, test_root: Path, transform, device: str):
    """Evaluate model with specific transform."""
    test_ds = datasets.ImageFolder(test_root, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    
    correct = 0
    total = 0
    
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)
        pred = logits.argmax(dim=1)
        
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Paths
    checkpoint = Path("runs/baseline_mnv3/best.pt")
    test_root = Path("../texture_data_thesis/test")
    
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_root}")
    
    print("="*80)
    print("ROTATION & SHEARING ROBUSTNESS TEST")
    print("="*80)
    print(f"Model: {checkpoint}")
    print(f"Test set: {test_root}")
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model, class_names = load_model(checkpoint, device=device)
    print(f"Classes: {class_names}")
    
    # Test configurations
    test_configs = [
        ("Baseline (0° rotation)", "rotation", 0),
        ("Rotation 90°", "rotation", 90),
        ("Rotation 180°", "rotation", 180),
        ("Rotation 270°", "rotation", 270),
        ("Shear +10°", "shear", 10),
        ("Shear -10°", "shear", -10),
        ("Shear +20°", "shear", 20),
        ("Shear -20°", "shear", -20),
    ]
    
    results = {}
    
    print("\n" + "="*80)
    print("RUNNING TESTS")
    print("="*80)
    
    for test_name, perturb_type, angle in test_configs:
        print(f"\nTesting: {test_name}")
        transform = create_transform_with_perturbation(perturb_type, angle)
        accuracy, num_samples = evaluate_with_transform(model, test_root, transform, device)
        
        results[test_name] = {
            "accuracy": float(accuracy),
            "num_samples": num_samples,
            "perturbation_type": perturb_type,
            "angle": angle
        }
        
        print(f"  Accuracy: {accuracy*100:.2f}% ({num_samples} samples)")
    
    # Save results
    output_file = Path("runs/baseline_mnv3/rotation_robustness_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Rotation results
    print("\nRotation Robustness:")
    print("-" * 40)
    for test_name, data in results.items():
        if data["perturbation_type"] == "rotation":
            print(f"  {test_name:<25}: {data['accuracy']*100:>6.2f}%")
    
    # Shearing results
    print("\nShearing Robustness:")
    print("-" * 40)
    for test_name, data in results.items():
        if data["perturbation_type"] == "shear":
            print(f"  {test_name:<25}: {data['accuracy']*100:>6.2f}%")
    
    print(f"\n✅ Results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
