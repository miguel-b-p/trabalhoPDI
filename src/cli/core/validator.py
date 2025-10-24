"""
YOLO Argument Validation Module

This module provides validation and sanitization of YOLO training arguments
to ensure compatibility with the current Ultralytics API version.

Critical Changes Handled:
1. 'accumulate' -> 'nbs' (Nominal Batch Size)
2. 'label_smoothing' -> REMOVED (no replacement)
3. Other deprecated arguments from older YOLO versions

Author: YOLO Benchmark System
Institution: UNIP - Universidade Paulista
Year: 2025
"""

from typing import Dict, Any, Set, Tuple, Optional
import warnings
from rich.console import Console

console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# VALID YOLO ARGUMENTS (Ultralytics v8.3.x / v11.x)
# ══════════════════════════════════════════════════════════════════════════════

VALID_TRAIN_ARGUMENTS: Set[str] = {
    # Core Training
    'model', 'data', 'epochs', 'time', 'patience', 'batch', 'imgsz',
    'save', 'save_period', 'cache', 'device', 'workers', 'project', 'name',
    'exist_ok', 'pretrained', 'verbose', 'seed', 'deterministic',
    'single_cls', 'rect', 'cos_lr', 'close_mosaic', 'resume', 'amp',
    'fraction', 'profile', 'freeze', 'multi_scale', 'overlap_mask',
    'mask_ratio', 'dropout', 'val', 'plots', 'compile',
    
    # Optimization
    'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay',
    'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr',
    
    # Loss Weights
    'box', 'cls', 'dfl', 'pose', 'kobj', 'nbs',
    
    # Data Augmentation
    'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
    'shear', 'perspective', 'flipud', 'fliplr', 'bgr',
    'mosaic', 'mixup', 'copy_paste', 'copy_paste_mode',
    'auto_augment', 'erasing', 'cutmix',
    
    # Post-processing (used in validation/inference)
    'conf', 'iou', 'max_det', 'classes',
}

# ══════════════════════════════════════════════════════════════════════════════
# DEPRECATED & REMOVED ARGUMENTS
# ══════════════════════════════════════════════════════════════════════════════

# Arguments that have been renamed
DEPRECATED_MAPPINGS: Dict[str, Tuple[str, str]] = {
    'boxes': ('show_boxes', 'Renamed to show_boxes'),
    'hide_labels': ('show_labels', 'Inverted logic: hide_labels -> show_labels'),
    'hide_conf': ('show_conf', 'Inverted logic: hide_conf -> show_conf'),
    'line_thickness': ('line_width', 'Renamed to line_width'),
}

# Arguments that have been completely removed (no replacement)
REMOVED_ARGUMENTS: Set[str] = {
    'label_smoothing',  # Removed in v8.0+
    'save_hybrid',      # Removed in v8.0+
    'crop_fraction',    # Removed in v8.0+
}

# Critical argument replacements
CRITICAL_REPLACEMENTS: Dict[str, Tuple[str, str, Any]] = {
    'accumulate': (
        'nbs',
        'Gradient accumulation is now handled via nbs (Nominal Batch Size)',
        64  # Default nbs value
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def validate_and_sanitize_args(
    args: Dict[str, Any],
    strict: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Validate and sanitize YOLO training arguments.
    
    This function ensures all arguments are compatible with the current
    Ultralytics API version by:
    1. Removing deprecated arguments
    2. Mapping renamed arguments to their new names
    3. Handling critical replacements (e.g., accumulate -> nbs)
    4. Filtering out invalid arguments
    
    Args:
        args: Dictionary of YOLO training arguments
        strict: If True, raise exception on invalid args. If False, filter them.
        verbose: If True, print warnings for modified arguments
    
    Returns:
        Sanitized dictionary of valid arguments
    
    Raises:
        ValueError: If strict=True and invalid arguments are found
    """
    sanitized = {}
    warnings_list = []
    errors_list = []
    
    for key, value in args.items():
        # Handle removed arguments
        if key in REMOVED_ARGUMENTS:
            warning_msg = f"⚠️  '{key}' has been REMOVED from Ultralytics and will be ignored"
            warnings_list.append(warning_msg)
            continue
        
        # Handle critical replacements
        if key in CRITICAL_REPLACEMENTS:
            new_key, reason, default_value = CRITICAL_REPLACEMENTS[key]
            warning_msg = (
                f"🔄 '{key}' -> '{new_key}': {reason}\n"
                f"   Original value: {value}, Using nbs with default: {default_value}"
            )
            warnings_list.append(warning_msg)
            
            # For accumulate -> nbs: we use the default nbs value
            # The effective batch size behavior is now: batch * (nbs / 64)
            sanitized[new_key] = default_value
            continue
        
        # Handle deprecated mappings
        if key in DEPRECATED_MAPPINGS:
            new_key, reason = DEPRECATED_MAPPINGS[key]
            warning_msg = f"🔄 '{key}' -> '{new_key}': {reason}"
            warnings_list.append(warning_msg)
            
            # Handle inverted boolean logic
            if key in ['hide_labels', 'hide_conf']:
                value = not bool(value)
            
            sanitized[new_key] = value
            continue
        
        # Check if argument is valid
        if key not in VALID_TRAIN_ARGUMENTS:
            error_msg = f"❌ '{key}' is not a valid YOLO argument"
            
            if strict:
                errors_list.append(error_msg)
            else:
                warnings_list.append(f"{error_msg} (will be ignored)")
            continue
        
        # Valid argument - keep it
        sanitized[key] = value
    
    # Print warnings if verbose
    if verbose and warnings_list:
        console.print("\n[yellow]═══ YOLO Arguments Validation ═══[/yellow]")
        for warning in warnings_list:
            console.print(f"[yellow]{warning}[/yellow]")
        console.print()
    
    # Raise errors if strict mode
    if strict and errors_list:
        error_msg = "\n".join(errors_list)
        raise ValueError(f"Invalid YOLO arguments found:\n{error_msg}")
    
    return sanitized


def get_nbs_from_accumulate(batch_size: int, accumulate: int) -> int:
    """
    Calculate appropriate nbs value based on old accumulate logic.
    
    In older YOLO versions, gradient accumulation was controlled by 'accumulate'.
    Now it's handled via 'nbs' (Nominal Batch Size).
    
    The relationship is:
    - Old: effective_batch = batch * accumulate
    - New: nbs controls loss normalization (typically 64)
    
    Args:
        batch_size: Current batch size
        accumulate: Desired accumulation steps (old parameter)
    
    Returns:
        Recommended nbs value
    
    Note:
        In practice, nbs should usually stay at 64 (default).
        This function is provided for reference only.
    """
    effective_batch = batch_size * accumulate
    
    # nbs is typically set to 64 regardless of batch size
    # But for very large effective batches, it could be adjusted
    if effective_batch <= 64:
        return 64
    else:
        # For very large batches, use the effective batch size
        return effective_batch


def check_argument_compatibility(args: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if arguments are compatible with current Ultralytics version.
    
    Args:
        args: Dictionary of YOLO training arguments
    
    Returns:
        Tuple of (is_compatible, error_message)
        If compatible, error_message is empty string
    """
    issues = []
    
    # Check for removed arguments
    for key in args:
        if key in REMOVED_ARGUMENTS:
            issues.append(f"'{key}' has been removed")
        elif key in CRITICAL_REPLACEMENTS:
            new_key, _, _ = CRITICAL_REPLACEMENTS[key]
            issues.append(f"'{key}' should be replaced with '{new_key}'")
        elif key not in VALID_TRAIN_ARGUMENTS and key not in DEPRECATED_MAPPINGS:
            issues.append(f"'{key}' is not a valid argument")
    
    if issues:
        return False, "\n".join(issues)
    
    return True, ""


def print_valid_arguments_reference():
    """Print a reference guide of all valid YOLO training arguments."""
    console.print("\n[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]")
    console.print("[bold cyan]Valid YOLO Training Arguments Reference[/bold cyan]")
    console.print("[bold cyan]═══════════════════════════════════════════════════════════[/bold cyan]\n")
    
    categories = {
        "Core Training": [
            'model', 'data', 'epochs', 'batch', 'imgsz', 'patience',
            'device', 'workers', 'seed', 'verbose'
        ],
        "Optimization": [
            'optimizer', 'lr0', 'lrf', 'momentum', 'weight_decay',
            'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr'
        ],
        "Loss Weights": [
            'box', 'cls', 'dfl', 'pose', 'kobj', 'nbs'
        ],
        "Data Augmentation": [
            'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale',
            'shear', 'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup'
        ],
        "Regularization": [
            'dropout', 'weight_decay'
        ],
    }
    
    for category, args in categories.items():
        console.print(f"[bold green]{category}:[/bold green]")
        for arg in args:
            console.print(f"  • {arg}")
        console.print()
    
    console.print("[bold red]⚠️  REMOVED ARGUMENTS:[/bold red]")
    for arg in REMOVED_ARGUMENTS:
        console.print(f"  ✗ {arg}")
    print()
    
    console.print("[bold yellow]🔄 REPLACED ARGUMENTS:[/bold yellow]")
    for old, (new, reason, default_value) in CRITICAL_REPLACEMENTS.items():
        console.print(f"  {old} → {new}")
        console.print(f"    Reason: {reason}")
        console.print(f"    Default: {default_value}")
    console.print()


# ══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES (for testing)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Example: Validate problematic arguments
    test_args = {
        'epochs': 100,
        'batch': 16,
        'accumulate': 4,  # INVALID - will be converted to nbs
        'label_smoothing': 0.1,  # REMOVED - will be filtered out
        'lr0': 0.01,
        'optimizer': 'SGD',
        'invalid_arg': 'test',  # INVALID - will be filtered out
    }
    
    print("Original arguments:")
    print(test_args)
    
    print("\nValidating and sanitizing...")
    sanitized = validate_and_sanitize_args(test_args, verbose=True)
    
    print("\nSanitized arguments:")
    print(sanitized)
    
    print("\nReference:")
    print_valid_arguments_reference()
