#!/usr/bin/env python3
"""
Compare original SLiCAP_C18.lib with fitted SLiCAP_fitted.lib

This script extracts and compares key EKV parameters
"""

def parse_spice_value(value_str):
    """
    Parse SPICE value with unit suffixes

    Supports: T, G, Meg, M, k, m, u, n, p, f
    Examples: 4.2n → 4.2e-9, 8.8M → 8.8e6, 70m → 0.07, 1.06e+07 → 1.06e7
    """
    # SPICE suffix multipliers
    suffixes = {
        'T': 1e12,
        'G': 1e9,
        'Meg': 1e6,
        'k': 1e3,
        'm': 1e-3,
        'u': 1e-6,
        'n': 1e-9,
        'p': 1e-12,
        'f': 1e-15
    }

    value_str = value_str.strip()

    # Try direct float conversion first (handles scientific notation)
    try:
        return float(value_str)
    except ValueError:
        pass

    # Check for formulas (contains {, *, /)
    # NOTE: Don't check for + or - as they appear in scientific notation (e+07, e-12)
    if any(c in value_str for c in ['{', '*', '/']):
        return None  # Skip formulas

    # Also skip if it contains operators with spaces around them
    if any(op in value_str for op in [' + ', ' - ', ' * ', ' / ']):
        return None

    # Check for suffixes
    # Try 'Meg' first (3 characters)
    if value_str.endswith('Meg'):
        try:
            return float(value_str[:-3]) * suffixes['Meg']
        except ValueError:
            return None

    # Try single-character suffixes
    for suffix, multiplier in suffixes.items():
        if len(suffix) == 1 and value_str.endswith(suffix):
            try:
                return float(value_str[:-1]) * multiplier
            except ValueError:
                return None

    # Special case: 'M' can mean Mega (M) or Milli (m)
    # In SPICE, uppercase M usually means Mega when used with units like Hz, V
    # but context matters. We'll treat standalone M as Mega.
    if value_str.endswith('M') and value_str[-1].isupper():
        try:
            return float(value_str[:-1]) * 1e6
        except ValueError:
            return None

    return None


def extract_params(lib_file):
    """Extract EKV parameters from library file"""
    params = {}
    current_device = None

    with open(lib_file, 'r') as f:
        for line in f:
            line = line.strip()

            # Detect device type from .param block
            if '.param' in line:
                continue

            # Parse parameter lines
            if line.startswith('+'):
                parts = line.split('=')
                if len(parts) >= 2:
                    param_name = parts[0].strip().lstrip('+').strip()
                    value_part = parts[1].split(';')[0].strip()

                    # Try to parse SPICE value with unit suffixes
                    value = parse_spice_value(value_part)
                    if value is not None:
                        params[param_name] = value

    return params


def compare_params(original, fitted):
    """Compare parameters between two libraries"""

    # Key parameters to compare
    nmos_params = [
        'TOX_N18', 'Vth_N18', 'N_s_N18', 'Theta_N18', 'E_CRIT_N18', 'u_0_N18',
        'CGBO_N18', 'CGSO_N18', 'KF_N18', 'AF_N18', 'V_KF_N18'
    ]

    pmos_params = [
        'TOX_P18', 'Vth_P18', 'N_s_P18', 'Theta_P18', 'E_CRIT_P18', 'u_0_P18',
        'CGBO_P18', 'CGSO_P18', 'KF_P18', 'AF_P18', 'V_KF_P18'
    ]

    print("="*80)
    print("COMPARISON: Original vs Fitted EKV Parameters")
    print("="*80)

    # NMOS comparison
    print("\nNMOS (N18) Parameters:")
    print("-"*80)
    print(f"{'Parameter':<15} {'Original':<20} {'Fitted':<20} {'Change (%)':<15}")
    print("-"*80)

    for param in nmos_params:
        orig_val = original.get(param, None)
        fit_val = fitted.get(param, None)

        if orig_val is not None and fit_val is not None:
            change_pct = ((fit_val - orig_val) / orig_val * 100) if orig_val != 0 else 0
            print(f"{param:<15} {orig_val:<20.6e} {fit_val:<20.6e} {change_pct:>+14.2f}%")
        elif fit_val is not None:
            print(f"{param:<15} {'N/A':<20} {fit_val:<20.6e} {'NEW':<15}")
        else:
            print(f"{param:<15} {orig_val or 'N/A':<20} {'N/A':<20} {'MISSING':<15}")

    # PMOS comparison
    print("\nPMOS (P18) Parameters:")
    print("-"*80)
    print(f"{'Parameter':<15} {'Original':<20} {'Fitted':<20} {'Change (%)':<15}")
    print("-"*80)

    for param in pmos_params:
        orig_val = original.get(param, None)
        fit_val = fitted.get(param, None)

        if orig_val is not None and fit_val is not None:
            change_pct = ((fit_val - orig_val) / orig_val * 100) if orig_val != 0 else 0
            print(f"{param:<15} {orig_val:<20.6e} {fit_val:<20.6e} {change_pct:>+14.2f}%")
        elif fit_val is not None:
            print(f"{param:<15} {'N/A':<20} {fit_val:<20.6e} {'NEW':<15}")
        else:
            print(f"{param:<15} {orig_val or 'N/A':<20} {'N/A':<20} {'MISSING':<15}")


if __name__ == "__main__":
    import os

    # Check if files exist
    original_lib = "lib/SLiCAP_C18.lib"
    fitted_lib = "SLiCAP_fitted.lib"

    if not os.path.exists(original_lib):
        print(f"Error: {original_lib} not found!")
        exit(1)

    if not os.path.exists(fitted_lib):
        print(f"Error: {fitted_lib} not found!")
        print("Run: python generate_lib.py --skip-fitting")
        exit(1)

    # Extract parameters
    print("Reading libraries...")
    original = extract_params(original_lib)
    fitted = extract_params(fitted_lib)

    print(f"Original library: {len(original)} parameters")
    print(f"Fitted library:   {len(fitted)} parameters\n")

    # Compare
    compare_params(original, fitted)
