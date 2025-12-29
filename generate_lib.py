#!/usr/bin/env python3
"""
SLiCAP Library Generator

This script generates a SLiCAP-compatible library file from fitted EKV parameters.
It can automatically detect missing parameter files and trigger fitting if needed.

Usage:
    python generate_lib.py                    # Auto-detect and fit if needed
    python generate_lib.py --devices nch pch  # Specify devices
    python generate_lib.py --skip-fitting     # Only generate, don't fit
"""

import os
import sys
import argparse
from datetime import datetime


def load_ekv_parameters(device):
    """Load EKV parameters from file

    Parameters:
    -----------
    device : str - Device type ('nch' or 'pch')

    Returns:
    --------
    dict - Dictionary with EKV parameters
    """
    filename = f'ekv_parameters_{device}.txt'

    if not os.path.exists(filename):
        raise FileNotFoundError(f"EKV parameter file not found: {filename}")

    params = {}
    with open(filename, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                line = line.split('#')[0].strip()
                if '=' in line:
                    key, value = line.split('=')
                    key = key.strip()
                    value = value.strip()
                    try:
                        params[key] = float(value)
                    except ValueError:
                        pass

    required_keys = ['Theta', 'I_0', 'E_CRIT', 'N_s', 'Vth', 'C_OX', 'CGBO', 'CGSO', 'u_0']
    missing_keys = [k for k in required_keys if k not in params]

    if missing_keys:
        raise ValueError(f"Missing required parameters in {filename}: {missing_keys}")

    return params


def load_noise_parameters(device):
    """Load noise parameters from file

    Parameters:
    -----------
    device : str - Device type ('nch' or 'pch')

    Returns:
    --------
    dict - Dictionary with noise parameters, or empty dict if file not found
    """
    filename = f'noise_parameters_{device}.txt'

    if not os.path.exists(filename):
        return {}

    params = {}
    with open(filename, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                line = line.split('#')[0].strip()
                if '=' in line:
                    key, value = line.split('=')
                    key = key.strip()
                    value = value.strip()
                    try:
                        params[key] = float(value)
                    except ValueError:
                        pass

    return params


def check_parameter_files(devices):
    """Check which parameter files exist

    Parameters:
    -----------
    devices : list - List of device types to check

    Returns:
    --------
    tuple - (available_devices, missing_devices)
    """
    available = []
    missing = []

    for dev in devices:
        ekv_file = f'ekv_parameters_{dev}.txt'

        if os.path.exists(ekv_file):
            available.append(dev)
            print(f"✓ Found: {ekv_file}")

            noise_file = f'noise_parameters_{dev}.txt'
            if os.path.exists(noise_file):
                print(f"✓ Found: {noise_file}")
            else:
                print(f"⚠ Missing: {noise_file} (will use default noise parameters)")
        else:
            missing.append(dev)
            print(f"✗ Missing: {ekv_file}")

    return available, missing


def run_fitting(device, lib_path='.lib lib/cr018gpii_v1d0.l TT', ekvlib='lib/SLiCAP_C18.lib'):
    """Run fitting for a device by importing and calling fitting functions

    Parameters:
    -----------
    device : str - Device type ('nch' or 'pch')
    lib_path : str - BSIM library path
    ekvlib : str - EKV library template path
    """
    print(f"\n{'='*60}")
    print(f"Running fitting for {device.upper()}")
    print(f"{'='*60}")

    try:
        from fitting import gmfitting, noisefitting

        print(f"\nStep 1/2: Fitting DC parameters (gm, ft, Ciss)...")
        gmfitting(
            DEV=device,
            LIB=lib_path,
            EKVlib=ekvlib,
            Lmin=0.18e-6, Lmax=10e-6,
            Wmin=0.22e-6, Wmax=50e-6,
            gridnumL=4, gridnumW=4,
            Npts=50,
            IC_min=0.01, IC_max=100
        )

        print(f"\nStep 2/2: Fitting noise parameters (thermal + flicker)...")
        noisefitting(
            DEV=device,
            LIB=lib_path,
            EKVlib=ekvlib,
            Lmin=0.18e-6, Lmax=10e-6,
            Wmin=0.22e-6, Wmax=100e-6,
            gridnumL=4, gridnumW=4,
            Npts_ID=4,
            fmin=0.01,
            IC_min=0.01, IC_max=10
        )

        print(f"\n✓ Fitting completed for {device.upper()}")
        return True

    except ImportError as e:
        print(f"Error: Cannot import fitting module: {e}")
        print("Make sure fitting.py is in the same directory.")
        return False
    except Exception as e:
        print(f"Error during fitting: {e}")
        return False


def generate_library(devices, template_lib='lib/SLiCAP_C18.lib', output_file='SLiCAP_fitted.lib'):
    """Generate SLiCAP library file with fitted parameters

    Parameters:
    -----------
    devices : list - List of device types to include
    template_lib : str - Template library file path
    output_file : str - Output library file name

    Returns:
    --------
    bool - True if successful
    """
    print(f"\n{'='*60}")
    print(f"Generating SLiCAP Library: {output_file}")
    print(f"{'='*60}")

    # Physical constants
    epsilon_0 = 8.8541878128e-12
    epsilon_SiO2 = 3.9

    # Collect all parameter updates for all devices
    param_updates = {}
    fitting_info = {}

    for DEV in devices:
        try:
            # Load fitted DC parameters
            ekv_params = load_ekv_parameters(DEV)

            # Load fitted noise parameters
            noise_params = load_noise_parameters(DEV)

            # Prepare parameters for this device
            suffix = '_N18' if DEV == 'nch' else '_P18'
            dev_upper = DEV[0].upper() + '18'

            # Calculate TOX from C_OX
            TOX = epsilon_0 * epsilon_SiO2 / ekv_params['C_OX']

            # Create parameter replacement dictionary for this device
            device_params = {
                f'TOX{suffix}': f'{TOX:.6e}',
                f'Vth{suffix}': f'{ekv_params["Vth"]:.6f}',
                f'N_s{suffix}': f'{ekv_params["N_s"]:.6f}',
                f'Theta{suffix}': f'{ekv_params["Theta"]:.6f}',
                f'E_CRIT{suffix}': f'{ekv_params["E_CRIT"]:.6e}',
                f'u_0{suffix}': f'{ekv_params["u_0"]:.6e}',
                f'CGBO{suffix}': f'{ekv_params["CGBO"]:.6e}',
                f'CGSO{suffix}': f'{ekv_params["CGSO"]:.6e}',
            }

            # Add noise parameters if available
            if noise_params:
                kf_key = f'KF_{dev_upper}'
                af_key = f'AF_{dev_upper}'
                vkf_key = f'V_KF_{dev_upper}'
                if kf_key in noise_params:
                    device_params[f'KF{suffix}'] = f'{noise_params[kf_key]:.6e}'
                if af_key in noise_params:
                    device_params[f'AF{suffix}'] = f'{noise_params[af_key]:.6f}'
                if vkf_key in noise_params:
                    device_params[f'V_KF{suffix}'] = f'{noise_params[vkf_key]:.6f}'

            param_updates.update(device_params)
            fitting_info[DEV] = dev_upper

            print(f"✓ Loaded parameters for {dev_upper}")

        except FileNotFoundError as e:
            print(f"⚠ Warning: {e}")
            continue
        except Exception as e:
            print(f"⚠ Error loading parameters for {DEV}: {e}")
            continue

    if not param_updates:
        print("\nError: No valid parameters found for any device!")
        return False

    # Read template file and update parameters
    if not os.path.exists(template_lib):
        print(f"\nError: Template library not found: {template_lib}")
        return False

    with open(template_lib, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        updated_line = line

        # Check if this line contains a parameter definition
        if line.strip().startswith('+') and '=' in line:
            # Extract parameter name
            parts = line.split('=')
            if len(parts) >= 2:
                param_part = parts[0].strip().lstrip('+').strip()

                # Check if we have an update for this parameter
                if param_part in param_updates:
                    # Preserve the comment if it exists
                    comment = ''
                    if ';' in line:
                        comment = ' ; ' + line.split(';', 1)[1].strip()

                    # Reconstruct the line with updated value
                    spacing = ' ' * (20 - len(param_part))
                    updated_line = f'+ {param_part}{spacing}= {param_updates[param_part]}{comment}\n'

        updated_lines.append(updated_line)

    # Write updated library
    with open(output_file, 'w') as f:
        # Add header comment
        devices_str = ', '.join([fitting_info[d] for d in fitting_info])
        f.write(f'"SLiCAP CMOS18 library with fitted parameters"\n')
        f.write(f"* Generated by BinkleyFitting from {template_lib}\n")
        f.write(f"* Generation date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"* Fitted devices: {devices_str}\n")
        f.write("*" + "="*60 + "\n")

        # Skip original header lines and write updated content
        skip_header = True
        for i, line in enumerate(updated_lines):
            if i == 0:  # Skip first title line
                continue
            if skip_header and line.strip().startswith('*'):
                continue
            if skip_header and (line.strip() == '' or line.strip().startswith('*')):
                continue
            skip_header = False
            f.write(line)

    print(f"\n{'='*60}")
    print(f"✓ Library file generated: {output_file}")
    print(f"{'='*60}")
    print(f"Fitted devices: {devices_str}")
    print(f"\nUpdated parameters:")
    for DEV in devices:
        if DEV in fitting_info:
            print(f"\n{fitting_info[DEV]}:")
            dev_params = {k: v for k, v in param_updates.items()
                         if ('_N18' in k and DEV == 'nch') or ('_P18' in k and DEV == 'pch')}
            for param, value in sorted(dev_params.items()):
                print(f"  {param:15} = {value}")

    print(f"\n{'='*60}")
    print(f"You can now use this library in SLiCAP:")
    print(f"  .lib {output_file}")
    print(f"{'='*60}\n")

    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate SLiCAP library with fitted EKV parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python generate_lib.py                      # Auto-fit and generate for both devices
  python generate_lib.py --devices nch        # Only NMOS
  python generate_lib.py --skip-fitting       # Generate only, no fitting
  python generate_lib.py --output my.lib      # Custom output filename
        '''
    )

    parser.add_argument('--devices', nargs='+', default=['nch', 'pch'],
                        choices=['nch', 'pch'],
                        help='Device types to include (default: nch pch)')
    parser.add_argument('--template', default='lib/SLiCAP_C18.lib',
                        help='Template library file (default: lib/SLiCAP_C18.lib)')
    parser.add_argument('--output', default='SLiCAP_fitted.lib',
                        help='Output library file (default: SLiCAP_fitted.lib)')
    parser.add_argument('--skip-fitting', action='store_true',
                        help='Skip fitting, only generate library from existing parameters')
    parser.add_argument('--lib', default='.lib lib/cr018gpii_v1d0.l TT',
                        help='BSIM library path for fitting')

    args = parser.parse_args()

    print("="*60)
    print("SLiCAP Library Generator")
    print("="*60)
    print(f"Devices: {', '.join(args.devices)}")
    print(f"Template: {args.template}")
    print(f"Output: {args.output}")
    print("="*60)

    # Check which parameter files exist
    available_devices, missing_devices = check_parameter_files(args.devices)

    # If some devices are missing and fitting is not skipped, run fitting
    if missing_devices and not args.skip_fitting:
        print(f"\n{'='*60}")
        print(f"Missing parameters for: {', '.join(missing_devices)}")
        print(f"Running fitting for missing devices...")
        print(f"{'='*60}")

        for dev in missing_devices:
            success = run_fitting(dev, lib_path=args.lib, ekvlib=args.template)
            if success:
                available_devices.append(dev)
            else:
                print(f"⚠ Warning: Fitting failed for {dev}, will skip this device")

    elif missing_devices and args.skip_fitting:
        print(f"\n⚠ Warning: Missing parameters for {', '.join(missing_devices)}")
        print(f"  Use --skip-fitting=False to run fitting automatically")

    # Generate library if we have at least one device
    if available_devices:
        success = generate_library(
            devices=available_devices,
            template_lib=args.template,
            output_file=args.output
        )

        if success:
            print("✓ Done!")
            return 0
        else:
            print("✗ Library generation failed")
            return 1
    else:
        print("\n✗ Error: No parameter files available and fitting was skipped or failed")
        print("  Run without --skip-fitting to perform fitting first")
        return 1


if __name__ == "__main__":
    sys.exit(main())
