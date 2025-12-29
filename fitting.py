import SLiCAP as sl
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from contextlib import contextmanager
from datetime import datetime

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

sl.ini.ngspice = "ngspice"
sl.ini.disp = 0  # Disable SLiCAP display messages

@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr at OS level"""
    # Save the original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Save copies of the original stdout/stderr
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)

    # Save Python-level stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        # Open devnull
        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        # Redirect stdout and stderr to devnull at OS level
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)

        # Flush before redirecting
        sys.stdout.flush()
        sys.stderr.flush()

        # Redirect Python-level stdout/stderr to devnull
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

        yield

    finally:
        # Flush redirected streams
        sys.stdout.flush()
        sys.stderr.flush()

        # Close Python-level redirected streams
        sys.stdout.close()
        sys.stderr.close()

        # Restore stdout and stderr at OS level
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)

        # Restore Python-level stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

        # Close saved file descriptors and devnull
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)

def createStep(device, biasPar, VS, Npts, Vdiff, Idiff, Istart):
    """
        A function to Generate current list.
        
        Parameters:
        device - includes device information: Reference designator, BISM lib, type, device name, W, L, M
        biasPar - Biasing Method ("IDS" or "VGS")
        VS - Source Voltage
        Npts - Number of points
        Vdiff - Volatge step size for "VGS" method
        Idiff - Current step size for "IDS" method
        Istart - Current step start position
    """
    if device.dev == "nch":
        if biasPar == "VGS":
            step = [VS, Npts, Vdiff]
        elif biasPar == "IDS":
            step = [Istart, Npts, Idiff]
    elif device.dev == "pch":
        if biasPar == "VGS":
            step = [VS, Npts, -Vdiff]
        elif biasPar == "IDS":
            step = [-Istart, Npts, -Idiff]
    return step

def expand_expression(cir, expr_name, keep_params=None):
    """
    Expand expression while keeping certain parameters as symbols

    Parameters:
    cir - SLiCAP circuit object
    expr_name - name of the expression to expand
    keep_params - list of parameter names to keep as symbols (not expand)
    """
    q             = 1.60217662e-19      # Electron charge in [C]
    c             = 2.99792458e+08      # Speed of light in [m/s]
    mu_0          = 4*np.pi*1e-7        # Permeability of vacuum in [H/m]
    epsilon_SiO2  = 3.9                 # Relative permittivity of SiO2 [-]
    k             = 1.38064852e-23      # Boltzmann constant in [J/K]
    epsilon_0     = 1/mu_0/c**2         # permittivity of vacuum in [F/m]
    T             = 300
    Ut            = k*T/q

    constants = {
        'q': q,
        'c': c,
        'mu_0': mu_0,
        'epsilon_SiO2': epsilon_SiO2,
        'k': k,
        'epsilon_0': epsilon_0,
        'T': T,
        'U_T': Ut,
        'Ut': Ut
    }

    if keep_params is None:
        keep_params = []

    expr = cir.getParValue(expr_name, False)

    if not hasattr(expr, 'free_symbols'):
        return expr

    changed = True
    while changed:
        changed = False
        for sym in list(expr.free_symbols):
            sym_name = str(sym)

            # Skip parameters that should be kept as symbols
            if sym_name in keep_params:
                continue

            if sym_name in constants:
                expr = expr.subs(sym, constants[sym_name])
                changed = True
                continue

            try:
                sym_value = cir.getParValue(sym_name, False)
                if isinstance(sym_value, (int, float)) or sym_value.is_number:
                    continue
                if sym_value != sym:
                    expr = expr.subs(sym, sym_value)
                    changed = True
            except:
                pass

    return expr

def create_numeric_function(expr, symbols):
    """
        A function to lambdify the expresion
        
        Parameters:
        expr - expression in SymPy
        symbols - symbols in the expr
    """
    func = sp.lambdify(symbols, expr, modules=['numpy', {'sqrt': np.sqrt}])
    
    # error considerations
    def safe_func(*args):
        try:
            result = func(*args)
        
            if np.isfinite(result) and not np.isnan(result):
                return float(result)
            else:
                print(f"Warning: Calculation null: {result}")
                return 1.0
        except Exception as e:
            print(f"Calculation error: {e} (parameters: {args})")
            return 1.0
    return safe_func

class MultiDimensionEKVFittingProblem(Problem):
    def __init__(self, w_values, l_values, traces_dict, g_m_expr, g_b_expr, V_GS_expr, device_type='nch'):
        """
        Initialize multi-dimensional EKV fitting problem

        Parameters:
        w_values - List of W values to optimize for
        l_values - List of L values to optimize for
        traces_dict - Nested dictionary with format {W: {L: {parameter_data}}}
        g_m_expr, g_b_expr, V_GS_expr - SymPy expressions
        device_type - 'nch' or 'pch'
        """
        self.device_type = device_type

        # Collect all symbols from the expressions
        all_symbols = g_m_expr.free_symbols | g_b_expr.free_symbols | V_GS_expr.free_symbols
        self.all_symbols = sorted(all_symbols, key=lambda x: str(x))

        # Classify symbols: fixed vs fitting
        # Fixed parameters (removed TOX since I_0 should be independent)
        self.fixed_params = ['L', 'W', 'I_D']

        # Fitting parameters (optimization variables) - determined by device type suffix
        suffix = '_N18' if device_type == 'nch' else '_P18'

        # Check if beta is present in the expressions
        self.has_beta = any(f'beta{suffix}' in str(sym) for sym in self.all_symbols)

        # Base fitting parameters - NOW USING I_0 instead of u_0
        self.fitting_params = [
            f'Theta{suffix}',
            f'I_0{suffix}',
            f'E_CRIT{suffix}',
            f'N_s{suffix}',
            f'Vth{suffix}'
        ]

        # Add beta if present in expressions
        if self.has_beta:
            self.fitting_params.append(f'beta{suffix}')
            n_var = 6
            # Boundaries: [Theta, I_0, E_CRIT, N_s, Vth, beta]
            if device_type == 'nch':
                xl = np.array([0.2, 3e-7, 1e6, 1.20, 0.4, 0.5])
                xu = np.array([0.5, 15e-7, 20e6, 1.6, 0.5, 1.5])
            else:
                xl = np.array([0.1, 1e-8, 1e6, 1.20, -0.6, 0.5])
                xu = np.array([0.5, 10e-7, 40e6, 1.6, -0.3, 1.5])
        else:
            n_var = 5
            # Boundaries: [Theta, I_0, E_CRIT, N_s, Vth]
            if device_type == 'nch':
                xl = np.array([0.2, 3e-7, 1e6, 1.20, 0.4])
                xu = np.array([0.5, 15e-7, 20e6, 1.6, 0.5])
            else:
                xl = np.array([0.2, 1e-7, 2e6, 1.20, -0.6])
                xu = np.array([0.5, 10e-7, 40e6, 1.6, -0.4])
        
        super().__init__(n_var=n_var, n_obj=3, n_constr=0, elementwise_evaluation=False)
        
        self.xl = xl
        self.xu = xu

        self.symbol_name_map = {str(sym): sym for sym in self.all_symbols}
        
        # Create lambdify functions using the fixed symbol order
        self.g_m_func = sp.lambdify(self.all_symbols, g_m_expr, modules=['numpy', {'sqrt': np.sqrt}])
        self.g_b_func = sp.lambdify(self.all_symbols, g_b_expr, modules=['numpy', {'sqrt': np.sqrt}])
        self.vgs_func = sp.lambdify(self.all_symbols, V_GS_expr, modules=['numpy', {'sqrt': np.sqrt}])
        
        # Print symbol order for reference
        print(f"Symbol order: {[str(s) for s in self.all_symbols]}")
        print(f"Fitting params: {self.fitting_params}")
        print(f"Beta parameter detected: {self.has_beta}")

        # Define fixed parameters
        self.w_values = w_values
        self.l_values = l_values
        self.traces_dict = traces_dict

    def _build_args(self, fitting_vals, W, L, I_D):
        """
        Build argument list for the lambdified functions based on symbol order
        fitting_vals: [Theta, I_0, E_CRIT, N_s, Vth] or [Theta, I_0, E_CRIT, N_s, Vth, beta]
        """
        suffix = '_N18' if self.device_type == 'nch' else '_P18'

        # Create parameter values dictionary (removed TOX)
        param_values = {
            'L': L,
            'W': W,
            'I_D': I_D,
            f'Theta{suffix}': fitting_vals[0],
            f'I_0{suffix}': fitting_vals[1],
            f'E_CRIT{suffix}': fitting_vals[2],
            f'N_s{suffix}': fitting_vals[3],
            f'Vth{suffix}': fitting_vals[4]
        }

        # Add beta if present
        if self.has_beta:
            param_values[f'beta{suffix}'] = fitting_vals[5]

        args = []
        for sym in self.all_symbols:
            sym_name = str(sym)
            if sym_name in param_values:
                args.append(param_values[sym_name])
            else:
                raise ValueError(f"Unknown symbol: {sym_name}")

        return args

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate model fitting errors across multiple W and L values
        """
        n_points = X.shape[0]
        F = np.zeros((n_points, 3))  # n_obj = 3 (VGS, gm, gb)
        
        for i in range(n_points):
            try:
                fitting_vals = X[i]  # [Theta, u_0, E_CRIT, N_s, Vth] or with beta
                
                # Initialize error accumulators
                all_errors = {'gm': [], 'gb': [], 'vgs': []}
                
                # Iterate through all W,L combinations
                for W in self.w_values:
                    for L in self.l_values:
                        traces = self.traces_dict[W][L]
                        
                        id_data = traces['id']
                        gm_data = traces['gm']
                        gb_data = traces['gb']
                        vgs_data = traces['vgs']
                        
                        # Process each ID point
                        for j, id_val in enumerate(id_data):
                            try:
                                # Build exact parameter list
                                func_args = self._build_args(fitting_vals, W, L, id_val)
                                
                                gm_pred = self.g_m_func(*func_args)
                                gb_pred = self.g_b_func(*func_args)
                                vgs_pred = self.vgs_func(*func_args)
                                
                                # Calculate relative errors
                                if np.isfinite(gm_pred) and gm_data[j] != 0:
                                    all_errors['gm'].append(((gm_pred - gm_data[j]) / gm_data[j])**2)
                                if np.isfinite(gb_pred) and gb_data[j] != 0:
                                    all_errors['gb'].append(((gb_pred - gb_data[j]) / gb_data[j])**2)
                                if np.isfinite(vgs_pred) and vgs_data[j] != 0:
                                    all_errors['vgs'].append(((vgs_pred - vgs_data[j]) / vgs_data[j])**2)
                            except Exception as e:
                                continue
                
                # Calculate RMSE
                f1 = np.sqrt(np.mean(all_errors['gm'])) if all_errors['gm'] else 1e10
                f2 = np.sqrt(np.mean(all_errors['gb'])) if all_errors['gb'] else 1e10
                f3 = np.sqrt(np.mean(all_errors['vgs'])) if all_errors['vgs'] else 1e10
                F[i] = [f1, f2, f3]

            except Exception as e:
                print(f"Parameter {i} error: {e}")
                F[i] = [1e10, 1e10, 1e10]
        
        out["F"] = F

class CissFittingProblem(Problem):
    def __init__(self, w_values, l_values, traces_dict, c_iss_expr, best_solution_stage1, device_type='nch'):
        """
        Initialize Ciss fitting problem (Stage 2)

        Parameters:
        w_values - List of W values to optimize for
        l_values - List of L values to optimize for
        traces_dict - Nested dictionary with format {W: {L: {parameter_data}}}
        c_iss_expr - SymPy expression for c_iss
        best_solution_stage1 - Best solution from stage 1 [Theta, I_0, E_CRIT, N_s, Vth, (beta)]
        device_type - 'nch' or 'pch'
        """
        self.device_type = device_type
        self.best_solution_stage1 = best_solution_stage1

        # Collect all symbols from the expression
        self.all_symbols = sorted(c_iss_expr.free_symbols, key=lambda x: str(x))

        # Fitting parameters (optimization variables)
        suffix = '_N18' if device_type == 'nch' else '_P18'

        # Check if beta is present in stage 1 solution
        self.has_beta = len(best_solution_stage1) == 6

        # Capacitance fitting parameters - 3 PARAMS
        self.fitting_params = [
            f'C_OX{suffix}',
            f'CGBO{suffix}',
            f'CGSO{suffix}'
        ]

        n_var = 3
        # Boundaries: [C_OX, CGBO, CGSO]
        # C_OX: 1e-4 to 20e-3 F/m²
        # CGBO: 1e-12 to 1e-9 F/m (typically ~1p, low sensitivity due to 2*L coefficient)
        # CGSO: 1e-12 to 1e-9 F/m
        xl = np.array([1e-4, 1e-12, 1e-12])
        xu = np.array([20e-3, 1e-9, 1e-9])

        super().__init__(n_var=n_var, n_obj=1, n_constr=0, elementwise_evaluation=False)

        self.xl = xl
        self.xu = xu

        # Create lambdify function for c_iss
        self.c_iss_func = sp.lambdify(self.all_symbols, c_iss_expr, modules=['numpy', {'sqrt': np.sqrt}])

        # Print symbol order for reference
        print(f"\n=== Stage 2: Ciss Fitting ===")
        print(f"Symbol order: {[str(s) for s in self.all_symbols]}")
        print(f"Fitting params: {self.fitting_params}")

        # Define fixed parameters (removed TOX)
        self.w_values = w_values
        self.l_values = l_values
        self.traces_dict = traces_dict

    def _build_args(self, fitting_vals, W, L, I_D):
        """
        Build argument list for the lambdified function based on symbol order
        fitting_vals: [C_OX, CGBO, CGSO]
        """
        suffix = '_N18' if self.device_type == 'nch' else '_P18'

        # Extract stage 1 parameters
        Theta_best = self.best_solution_stage1[0]
        I_0_best = self.best_solution_stage1[1]
        E_CRIT_best = self.best_solution_stage1[2]
        N_s_best = self.best_solution_stage1[3]
        Vth_best = self.best_solution_stage1[4]

        # Create parameter values dictionary (removed TOX)
        param_values = {
            'L': L,
            'W': W,
            'I_D': I_D,
            f'Theta{suffix}': Theta_best,
            f'I_0{suffix}': I_0_best,
            f'E_CRIT{suffix}': E_CRIT_best,
            f'N_s{suffix}': N_s_best,
            f'Vth{suffix}': Vth_best,
            f'C_OX{suffix}': fitting_vals[0],
            f'CGBO{suffix}': fitting_vals[1],
            f'CGSO{suffix}': fitting_vals[2]
        }

        # Add beta if present
        if self.has_beta:
            beta_best = self.best_solution_stage1[5]
            param_values[f'beta{suffix}'] = beta_best

        args = []
        for sym in self.all_symbols:
            sym_name = str(sym)
            if sym_name in param_values:
                args.append(param_values[sym_name])
            else:
                raise ValueError(f"Unknown symbol: {sym_name}")

        return args

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate Ciss fitting errors across multiple W and L values
        """
        n_points = X.shape[0]
        F = np.zeros((n_points, 1))  # n_obj = 1 (ciss only)

        for i in range(n_points):
            try:
                fitting_vals = X[i]  # [CGBO, CGSO]

                # Initialize error accumulator
                all_errors = []

                # Iterate through all W,L combinations
                for W in self.w_values:
                    for L in self.l_values:
                        traces = self.traces_dict[W][L]

                        id_data = traces['id']
                        ciss_data = traces['ciss']

                        # Process each ID point
                        for j, id_val in enumerate(id_data):
                            try:
                                # Build exact parameter list
                                func_args = self._build_args(fitting_vals, W, L, id_val)

                                ciss_pred = self.c_iss_func(*func_args)

                                # Calculate relative errors
                                if np.isfinite(ciss_pred) and ciss_data[j] != 0:
                                    all_errors.append(((ciss_pred - ciss_data[j]) / ciss_data[j])**2)
                            except Exception as e:
                                continue

                # Calculate RMSE
                f1 = np.sqrt(np.mean(all_errors)) if all_errors else 1e10
                F[i] = [f1]

            except Exception as e:
                print(f"Parameter {i} error: {e}")
                F[i] = [1e10]

        out["F"] = F

def getNgspiceResult(W_values, L_values, LIB, DEV, Npts, IC_min, IC_max):
    refDes   = 'M1'
    biasPar  = "IDS"
    M        = 1                                    # Number of devices in parallel

    # Use parameters passed from caller (don't override DEV, IC_min, IC_max, Npts)
    # DEV is passed as parameter, don't override it here!

    # actually not relavant here
    Vdiff    = 0.01
    ID       = 1E-6                      # Drain current single point, if not use 'step' in function 'getOPid'
    VP       = 1.8
    VS       = 0
    VD       = 0.9
    VB       = 0

    if DEV == "pch":
        VS = VP
        VB = VP
        VD = VP - VD

    vgs_sign = -1 if DEV == "pch" else 1
    freq     = 1E5

    if DEV == "nch":
        I0       = 6e-07                        # Approximated value of technology current to estmate current range
    else:
        I0       = 1.5e-07                      # TBD (I_0 for PMOS)
    traces_dict = {}
    for W_v in W_values:
        traces_dict[W_v] = {}
        
        for L_v in L_values:
            
            # IC = 0.01
            Imin = IC_min * I0 * W_v / L_v
            # IC = 100
            Imax = IC_max* I0 * W_v / L_v

            # Create Current Step
            Idiff = (Imax - Imin) / Npts
            
            # Create the device and the current steps
            device = sl.MOS(refDes, LIB, DEV, W_v, L_v, M)
            step = createStep(device, biasPar, VS, Npts, Vdiff, Idiff, Imin)

            # use NGspice to get BSIM data
            with suppress_output():
                device.getOPid(ID, VD, VS, VB, freq, step)

            derivedParams = {}
            derivedParams['fT'] = device.params['ggs']/(2*np.pi*device.params['cgg'])
            derivedParams['gmId'] = device.params['ggs']/device.params['i(ids)']
            # Store results in the nested dictionary
            traces_dict[W_v][L_v] = {
                'id': np.array(device.params['i(ids)'], dtype=float),
                'gm': np.array(device.params['ggs'], dtype=float),
                'gb': np.array(device.params['gbs'], dtype=float),
                'ft': np.array(derivedParams['fT'], dtype=float),
                'go': np.array(device.params['gdd'], dtype=float),
                'vgs': np.array(device.params['v(vgs)'], dtype=float)*vgs_sign,
                'gmId': np.array(derivedParams['gmId'], dtype=float),
                'ciss': np.array(device.params['cgg'], dtype=float)
            }
    return traces_dict

def createEKVCir(DEV, EKVlib):
        if DEV == "nch":
            cirText = "EKV model\n.lib %s\nX1 D G S 0 CMOS18N W={W} L={L} ID={I_D} \n.param I_D=0 W=0.22e6 L=0.18e6\n.end\n"%(EKVlib)
        elif DEV == "pch":
            cirText = "EKV model\n.lib %s\nX1 D G S 0 CMOS18P W={W} L={L} ID={-I_D} \n.param I_D=0 W=0.22e6 L=0.18e6\n.end\n"%(EKVlib)
        f = open('cir/MOS.cir', 'w')
        f.write(cirText)
        f.close()

def createEKVNoiseCir(DEV, EKVlib):
        """Create circuit file for noise subcircuit (MN18_noise or MP18_noise)"""
        if DEV == "nch":
            cirText = "EKV noise model\n.lib %s\nX1 ext comm int MN18_noise ID={I_D} IG=0 W={W} L={L} \n.param I_D=0 W=0.22e6 L=0.18e6\n.end\n"%(EKVlib)
        elif DEV == "pch":
            cirText = "EKV noise model\n.lib %s\nX1 ext comm int MP18_noise ID={-I_D} IG=0 W={W} L={L} \n.param I_D=0 W=0.22e6 L=0.18e6\n.end\n"%(EKVlib)
        f = open('cir/MOS_noise.cir', 'w')
        f.write(cirText)
        f.close()

def findSolution(problem):
    sampling = LHS()
    algorithm = NSGA2(
        pop_size=150,
        n_offsprings=45,
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                algorithm,
                ('n_gen', 100),
                seed=1,
                verbose=True)

    optimal_solutions = res.X
    optimal_objectives = res.F
    errors_params = optimal_objectives[:, :3]

    combined_error = np.sqrt(np.sum(np.square(errors_params), axis=1))

    weights = np.array([10.0, 1.0, 0.5])
    weights = weights / np.sum(weights)
    weighted_combined_error = np.sum(weights * errors_params, axis=1)

    min_combined_idx = np.argmin(combined_error)
    min_weighted_idx = np.argmin(weighted_combined_error)

    # Find minimum for each objective
    min_indices = [np.argmin(optimal_objectives[:, i]) for i in range(3)]
    obj_names = ["gm", "gb", "vgs"]

    # Extract best solution parameters
    best_idx = min_weighted_idx
    best_solution = optimal_solutions[min_weighted_idx]

    if problem.has_beta:
        beta_best = best_solution[5]
    else:
        beta_best = None
    suffix = '_N18' if problem.device_type == 'nch' else '_P18'
    if problem.has_beta:
        print(f"beta{suffix}: {beta_best:.6e}")
    print(f"Weighted combined error: {weighted_combined_error[min_weighted_idx]:.6f}")
    print(f"Euclidean distance error: {combined_error[min_weighted_idx]:.6f}")

    return best_solution

def findSolutionCiss(problem):
    """Find solution for Ciss fitting (single objective optimization)"""
    sampling = LHS()

    # Use simpler algorithm for single-objective optimization
    from pymoo.algorithms.soo.nonconvex.ga import GA

    algorithm = GA(
        pop_size=100,
        sampling=sampling,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    res = minimize(problem,
                   algorithm,
                   ('n_gen', 50),
                   seed=1,
                   verbose=True)

    best_solution = res.X
    best_objective = res.F

    suffix = '_N18' if problem.device_type == 'nch' else '_P18'

    print(f"\n=== Stage 2 Results ===")
    print(f"C_OX{suffix}: {best_solution[0]:.6e}")
    print(f"CGBO{suffix}: {best_solution[1]:.6e}")
    print(f"CGSO{suffix}: {best_solution[2]:.6e}")
    print(f"Ciss RMSE error: {best_objective[0]:.6f}")

    return best_solution

def generate_ciss_predictions(best_solution_stage2, problem2):
    """Generate Ciss predictions using Stage 2 solution"""
    predictions = {}

    C_OX_best = best_solution_stage2[0]
    CGBO_best = best_solution_stage2[1]
    CGSO_best = best_solution_stage2[2]
    fitting_vals = [C_OX_best, CGBO_best, CGSO_best]

    for W in problem2.w_values:
        predictions[W] = {}

        for L in problem2.l_values:
            traces = problem2.traces_dict[W][L]

            id_data = traces['id']
            ciss_data = traces['ciss']

            ciss_pred = []

            for id_val in id_data:
                try:
                    func_args = problem2._build_args(fitting_vals, W, L, id_val)
                    ciss_val = problem2.c_iss_func(*func_args)
                    ciss_pred.append(ciss_val)
                except Exception as e:
                    ciss_pred.append(None)

            # Filter valid indices
            valid_indices = [i for i in range(len(id_data)) if ciss_pred[i] is not None]

            valid_id = [id_data[i] for i in valid_indices]
            valid_ciss_pred = [ciss_pred[i] for i in valid_indices]
            valid_ciss_data = [ciss_data[i] for i in valid_indices]

            predictions[W][L] = {
                'id': valid_id,
                'ciss_pred': valid_ciss_pred,
                'ciss_data': valid_ciss_data
            }

    return predictions

def calculate_ciss_error_metrics(predictions, problem2):
    """Calculate Ciss error metrics for each W,L combination"""
    error_metrics = {}

    for W in problem2.w_values:
        error_metrics[W] = {}

        for L in problem2.l_values:
            pred = predictions[W][L]

            pred_values = pred['ciss_pred']
            data_values = pred['ciss_data']

            metrics = {}
            metrics['ciss_rmse'] = calculate_relative_rmse(pred_values, data_values)
            metrics['ciss_mape'] = calculate_mape(pred_values, data_values)
            metrics['ciss_mae'] = calculate_mae(pred_values, data_values)

            error_metrics[W][L] = metrics

    return error_metrics

def resultCissParam(best_solution_stage2, DEV, problem2):
    """Display Stage 2 Ciss fitting results with error metrics"""
    C_OX_best = best_solution_stage2[0]
    CGBO_best = best_solution_stage2[1]
    CGSO_best = best_solution_stage2[2]

    suffix = '_N18' if problem2.device_type == 'nch' else '_P18'

    print(f"\nC_OX{suffix}: {C_OX_best:.6e}")
    print(f"CGBO{suffix}: {CGBO_best:.6e}")
    print(f"CGSO{suffix}: {CGSO_best:.6e}")

    # Generate predictions and calculate error metrics
    predictions = generate_ciss_predictions(best_solution_stage2, problem2)
    error_metrics = calculate_ciss_error_metrics(predictions, problem2)

    # Print error summary
    print("\nCiss Error metrics summary:")
    print(f"{'W (μm)':10} {'L (μm)':10} {'Ciss RMSE (%)':18} {'Ciss MAPE (%)':18}")
    print("-" * 60)

    for W in problem2.w_values:
        for L in problem2.l_values:
            metrics = error_metrics[W][L]
            w_str = f"{W*1e6:.1f}"
            l_str = f"{L*1e6:.3f}"
            rmse = metrics['ciss_rmse'] * 100
            mape = metrics['ciss_mape']
            print(f"{w_str:10} {l_str:10} {rmse:18.2f} {mape:18.2f}")

    # Overall assessment
    all_ciss_rmse = [metrics['ciss_rmse'] for W in error_metrics for metrics in error_metrics[W].values()]
    all_ciss_mape = [metrics['ciss_mape'] for W in error_metrics for metrics in error_metrics[W].values()]

    avg_ciss_rmse = np.mean(all_ciss_rmse)
    avg_ciss_mape = np.mean(all_ciss_mape)

    print(f"\nAverage Ciss RMSE across all sizes: {avg_ciss_rmse*100:.2f}%")
    print(f"Average Ciss MAPE across all sizes: {avg_ciss_mape:.2f}%")

def calculate_relative_rmse(predicted, measured):
    """Calculate relative RMSE"""
    return np.sqrt(np.mean([((p-m)/m)**2 for p, m in zip(predicted, measured)]))

def calculate_mape(predicted, measured):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean([np.abs((p-m)/m)*100 for p, m in zip(predicted, measured)])

def calculate_mae(predicted, measured):
    """Calculate Mean Absolute Error"""
    return np.mean([np.abs(p-m) for p, m in zip(predicted, measured)])

def generate_predictions(best_solution, problem):
    """Generate predictions using best solution"""
    predictions = {}

    # Extract parameters based on whether beta is present
    Theta_best = best_solution[0]
    u_0_best = best_solution[1]
    E_CRIT_best = best_solution[2]
    N_s_best = best_solution[3]
    Vth_best = best_solution[4]

    if problem.has_beta:
        beta_best = best_solution[5]
        fitting_vals = [Theta_best, u_0_best, E_CRIT_best, N_s_best, Vth_best, beta_best]
    else:
        fitting_vals = [Theta_best, u_0_best, E_CRIT_best, N_s_best, Vth_best]

    for W in problem.w_values:
        predictions[W] = {}

        for L in problem.l_values:
            traces = problem.traces_dict[W][L]

            id_data = traces['id']
            gm_data = traces['gm']
            gb_data = traces['gb']
            vgs_data = traces['vgs']

            gm_pred = []
            gb_pred = []
            vgs_pred = []

            for id_val in id_data:
                try:
                    func_args = problem._build_args(fitting_vals, W, L, id_val)

                    gm_val = problem.g_m_func(*func_args)
                    gb_val = problem.g_b_func(*func_args)
                    vgs_val = problem.vgs_func(*func_args)

                    gm_pred.append(gm_val)
                    gb_pred.append(gb_val)
                    vgs_pred.append(vgs_val)
                except Exception as e:
                    gm_pred.append(None)
                    gb_pred.append(None)
                    vgs_pred.append(None)

            # Filter valid indices
            valid_indices = [i for i in range(len(id_data))
                            if gm_pred[i] is not None and gb_pred[i] is not None and vgs_pred[i] is not None]

            valid_id = [id_data[i] for i in valid_indices]
            valid_gm_pred = [gm_pred[i] for i in valid_indices]
            valid_gb_pred = [gb_pred[i] for i in valid_indices]
            valid_vgs_pred = [vgs_pred[i] for i in valid_indices]

            valid_gm_data = [gm_data[i] for i in valid_indices]
            valid_gb_data = [gb_data[i] for i in valid_indices]
            valid_vgs_data = [vgs_data[i] for i in valid_indices]

            predictions[W][L] = {
                'id': valid_id,
                'gm_pred': valid_gm_pred,
                'gb_pred': valid_gb_pred,
                'vgs_pred': valid_vgs_pred,
                'gm_data': valid_gm_data,
                'gb_data': valid_gb_data,
                'vgs_data': valid_vgs_data
            }

    return predictions

def calculate_error_metrics(predictions, problem):
    """Calculate error metrics for each W,L combination"""
    error_metrics = {}

    for W in problem.w_values:
        error_metrics[W] = {}

        for L in problem.l_values:
            pred = predictions[W][L]

            params = ['gm', 'gb', 'vgs']
            metrics = {}

            for param in params:
                pred_values = pred[f'{param}_pred']
                data_values = pred[f'{param}_data']

                metrics[f'{param}_rmse'] = calculate_relative_rmse(pred_values, data_values)
                metrics[f'{param}_mape'] = calculate_mape(pred_values, data_values)
                metrics[f'{param}_mae'] = calculate_mae(pred_values, data_values)

            # Combined errors
            rmse_values = [metrics[f'{param}_rmse'] for param in params]
            mape_values = [metrics[f'{param}_mape'] for param in params]

            metrics['combined_rmse'] = np.sqrt(np.sum(np.square(rmse_values)))
            metrics['combined_mape'] = np.mean(mape_values)

            error_metrics[W][L] = metrics

    return error_metrics

def plot_performance_analysis(error_metrics, problem):
    """Plot Model Performance Analysis Across Device Sizes"""
    plt.figure(figsize=(18, 15))

    w_labels = [f"{W*1e6:.1f}μm" for W in problem.w_values]
    l_labels = [f"{L*1e6:.3f}μm" for L in problem.l_values]

    # Combined RMSE and MAPE heatmaps
    rmse_data = np.array([[error_metrics[W][L]['combined_rmse'] * 100
                          for L in problem.l_values]
                          for W in problem.w_values])

    mape_data = np.array([[error_metrics[W][L]['combined_mape']
                          for L in problem.l_values]
                          for W in problem.w_values])

    # Individual parameter MAPE data
    gm_mape_data = np.array([[error_metrics[W][L]['gm_mape']
                              for L in problem.l_values]
                              for W in problem.w_values])

    gb_mape_data = np.array([[error_metrics[W][L]['gb_mape']
                              for L in problem.l_values]
                              for W in problem.w_values])

    vgs_mape_data = np.array([[error_metrics[W][L]['vgs_mape']
                               for L in problem.l_values]
                               for W in problem.w_values])

    max_mape = max(np.max(gm_mape_data), np.max(gb_mape_data), np.max(vgs_mape_data))
    vmin = 0
    vmax = min(max_mape, max_mape * 1.2)

    # Plot combined RMSE
    plt.subplot(3, 2, 1)
    sns.heatmap(rmse_data, annot=True, fmt=".2f", xticklabels=l_labels, yticklabels=w_labels, cmap="YlGnBu")
    plt.xlabel('Channel Length (L)')
    plt.ylabel('Channel Width (W)')
    plt.title('Combined RMSE (%)')

    # Plot combined MAPE
    plt.subplot(3, 2, 2)
    sns.heatmap(mape_data, annot=True, fmt=".2f", xticklabels=l_labels, yticklabels=w_labels, cmap="YlGnBu")
    plt.xlabel('Channel Length (L)')
    plt.ylabel('Channel Width (W)')
    plt.title('Combined MAPE (%)')

    # Plot gm MAPE
    plt.subplot(3, 2, 3)
    sns.heatmap(gm_mape_data, annot=True, fmt=".2f", xticklabels=l_labels, yticklabels=w_labels,
                cmap="YlGnBu", vmin=vmin, vmax=vmax)
    plt.xlabel('Channel Length (L)')
    plt.ylabel('Channel Width (W)')
    plt.title('Transconductance (gm) MAPE (%)')

    # Plot gb MAPE
    plt.subplot(3, 2, 4)
    sns.heatmap(gb_mape_data, annot=True, fmt=".2f", xticklabels=l_labels, yticklabels=w_labels,
                cmap="YlGnBu", vmin=vmin, vmax=vmax)
    plt.xlabel('Channel Length (L)')
    plt.ylabel('Channel Width (W)')
    plt.title('Body Effect (gb) MAPE (%)')

    # Plot vgs MAPE
    plt.subplot(3, 2, 5)
    sns.heatmap(vgs_mape_data, annot=True, fmt=".2f", xticklabels=l_labels, yticklabels=w_labels,
                cmap="YlGnBu", vmin=vmin, vmax=vmax)
    plt.xlabel('Channel Length (L)')
    plt.ylabel('Channel Width (W)')
    plt.title('Gate-Source Voltage (vgs) MAPE (%)')

    # Parameter error comparison bar chart
    params = ['gm', 'gb', 'vgs']
    param_titles = ['Transconductance (gm)', 'Body Effect (gb)', 'Gate-Source Voltage (vgs)']

    param_mape_avg = []
    for param in params:
        values = [error_metrics[W][L][f'{param}_mape']
                  for W in problem.w_values
                  for L in problem.l_values]
        param_mape_avg.append(np.mean(values))

    plt.subplot(3, 2, 6)
    bars = plt.bar(param_titles, param_mape_avg)
    plt.ylabel('Average MAPE (%)')
    plt.title('Parameter Error Comparison')
    plt.xticks(rotation=15)
    plt.ylim(0, max(param_mape_avg) * 1.2)

    for bar, val in zip(bars, param_mape_avg):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.2f}%", ha='center')

    plt.tight_layout()
    plt.suptitle('Model Performance Analysis Across Device Sizes', fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.show()

def resultOPParm(best_solution, DEV, problem):
    Theta_best = best_solution[0]
    I_0_best = best_solution[1]
    E_CRIT_best = best_solution[2]
    N_s_best = best_solution[3]
    Vth_best = best_solution[4]

    if problem.has_beta:
        beta_best = best_solution[5]
    else:
        beta_best = None

    suffix = '_N18' if problem.device_type == 'nch' else '_P18'

    print(f"Theta{suffix}: {Theta_best:.6e}")
    print(f"I_0{suffix}: {I_0_best:.6e}")
    print(f"E_CRIT{suffix}: {E_CRIT_best:.6e}")
    print(f"N_s{suffix}: {N_s_best:.6f}")
    print(f"Vth{suffix}: {Vth_best:.6e}")

    # Generate predictions and calculate error metrics
    predictions = generate_predictions(best_solution, problem)
    error_metrics = calculate_error_metrics(predictions, problem)

    # Print error summary
    print("\nError metrics summary:")
    print(f"{'W (μm)':10} {'L (μm)':10} {'Combined RMSE (%)':18} {'Combined MAPE (%)':18}")
    print("-" * 60)

    for W in problem.w_values:
        for L in problem.l_values:
            metrics = error_metrics[W][L]
            w_str = f"{W*1e6:.1f}"
            l_str = f"{L*1e6:.3f}"
            rmse = metrics['combined_rmse'] * 100
            mape = metrics['combined_mape']
            print(f"{w_str:10} {l_str:10} {rmse:18.2f} {mape:18.2f}")

    # Overall assessment
    all_combined_rmse = [metrics['combined_rmse'] for W in error_metrics for metrics in error_metrics[W].values()]
    all_combined_mape = [metrics['combined_mape'] for W in error_metrics for metrics in error_metrics[W].values()]

    avg_combined_rmse = np.mean(all_combined_rmse)
    avg_combined_mape = np.mean(all_combined_mape)

    print(f"\nAverage combined RMSE across all sizes: {avg_combined_rmse*100:.2f}%")
    print(f"Average combined MAPE across all sizes: {avg_combined_mape:.2f}%")

def gmfitting(DEV='nch', LIB='.lib lib/cr018gpii_v1d0.l TT', EKVlib='SLiCAP_C18.lib',
              Lmin=0.18e-6, Lmax=10e-6, Wmin=0.22e-6, Wmax=50e-6,
              gridnumL=4, gridnumW=4, Npts=50, IC_min=0.01, IC_max=100):
    """Two-stage EKV model fitting: Stage 1 (gm, gb, vgs) -> Stage 2 (ciss)

    Parameters:
    -----------
    DEV : str - Device type ('nch' or 'pch')
    LIB : str - BSIM library path
    EKVlib : str - EKV library filename
    Lmin, Lmax : float - Min/max channel length (m)
    Wmin, Wmax : float - Min/max channel width (m)
    gridnumL, gridnumW : int - Number of grid points
    Npts : int - Number of bias points per device
    IC_min, IC_max : float - Inversion coefficient range

    Returns:
    --------
    best_solution_stage1 : array - Stage 1 fitted parameters
    best_solution_stage2 : array - Stage 2 fitted parameters
    """

    L_values = np.geomspace(Lmin, Lmax, gridnumL)
    W_values = np.geomspace(Wmin, Wmax, gridnumW)

    # Get NGSPICE simulation results (includes ciss data)
    print("=== Getting NGSPICE simulation data ===")
    traces_dict = getNgspiceResult(W_values, L_values, LIB, DEV, Npts, IC_min, IC_max)

    # Initialize EKV circuit
    with suppress_output():
        prj = sl.initProject(f"BinkleyFitting")
        createEKVCir(DEV, EKVlib)
        cir = sl.makeCircuit('cir/MOS.cir', imgWidth=None)

    # Extract expressions from EKV model
    # For Stage 1: Keep I_0 as independent symbol (don't expand to u_0*C_OX)
    #              Keep C_OX as symbol (don't expand to epsilon/TOX)
    #              This makes I_0 an independent fitting parameter
    suffix = '_N18' if DEV == 'nch' else '_P18'
    keep_params_stage1 = [f'I_0{suffix}', f'C_OX{suffix}']

    g_m = expand_expression(cir, 'g_m_X1', keep_params=keep_params_stage1)
    g_b = expand_expression(cir, 'g_b_X1', keep_params=keep_params_stage1)
    V_GS = expand_expression(cir, 'V_GS_X1', keep_params=keep_params_stage1)

    # For Stage 2: Keep I_0 and C_OX as symbols
    #              C_OX will be a fitting parameter in Stage 2
    keep_params_stage2 = [f'I_0{suffix}', f'C_OX{suffix}']
    c_iss = expand_expression(cir, 'c_iss_X1', keep_params=keep_params_stage2)

    # ========== Stage 1: Fit gm, gb, vgs ==========
    print("\n" + "="*60)
    print("=== Stage 1: Fitting gm, gb, vgs ===")
    print("="*60)

    # Debug: Print expressions to verify I_0 is kept as symbol
    print(f"\nDebug - Stage 1 expressions:")
    print(f"g_m symbols: {sorted([str(s) for s in g_m.free_symbols])}")
    print(f"g_b symbols: {sorted([str(s) for s in g_b.free_symbols])}")
    print(f"V_GS symbols: {sorted([str(s) for s in V_GS.free_symbols])}")

    problem1 = MultiDimensionEKVFittingProblem(W_values, L_values, traces_dict, g_m, g_b, V_GS, DEV)
    best_solution_stage1 = findSolution(problem1)

    # Display Stage 1 results
    resultOPParm(best_solution_stage1, DEV, problem1)

    # ========== Stage 2: Fit ciss using Stage 1 results ==========
    print("\n" + "="*60)
    print("=== Stage 2: Fitting ciss (C_OX, CGBO, CGSO) ===")
    print("="*60)

    # Debug: Print expressions to verify C_OX is kept as symbol
    print(f"\nDebug - Stage 2 expressions:")
    print(f"c_iss symbols: {sorted([str(s) for s in c_iss.free_symbols])}")

    problem2 = CissFittingProblem(W_values, L_values, traces_dict, c_iss, best_solution_stage1, DEV)
    best_solution_stage2 = findSolutionCiss(problem2)

    # Display Stage 2 results with error metrics
    resultCissParam(best_solution_stage2, DEV, problem2)

    # Display combined results
    print("\n" + "="*60)
    print("=== Final Combined Results ===")
    print("="*60)
    displayCombinedResults(best_solution_stage1, best_solution_stage2, DEV, problem1, problem2,
                          LIB, EKVlib, Lmin, Lmax, Wmin, Wmax, IC_min, IC_max)

    return best_solution_stage1, best_solution_stage2

def displayCombinedResults(best_solution_stage1, best_solution_stage2, DEV, problem1, problem2,
                          LIB, EKVlib, Lmin, Lmax, Wmin, Wmax, IC_min, IC_max):
    """Display combined results from both stages and calculate u_0

    Parameters:
    -----------
    best_solution_stage1 : array - Stage 1 solution
    best_solution_stage2 : array - Stage 2 solution
    DEV : str - Device type
    problem1 : Problem - Stage 1 problem instance
    problem2 : Problem - Stage 2 problem instance
    LIB : str - BSIM library path
    EKVlib : str - EKV library filename
    Lmin, Lmax : float - Channel length range
    Wmin, Wmax : float - Channel width range
    IC_min, IC_max : float - IC range
    """
    suffix = '_N18' if DEV == 'nch' else '_P18'

    # Extract Stage 1 parameters
    Theta_best = best_solution_stage1[0]
    I_0_best = best_solution_stage1[1]
    E_CRIT_best = best_solution_stage1[2]
    N_s_best = best_solution_stage1[3]
    Vth_best = best_solution_stage1[4]
    beta_best = best_solution_stage1[5] if problem1.has_beta else None

    # Extract Stage 2 parameters
    C_OX_best = best_solution_stage2[0]
    CGBO_best = best_solution_stage2[1]
    CGSO_best = best_solution_stage2[2]

    # Calculate u_0 from I_0 and C_OX
    # I_0 = 2 * N_s * u_0 * C_OX * U_T^2
    # u_0 = I_0 / (2 * N_s * C_OX * U_T^2)
    k = 1.38064852e-23
    q = 1.60217662e-19
    T = 300
    U_T = k * T / q
    u_0_best = I_0_best / (2 * N_s_best * C_OX_best * U_T**2)

    print("\n=== All Fitted Parameters ===")
    print(f"Theta{suffix}: {Theta_best:.6e}")
    print(f"I_0{suffix}: {I_0_best:.6e}")
    print(f"E_CRIT{suffix}: {E_CRIT_best:.6e}")
    print(f"N_s{suffix}: {N_s_best:.6f}")
    print(f"Vth{suffix}: {Vth_best:.6e}")

    if problem1.has_beta:
        print(f"beta{suffix}: {beta_best:.6e}")

    print(f"C_OX{suffix}: {C_OX_best:.6e}")
    print(f"CGBO{suffix}: {CGBO_best:.6e}")
    print(f"CGSO{suffix}: {CGSO_best:.6e}")
    print(f"\n=== Derived Parameter ===")
    print(f"u_0{suffix}: {u_0_best:.6e}  (calculated from I_0 and C_OX)")

    # Save to file
    # Save parameters with metadata
    with open(f'ekv_parameters_{DEV}.txt', 'w') as f:
        f.write(f"# EKV Model Parameters for {DEV}\n")
        f.write(f"# Fitting metadata\n")
        f.write(f"# LIB = {LIB}\n")
        f.write(f"# EKVlib = {EKVlib}\n")
        f.write(f"# Device = {DEV}\n")
        f.write(f"# Date = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Fitting range: L=[{Lmin:.2e}, {Lmax:.2e}], W=[{Wmin:.2e}, {Wmax:.2e}]\n")
        f.write(f"# IC range: [{IC_min}, {IC_max}]\n")
        f.write(f"\n# Stage 1: DC parameters\n")
        f.write(f"Theta = {Theta_best}\n")
        f.write(f"I_0 = {I_0_best}\n")
        f.write(f"E_CRIT = {E_CRIT_best}\n")
        f.write(f"N_s = {N_s_best}\n")
        f.write(f"Vth = {Vth_best}\n")
        if problem1.has_beta:
            f.write(f"beta = {beta_best}\n")
        f.write(f"\n# Stage 2: Capacitance parameters\n")
        f.write(f"C_OX = {C_OX_best}\n")
        f.write(f"CGBO = {CGBO_best}\n")
        f.write(f"CGSO = {CGSO_best}\n")
        f.write(f"\n# Derived parameters\n")
        f.write(f"u_0 = {u_0_best}  # Calculated from I_0 / (2 * N_s * C_OX * U_T^2)\n")

    print(f"\nParameters saved to 'ekv_parameters_{DEV}.txt'")

def noisefitting(DEV='nch', LIB='.lib lib/cr018gpii_v1d0.l TT', EKVlib='SLiCAP_C18.lib',
                 Lmin=0.18e-6, Lmax=10e-6, Wmin=0.22e-6, Wmax=100e-6,
                 gridnumL=4, gridnumW=4, Npts_ID=4, fmin=0.01, IC_min=0.01, IC_max=10):
    """Two-stage noise fitting: Stage 1 (Thermal) -> Stage 2 (Flicker)
    Only uses devices where W >= L

    Parameters:
    -----------
    DEV : str - Device type ('nch' or 'pch')
    LIB : str - BSIM library path
    EKVlib : str - EKV library filename
    Lmin, Lmax : float - Min/max channel length (m)
    Wmin, Wmax : float - Min/max channel width (m)
    gridnumL, gridnumW : int - Number of grid points
    Npts_ID : int - Number of bias points per device
    fmin : float - Minimum frequency for noise extraction (Hz)
    IC_min, IC_max : float - Inversion coefficient range

    Returns:
    --------
    thermal_results : list - Thermal noise analysis results
    flicker_results : dict - Flicker noise fitting results
    """

    L_values = np.geomspace(Lmin, Lmax, gridnumL)
    W_values = np.geomspace(Wmin, Wmax, gridnumW)

    # Load previous fitted parameters
    print("=== Loading previous EKV parameters ===")
    ekv_params = load_ekv_parameters(DEV)

    # Initialize EKV noise circuit and extract expressions
    print("=== Initializing EKV noise circuit ===")
    with suppress_output():
        prj = sl.initProject(f"BinkleyNoiseFitting")
        createEKVNoiseCir(DEV, EKVlib)
        cir = sl.makeCircuit('cir/MOS_noise.cir', imgWidth=None)

    # Extract noise-related expressions from EKV library
    suffix = '_N18' if DEV == 'nch' else '_P18'
    keep_params = [f'I_0{suffix}', f'C_OX{suffix}', f'KF{suffix}', f'AF{suffix}']

    print("=== Extracting noise expressions from EKV library ===")
    Gamma_expr = expand_expression(cir, 'Gamma_X1', keep_params=keep_params)
    IC_expr = expand_expression(cir, 'IC_X1', keep_params=keep_params)
    g_m_expr = expand_expression(cir, 'g_m_X1', keep_params=keep_params)
    c_iss_expr = expand_expression(cir, 'c_iss_X1', keep_params=keep_params)
    K_F_expr = expand_expression(cir, 'K_F_X1', keep_params=keep_params)

    print(f"Gamma symbols: {sorted([str(s) for s in Gamma_expr.free_symbols])}")
    print(f"IC symbols: {sorted([str(s) for s in IC_expr.free_symbols])}")
    print(f"K_F symbols: {sorted([str(s) for s in K_F_expr.free_symbols])}")

    I_0 = ekv_params['I_0']
    N_s = ekv_params['N_s']
    E_CRIT = ekv_params['E_CRIT']
    Theta = ekv_params['Theta']
    Vth = ekv_params['Vth']
    C_OX = ekv_params['C_OX']
    CGSO = ekv_params['CGSO']
    CGBO = ekv_params['CGBO']

    print(f"Loaded: I_0={I_0:.6e}, N_s={N_s:.6f}, C_OX={C_OX:.6e}")

    # Filter: only W >= L
    valid_WL = [(W, L) for W in W_values for L in L_values if W >= L]
    print(f"\n=== Device Selection ===")
    print(f"Total combinations: {len(W_values) * len(L_values)}")
    print(f"Valid (W>=L): {len(valid_WL)}")

    # Pack expressions for noise analysis
    noise_exprs = {
        'Gamma': Gamma_expr,
        'IC': IC_expr,
        'g_m': g_m_expr,
        'c_iss': c_iss_expr,
        'K_F': K_F_expr
    }

    # Extract noise data
    print("\n=== Extracting Noise Data ===")
    noise_data = extract_noise_data(valid_WL, LIB, DEV, Npts_ID, IC_min, IC_max, fmin, I_0)

    # Stage 1: Thermal noise (using library gamma expression)
    print("\n" + "="*60)
    print("=== Stage 1: Thermal Noise (ft/10 extraction) ===")
    print("="*60)
    thermal_results = analyze_thermal_noise(noise_data, ekv_params, noise_exprs, DEV)

    # Stage 2: Flicker noise
    print("\n" + "="*60)
    print("=== Stage 2: Flicker Noise (residual fitting) ===")
    print("="*60)
    flicker_results = fit_flicker_noise(noise_data, ekv_params, thermal_results, noise_exprs, DEV,
                                       LIB, EKVlib, Lmin, Lmax, Wmin, Wmax, IC_min, IC_max)

    # Visualization and RMS error analysis
    print("\n" + "="*60)
    print("=== Visualization and RMS Error Analysis ===")
    print("="*60)
    plot_noise_fitting(noise_data, thermal_results, flicker_results, ekv_params, DEV)
    calculate_rms_errors(noise_data, thermal_results, flicker_results, ekv_params, DEV)

    print("\n=== Noise Fitting Complete ===")
    return thermal_results, flicker_results

def load_ekv_parameters(DEV):
    """Load previously fitted EKV parameters from file"""
    filename = f'ekv_parameters_{DEV}.txt'

    ekv_params = {}

    try:
        with open(filename, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    # Remove comments after the value
                    line = line.split('#')[0].strip()
                    if '=' in line:
                        key, value = line.split('=')
                        key = key.strip()
                        value = value.strip()
                        ekv_params[key] = float(value)

        print(f"✓ Loaded parameters from {filename}")
        return ekv_params

    except FileNotFoundError:
        print(f"⚠️  File {filename} not found. Using default values.")
        # Return default values
        if DEV == 'nch':
            return {
                'I_0': 8e-7,
                'N_s': 1.35,
                'E_CRIT': 8.8e6,
                'Theta': 0.45,
                'Vth': 0.43,
                'C_OX': 8.2e-3,
                'CGSO': 300e-12,
                'CGBO': 1e-12
            }
        else:
            return {
                'I_0': 1.5e-7,
                'N_s': 1.35,
                'E_CRIT': 1e9,
                'Theta': 0.47,
                'Vth': -0.425,
                'C_OX': 8.4e-3,
                'CGSO': 350e-12,
                'CGBO': 1e-12
            }

def extract_noise_data(valid_WL, LIB, DEV, Npts_ID, IC_min, IC_max, fmin, I_0):
    """Extract noise data for W>=L devices at IC points from IC_min to IC_max (logspace)"""

    refDes = 'M1'
    M = 1
    VP = 1.8
    VS, VD, VB = 0, 0.9, 0

    if DEV == "pch":
        VS = VP
        VB = VP
        VD = VP - VD

    freq_dc = 1e5
    ID_MIN_THRESHOLD = 1e-15

    noise_data = {}
    total_extracted = 0

    for idx, (W_v, L_v) in enumerate(valid_WL):

        noise_data[(W_v, L_v)] = {
            'id': [],
            'freq': [],
            'sv_inoise': [],
            'ft': [],
            'gm': [],
            'ciss': []
        }

        # Calculate ID for IC points (logspace from IC_min to IC_max)
        IC_list = np.logspace(np.log10(IC_min), np.log10(IC_max), Npts_ID)
        ID_list = IC_list * I_0 * W_v / L_v

        # For PMOS, ID should be negative
        if DEV == "pch":
            ID_list = -ID_list

        for ID_val in ID_list:
            try:
                # Get DC parameters
                device = sl.MOS(refDes, LIB, DEV, W_v, L_v, M)
                with suppress_output():
                    device.getOPid(ID_val, VD, VS, VB, freq_dc, False)

                # Get actual current
                actual_id = device.params.get('i(ids)', ID_val)
                if isinstance(actual_id, (list, np.ndarray)):
                    actual_id = actual_id[0]
                actual_id = float(actual_id)

                # Extract gm and ciss
                gm_val = device.params['ggs']
                cgg_val = device.params['cgg']
                if isinstance(gm_val, (list, np.ndarray)):
                    gm_val = gm_val[0]
                if isinstance(cgg_val, (list, np.ndarray)):
                    cgg_val = cgg_val[0]

                ft_val = gm_val / (2 * np.pi * cgg_val)

                # Dynamic fmax based on ft
                fmax = ft_val / 5
                fmax = np.clip(fmax, 1e5, 10e9)
                numDec = 10

                # Extract noise spectrum
                device2 = sl.MOS(refDes, LIB, DEV, W_v, L_v, M)
                with suppress_output():
                    inoiseTraceDict, _, _ = device2.getSv_inoise(
                        ID_val, VD, VS, VB, fmin, fmax, numDec
                    )

                trace = inoiseTraceDict.get('inoise')
                if trace is not None:
                    freq_data = np.array(trace.xData, dtype=float)
                    sv_data = np.array(trace.yData, dtype=float)

                    noise_data[(W_v, L_v)]['id'].append(actual_id)
                    noise_data[(W_v, L_v)]['freq'].append(freq_data)
                    noise_data[(W_v, L_v)]['sv_inoise'].append(sv_data)
                    noise_data[(W_v, L_v)]['ft'].append(ft_val)
                    noise_data[(W_v, L_v)]['gm'].append(gm_val)
                    noise_data[(W_v, L_v)]['ciss'].append(cgg_val)
                    total_extracted += 1

            except Exception as e:
                continue

    return noise_data

def analyze_thermal_noise(noise_data, ekv_params, noise_exprs, DEV):
    """Analyze thermal noise at ft/10 with pole correction
    Uses library expressions for Gamma, IC, and gm
    """

    I_0 = ekv_params['I_0']
    N_s = ekv_params['N_s']
    E_CRIT = ekv_params['E_CRIT']
    Theta = ekv_params['Theta']
    C_OX = ekv_params['C_OX']
    CGSO = ekv_params['CGSO']
    CGBO = ekv_params['CGBO']

    k = 1.38064852e-23
    q = 1.60217662e-19
    T = 300
    U_T = k * T / q

    # Get library expressions
    Gamma_expr = noise_exprs['Gamma']
    IC_expr = noise_exprs['IC']
    g_m_expr = noise_exprs['g_m']

    # Create numeric functions from library expressions
    suffix = '_N18' if DEV == 'nch' else '_P18'
    param_subs = {
        sp.Symbol(f'I_0{suffix}'): I_0,
        sp.Symbol(f'C_OX{suffix}'): C_OX,
        sp.Symbol(f'N_s{suffix}'): N_s,
        sp.Symbol(f'E_CRIT{suffix}'): E_CRIT,
        sp.Symbol(f'Theta{suffix}'): Theta,
        sp.Symbol(f'CGSO{suffix}'): CGSO,
        sp.Symbol(f'CGBO{suffix}'): CGBO,
        sp.Symbol('I_D'): sp.Symbol('ID'),
        sp.Symbol('W'): sp.Symbol('W'),
        sp.Symbol('L'): sp.Symbol('L')
    }

    Gamma_func = sp.lambdify([sp.Symbol('ID'), sp.Symbol('W'), sp.Symbol('L')],
                             Gamma_expr.subs(param_subs), 'numpy')
    IC_func = sp.lambdify([sp.Symbol('ID'), sp.Symbol('W'), sp.Symbol('L')],
                          IC_expr.subs(param_subs), 'numpy')
    g_m_func = sp.lambdify([sp.Symbol('ID'), sp.Symbol('W'), sp.Symbol('L')],
                           g_m_expr.subs(param_subs), 'numpy')

    thermal_data = []

    for (W, L), data in noise_data.items():
        for i, ID_val in enumerate(data['id']):
            freq = data['freq'][i]
            sv_bsim = data['sv_inoise'][i]
            ft_val = data['ft'][i]
            gm_val = data['gm'][i]

            # Calculate using library expressions
            gamma = float(Gamma_func(ID_val, W, L))
            IC = float(IC_func(ID_val, W, L))
            gm_lib = float(g_m_func(ID_val, W, L))

            # Theoretical thermal noise using library gamma and gm
            sv_thermal_theory = 4 * k * T * N_s * gamma / gm_lib

            # Extract at ft/10
            f_target = ft_val / 10
            idx = np.argmin(np.abs(freq - f_target))
            f_actual = freq[idx]
            sv_at_ft10 = sv_bsim[idx]

            # Pole correction: assume pole at ft
            # |H(f)|^2 = 1 / (1 + (f/ft)^2)
            correction_factor = 1 + (f_actual / ft_val)**2
            sv_thermal_extracted = sv_at_ft10 * correction_factor

            thermal_data.append({
                'W': W,
                'L': L,
                'ID': ID_val,
                'IC': IC,
                'ft': ft_val,
                'gm': gm_val,
                'gamma': gamma,
                'f_extract': f_actual,
                'sv_theory': sv_thermal_theory,
                'sv_extracted': sv_thermal_extracted,
                'sv_at_ft10': sv_at_ft10,
                'correction': correction_factor
            })


    return thermal_data

def fit_flicker_noise(noise_data, ekv_params, thermal_results, noise_exprs, DEV,
                     LIB, EKVlib, Lmin, Lmax, Wmin, Wmax, IC_min, IC_max):
    """Fit flicker noise from residual using library K_F expression
    Model from library: Sv_flicker = K_F / (C_OX^2 * W * L * f^AF)
    where K_F = KF_N18*(1+2*N_s*U_T*sqrt(IC)/V_KF)^2

    Parameters:
    -----------
    noise_data : dict - Noise data from BSIM
    ekv_params : dict - Previously fitted EKV parameters
    thermal_results : list - Thermal noise analysis results
    noise_exprs : dict - Noise-related expressions
    DEV : str - Device type
    LIB : str - BSIM library path
    EKVlib : str - EKV library filename
    Lmin, Lmax : float - Channel length range
    Wmin, Wmax : float - Channel width range
    IC_min, IC_max : float - IC range
    """

    I_0 = ekv_params['I_0']
    N_s = ekv_params['N_s']
    E_CRIT = ekv_params['E_CRIT']
    Theta = ekv_params['Theta']
    C_OX = ekv_params['C_OX']
    CGSO = ekv_params['CGSO']
    CGBO = ekv_params['CGBO']

    k = 1.38064852e-23
    q = 1.60217662e-19
    T = 300
    U_T = k * T / q

    # Get library expressions
    Gamma_expr = noise_exprs['Gamma']
    IC_expr = noise_exprs['IC']
    g_m_expr = noise_exprs['g_m']

    # Create numeric functions
    suffix = '_N18' if DEV == 'nch' else '_P18'
    param_subs = {
        sp.Symbol(f'I_0{suffix}'): I_0,
        sp.Symbol(f'C_OX{suffix}'): C_OX,
        sp.Symbol(f'N_s{suffix}'): N_s,
        sp.Symbol(f'E_CRIT{suffix}'): E_CRIT,
        sp.Symbol(f'Theta{suffix}'): Theta,
        sp.Symbol(f'CGSO{suffix}'): CGSO,
        sp.Symbol(f'CGBO{suffix}'): CGBO,
        sp.Symbol('I_D'): sp.Symbol('ID'),
        sp.Symbol('W'): sp.Symbol('W'),
        sp.Symbol('L'): sp.Symbol('L')
    }

    Gamma_func = sp.lambdify([sp.Symbol('ID'), sp.Symbol('W'), sp.Symbol('L')],
                             Gamma_expr.subs(param_subs), 'numpy')
    IC_func = sp.lambdify([sp.Symbol('ID'), sp.Symbol('W'), sp.Symbol('L')],
                          IC_expr.subs(param_subs), 'numpy')
    g_m_func = sp.lambdify([sp.Symbol('ID'), sp.Symbol('W'), sp.Symbol('L')],
                           g_m_expr.subs(param_subs), 'numpy')

    # Calculate residuals and IC values
    residual_data = []

    for (W, L), data in noise_data.items():
        for i, ID_val in enumerate(data['id']):
            freq = data['freq'][i]
            sv_bsim = data['sv_inoise'][i]
            ft_val = data['ft'][i]

            # Calculate using library expressions
            gamma = float(Gamma_func(ID_val, W, L))
            IC = float(IC_func(ID_val, W, L))
            gm_lib = float(g_m_func(ID_val, W, L))

            # Thermal noise from library
            sv_thermal = 4 * k * T * N_s * gamma / gm_lib

            # Residual
            sv_residual = sv_bsim - sv_thermal

            residual_data.append({
                'W': W,
                'L': L,
                'ID': ID_val,
                'IC': IC,
                'ft': ft_val,
                'freq': freq,
                'sv_residual': sv_residual
            })

    # Fit KF_base, AF, and V_KF using low frequency data (f < ft/100)
    # Model from library: Sv_flicker = K_F / (C_OX^2 * W * L * f^AF)
    # where K_F = KF_base*(1+2*N_s*U_T*sqrt(IC)/V_KF)^2
    from scipy.optimize import differential_evolution

    def objective(params):
        KF_base, AF, V_KF = params
        errors = []

        for rd in residual_data:
            freq = rd['freq']
            sv_res = rd['sv_residual']
            ft_val = rd['ft']
            W = rd['W']
            L = rd['L']
            IC = rd['IC']

            # Calculate K_F using library formula
            K_F = KF_base * (1 + 2*N_s*U_T*np.sqrt(IC)/V_KF)**2

            # Use low frequency points
            low_freq_mask = freq < (ft_val / 100)
            if np.sum(low_freq_mask) > 3:
                freq_lf = freq[low_freq_mask]
                sv_res_lf = sv_res[low_freq_mask]

                # Predicted flicker using library model: K_F / (C_OX^2 * W * L * f^AF)
                sv_flicker_pred = K_F / (C_OX**2 * W * L * freq_lf**AF)

                # Only use positive residuals
                valid_mask = sv_res_lf > 0
                if np.sum(valid_mask) > 0:
                    sv_res_lf = sv_res_lf[valid_mask]
                    sv_flk_lf = K_F / (C_OX**2 * W * L * freq_lf[valid_mask]**AF)

                    # Log error
                    log_err = (np.log10(sv_flk_lf) - np.log10(sv_res_lf))**2
                    errors.extend(log_err)

        return np.sqrt(np.mean(errors)) if errors else 1e10

    print("\nOptimizing KF, AF, and V_KF...")
    # Bounds: [KF_base, AF, V_KF]
    # V_KF: wider range from 0.1 to 10 V (default was 2.0 for nch, 0.2 for pch)
    bounds = [(1e-27, 1e-22), (0.1, 2.0), (0.1, 10.0)]

    result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=200,
        popsize=20,
        seed=42,
        disp=True,
        workers=1,
        atol=1e-6,
        tol=1e-6,
        polish=True
    )

    KF_best = result.x[0]
    AF_best = result.x[1]
    V_KF_best = result.x[2]
    best_error = result.fun

    print(f"\n{'='*60}")
    print("FLICKER NOISE PARAMETERS")
    print(f"{'='*60}")
    print(f"KF_base (KF_{DEV[0].upper()}18): {KF_best:.6e}")
    print(f"AF (AF_{DEV[0].upper()}18): {AF_best:.6f}")
    print(f"V_KF (V_KF_{DEV[0].upper()}18): {V_KF_best:.6f}")
    print(f"Log-RMSE: {best_error:.6f}")
    print(f"{'='*60}")

    # Save results with metadata
    suffix_upper = DEV[0].upper() + '18'
    with open(f'noise_parameters_{DEV}.txt', 'w') as f:
        f.write("# EKV Noise Parameters (matching library format)\n")
        f.write(f"# Fitting metadata\n")
        f.write(f"# LIB = {LIB}\n")
        f.write(f"# EKVlib = {EKVlib}\n")
        f.write(f"# Device = {DEV}\n")
        f.write(f"# Date = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Fitting range: L=[{Lmin:.2e}, {Lmax:.2e}], W=[{Wmin:.2e}, {Wmax:.2e}]\n")
        f.write(f"# IC range: [{IC_min}, {IC_max}]\n")
        f.write("\n# Thermal Noise\n")
        f.write("# Gamma = (1/2 + 2/3*IC) / (1+IC)\n")
        f.write("# Sv_thermal = 4*k*T*N_s*Gamma/gm\n")
        f.write(f"# N_s_{suffix_upper} already fitted: {N_s:.6f}\n")
        f.write("\n# Flicker Noise (Library model)\n")
        f.write(f"KF_{suffix_upper} = {KF_best:.6e}\n")
        f.write(f"AF_{suffix_upper} = {AF_best:.6f}\n")
        f.write(f"V_KF_{suffix_upper} = {V_KF_best:.6f}\n")
        f.write(f"# K_F = KF_{suffix_upper}*(1+2*N_s*U_T*sqrt(IC)/V_KF_{suffix_upper})^2\n")
        f.write(f"# Sv_flicker = K_F / (C_OX^2 * W * L * f^AF)\n")

    print(f"\n✓ Parameters saved to noise_parameters_{DEV}.txt")

    return {
        'KF_base': KF_best,
        'AF': AF_best,
        'V_KF': V_KF_best,
        'log_rmse': best_error,
        'residual_data': residual_data
    }

def plot_noise_fitting(noise_data, thermal_results, flicker_results, ekv_params, DEV):
    """Plot thermal and flicker noise fitting comparison"""
    import matplotlib.pyplot as plt

    KF_base = flicker_results['KF_base']
    AF = flicker_results['AF']
    V_KF = flicker_results['V_KF']
    I_0 = ekv_params['I_0']
    N_s = ekv_params['N_s']
    E_CRIT = ekv_params['E_CRIT']
    Theta = ekv_params['Theta']
    C_OX = ekv_params['C_OX']
    CGSO = ekv_params['CGSO']
    CGBO = ekv_params['CGBO']

    k = 1.38064852e-23
    q = 1.60217662e-19
    T = 300
    U_T = k * T / q

    # Select a few representative bias points
    thermal_dict = {}
    for td in thermal_results:
        key = (td['W'], td['L'], td['ID'])
        thermal_dict[key] = td

    # Pick 3 representative (W,L,IC) combinations
    unique_WL = list(set((td['W'], td['L']) for td in thermal_results))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    plot_idx = 0
    max_plots = 6

    for (W, L) in unique_WL[:max_plots//2]:
        data = noise_data.get((W, L))
        if data is None or len(data['id']) == 0:
            continue

        # Plot 2 ICs per (W,L)
        for i in [0, -1]:  # First and last IC
            if i >= len(data['id']):
                continue

            ID_val = data['id'][i]
            freq = data['freq'][i]
            sv_bsim = data['sv_inoise'][i]
            ft_val = data['ft'][i]
            gm_val = data['gm'][i]

            # Calculate IC (use abs for PMOS)
            IC_i = abs(ID_val) * L / W / I_0
            IC_CRIT = 1/(4*(N_s*U_T)*(Theta+1/L/E_CRIT))**2
            IC = IC_i * (1 + 3*IC_i/IC_CRIT)**(1/3)

            # Thermal noise
            gamma = (0.5 + 2/3*IC) / (1 + IC)
            sv_thermal = 4 * k * T * N_s * gamma / gm_val

            # Flicker noise using library model with IC-dependent K_F
            K_F = KF_base * (1 + 2*N_s*U_T*np.sqrt(IC)/V_KF)**2
            sv_flicker = K_F / (C_OX**2 * W * L * freq**AF)

            # Total EKV
            sv_ekv_total = sv_thermal + sv_flicker

            ax = axes[plot_idx]
            ax.loglog(freq, sv_bsim, 'b-', linewidth=2, label='BSIM (sim)')
            ax.loglog(freq, sv_ekv_total, 'r--', linewidth=2, label='EKV (thermal+flicker)')
            ax.loglog(freq, sv_thermal * np.ones_like(freq), 'g:', linewidth=1.5, label='EKV thermal')
            ax.loglog(freq, sv_flicker, 'm:', linewidth=1.5, label='EKV flicker')
            ax.axvline(ft_val/10, color='orange', linestyle='--', alpha=0.5, label=f'ft/10={ft_val/10:.2e}Hz')

            ax.set_xlabel('Frequency (Hz)', fontsize=10)
            ax.set_ylabel('Sv (A²/Hz)', fontsize=10)
            ax.set_title(f'W={W*1e6:.1f}μm, L={L*1e6:.2f}μm, IC={IC:.2f}', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, which='both', alpha=0.3)

            plot_idx += 1
            if plot_idx >= max_plots:
                break

        if plot_idx >= max_plots:
            break

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f'noise_fitting_{DEV}.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved noise fitting plots to noise_fitting_{DEV}.png")
    plt.close()

def calculate_rms_errors(noise_data, thermal_results, flicker_results, ekv_params, DEV):
    """Calculate RMS noise integrated from 1Hz to ft/10"""

    KF_base = flicker_results['KF_base']
    AF = flicker_results['AF']
    V_KF = flicker_results['V_KF']
    I_0 = ekv_params['I_0']
    N_s = ekv_params['N_s']
    E_CRIT = ekv_params['E_CRIT']
    Theta = ekv_params['Theta']
    C_OX = ekv_params['C_OX']
    CGSO = ekv_params['CGSO']
    CGBO = ekv_params['CGBO']

    k = 1.38064852e-23
    q = 1.60217662e-19
    T = 300
    U_T = k * T / q

    print("\n" + "="*60)
    print("RMS NOISE ERROR ANALYSIS (1Hz to ft/10)")
    print("="*60)

    rms_errors = []

    for (W, L), data in noise_data.items():
        for i, ID_val in enumerate(data['id']):
            freq = data['freq'][i]
            sv_bsim = data['sv_inoise'][i]
            ft_val = data['ft'][i]
            gm_val = data['gm'][i]

            # Calculate IC (use abs for PMOS)
            IC_i = abs(ID_val) * L / W / I_0
            IC_CRIT = 1/(4*(N_s*U_T)*(Theta+1/L/E_CRIT))**2
            IC = IC_i * (1 + 3*IC_i/IC_CRIT)**(1/3)

            # Thermal noise
            gamma = (0.5 + 2/3*IC) / (1 + IC)
            sv_thermal = 4 * k * T * N_s * gamma / gm_val

            # Flicker noise using library model with IC-dependent K_F
            K_F = KF_base * (1 + 2*N_s*U_T*np.sqrt(IC)/V_KF)**2
            sv_flicker = K_F / (C_OX**2 * W * L * freq**AF)

            # Total EKV
            sv_ekv_total = sv_thermal + sv_flicker

            # Integration range: 1Hz to ft/10
            f_low = 1.0
            f_high = ft_val / 10

            # Find integration range in data
            mask = (freq >= f_low) & (freq <= f_high)
            if np.sum(mask) < 2:
                continue

            freq_int = freq[mask]
            sv_bsim_int = sv_bsim[mask]
            sv_ekv_int = sv_ekv_total[mask]

            # Numerical integration (trapezoidal)
            vrms_bsim = np.sqrt(np.trapezoid(sv_bsim_int, freq_int))
            vrms_ekv = np.sqrt(np.trapezoid(sv_ekv_int, freq_int))

            # Relative error
            rel_error = abs(vrms_ekv - vrms_bsim) / vrms_bsim * 100

            rms_errors.append({
                'W': W,
                'L': L,
                'IC': IC,
                'vrms_bsim': vrms_bsim,
                'vrms_ekv': vrms_ekv,
                'rel_error': rel_error,
                'f_low': f_low,
                'f_high': f_high
            })

    # Statistics
    errors_pct = [r['rel_error'] for r in rms_errors]

    print(f"Total bias points analyzed: {len(rms_errors)}")
    print(f"\nRMS Noise Error Statistics:")
    print(f"  Mean error:   {np.mean(errors_pct):.2f}%")
    print(f"  Median error: {np.median(errors_pct):.2f}%")
    print(f"  Std error:    {np.std(errors_pct):.2f}%")
    print(f"  Min error:    {np.min(errors_pct):.2f}%")
    print(f"  Max error:    {np.max(errors_pct):.2f}%")
    print("="*60)

    # Save detailed results
    with open(f'rms_errors_{DEV}.txt', 'w') as f:
        f.write("# RMS Noise Error Analysis (1Hz to ft/10)\n")
        f.write(f"# Mean error: {np.mean(errors_pct):.2f}%\n")
        f.write(f"# Median error: {np.median(errors_pct):.2f}%\n")
        f.write("#\n")
        f.write("# W(μm)\tL(μm)\tIC\tVrms_BSIM(V)\tVrms_EKV(V)\tError(%)\tf_low(Hz)\tf_high(Hz)\n")

        for r in rms_errors:
            f.write(f"{r['W']*1e6:.2f}\t{r['L']*1e6:.3f}\t{r['IC']:.3f}\t")
            f.write(f"{r['vrms_bsim']:.6e}\t{r['vrms_ekv']:.6e}\t")
            f.write(f"{r['rel_error']:.2f}\t{r['f_low']:.1f}\t{r['f_high']:.2e}\n")

    print(f"✓ Detailed RMS errors saved to rms_errors_{DEV}.txt")

    return rms_errors

def generate_slicap_lib(devices=['nch', 'pch'], template_lib='lib/SLiCAP_C18.lib', output_file=None):
    """Generate SLiCAP-compatible library file by updating parameters in template

    This function can handle both NMOS and PMOS simultaneously by reading their
    respective fitted parameter files and updating the template library.

    Parameters:
    -----------
    devices : list - List of device types to include (default: ['nch', 'pch'])
                     Can be ['nch'], ['pch'], or ['nch', 'pch']
    template_lib : str - Template library file path
    output_file : str - Output filename (default: 'SLiCAP_fitted.lib')

    Outputs:
    --------
    A .lib file with updated parameters for all specified devices,
    keeping all subcircuits and formulas intact
    """
    if output_file is None:
        output_file = 'SLiCAP_fitted.lib'

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
            noise_file = f'noise_parameters_{DEV}.txt'
            noise_params = {}
            try:
                with open(noise_file, 'r') as f:
                    for line in f:
                        if '=' in line and not line.startswith('#'):
                            line = line.split('#')[0].strip()
                            if '=' in line:
                                key, value = line.split('=')
                                key = key.strip()
                                value = value.strip()
                                noise_params[key] = float(value)
            except FileNotFoundError:
                print(f"Warning: {noise_file} not found. Noise parameters for {DEV} will use original library values.")
                noise_params = {}

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

        except FileNotFoundError:
            print(f"Warning: Parameters for {DEV} not found. Skipping this device.")
            continue

    # Read template file and update parameters
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
                    # Preserve original formatting style
                    spacing = ' ' * (20 - len(param_part))  # Adjust spacing
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
    print(f"SLiCAP Library File Generated: {output_file}")
    print(f"{'='*60}")
    print(f"Fitted devices: {devices_str}")
    print(f"\nUpdated parameters:")
    for DEV in devices:
        if DEV in fitting_info:
            print(f"\n{fitting_info[DEV]}:")
            dev_params = {k: v for k, v in param_updates.items()
                         if ('_N18' in k and DEV == 'nch') or ('_P18' in k and DEV == 'pch')}
            for param, value in dev_params.items():
                print(f"  {param:15} = {value}")
    print(f"\nOriginal subcircuits and formulas preserved.")
    print(f"You can now use this file in SLiCAP scripts.")
    print(f"Example: .lib {output_file}")
    print(f"{'='*60}\n")

    return output_file
# LIB='.lib lib/cr018gpii_v1d0.l TT'
# LIB='.lib lib/log018.l TT' 
def run_complete_fitting(DEV='nch', LIB='.lib lib/log018.l TT', EKVlib='SLiCAP_C18.lib',
                         Lmin=0.18e-6, Lmax=10e-6, Wmin=0.22e-6, Wmax=50e-6,
                         gridnumL=10, gridnumW=10, Npts=50, Npts_ID=4, fmin=0.01,
                         IC_min_gm=0.01, IC_max_gm=100, IC_min_noise=0.01, IC_max_noise=10):
    """
    Complete fitting workflow for a single device (NMOS or PMOS)

    This is a convenience function that runs both DC and noise fitting in sequence.

    Parameters:
    -----------
    DEV : str - Device type ('nch' or 'pch')
    LIB : str - BSIM library path
    EKVlib : str - EKV library template path
    Lmin, Lmax : float - Channel length range
    Wmin, Wmax : float - Channel width range
    gridnumL, gridnumW : int - Grid points for L and W
    Npts : int - Number of bias points for DC fitting
    Npts_ID : int - Number of bias points for noise fitting
    fmin : float - Minimum frequency for noise fitting
    IC_min_gm, IC_max_gm : float - IC range for DC fitting
    IC_min_noise, IC_max_noise : float - IC range for noise fitting

    Returns:
    --------
    dict - Dictionary with fitting results
    """

    print("\n" + "="*60)
    print(f"COMPLETE FITTING WORKFLOW FOR {DEV.upper()}")
    print("="*60)

    # Step 1: DC parameter fitting (gm, ft, Ciss)
    print("\n" + "="*60)
    print("STEP 1/2: DC PARAMETER FITTING (gm, ft, Ciss)")
    print("="*60)

    gmfitting(
        DEV=DEV,
        LIB=LIB,
        EKVlib=EKVlib,
        Lmin=Lmin, Lmax=Lmax,
        Wmin=Wmin, Wmax=Wmax,
        gridnumL=gridnumL, gridnumW=gridnumW,
        Npts=Npts,
        IC_min=IC_min_gm, IC_max=IC_max_gm
    )

    # Step 2: Noise parameter fitting (thermal + flicker)
    print("\n" + "="*60)
    print("STEP 2/2: NOISE PARAMETER FITTING (thermal + flicker)")
    print("="*60)

    noise_results = noisefitting(
        DEV=DEV,
        LIB=LIB,
        EKVlib=EKVlib,
        Lmin=Lmin, Lmax=Lmax,
        Wmin=Wmin, Wmax=Wmax,
        gridnumL=gridnumL, gridnumW=gridnumW,
        Npts_ID=Npts_ID,
        fmin=fmin,
        IC_min=IC_min_noise, IC_max=IC_max_noise
    )

    print("\n" + "="*60)
    print(f"✓ COMPLETE FITTING FINISHED FOR {DEV.upper()}")
    print("="*60)
    print(f"Results saved to:")
    print(f"  - ekv_parameters_{DEV}.txt")
    print(f"  - noise_parameters_{DEV}.txt")
    print(f"\nTo generate library file, use:")
    print(f"  python generate_lib.py")
    print("="*60 + "\n")

    return {
        'device': DEV,
        'noise_results': noise_results
    }

if __name__ == "__main__":
    # Example: Complete fitting workflow
    DEV = 'pch'  # Change to 'pch' for PMOS

    # Option 1: Use the convenience function (recommended)
    run_complete_fitting(DEV=DEV)

    # Option 2: Manual step-by-step (for more control)
    # Step 1: DC parameter fitting
    # print("\n" + "="*60)
    # print("STEP 1: DC PARAMETER FITTING (gmfitting)")
    # print("="*60)
    # gmfitting(DEV=DEV)

    # Step 2: Noise parameter fitting
    # print("\n" + "="*60)
    # print("STEP 2: NOISE PARAMETER FITTING (noisefitting)")
    # print("="*60)
    # noisefitting(DEV=DEV)

    # Step 3: Generate library using separate script
    # print("\n" + "="*60)
    # print("STEP 3: GENERATE SLICAP LIBRARY")
    # print("="*60)
    # Use: python generate_lib.py