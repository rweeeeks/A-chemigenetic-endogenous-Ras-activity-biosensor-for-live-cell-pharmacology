#!/usr/bin/env python3
# Solve for Sensor Kd from Competition Assay Data
# full equation from Hulme & Trevethick (2010) paper (using total ligand/sensor)
# Ki = IC50 / (1 + ([L_T] × (1-δ₀/2))/Kd + (δ₀/(1-δ₀)))

# Where:
# - [L_T] = TOTAL ligand (sensor) concentration
# - δ₀ = fraction of total ligand bound (depletion factor)
# - The (1-δ₀/2) term corrects the total concentration for depletion

# Known inputs:
# - RBD Ki (competitor affinity)
# - Observed IC50
# - Sensor concentration (total)
# - Target concentration

# Unknown to solve for:
# - Sensor Kd to target


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def calculate_sensor_binding(sensor_conc_total, target_conc, sensor_kd):

    a = 1
    b = -(target_conc + sensor_conc_total + sensor_kd)
    c = target_conc * sensor_conc_total
    
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return 0
    
    # Take the smaller root (physical solution)
    bound = (-b - np.sqrt(discriminant)) / (2*a)
    return max(0, min(bound, min(target_conc, sensor_conc_total)))

def calculate_depletion_factor(sensor_conc_total, target_conc, sensor_kd):
    """
    """
    bound_sensor = calculate_sensor_binding(sensor_conc_total, target_conc, sensor_kd)
    delta_0 = bound_sensor / sensor_conc_total
    free_sensor = sensor_conc_total - bound_sensor
    
    return delta_0, bound_sensor, free_sensor

def goldstein_barrett_equation(sensor_kd, rbd_ki, observed_ic50, sensor_conc_total, target_conc):

    try:
        # Calculate depletion factor for this sensor_kd
        delta_0, _, _ = calculate_depletion_factor(sensor_conc_total, target_conc, sensor_kd)
        
        # Calculate what the IC50 should be given this sensor_kd
        cheng_prusoff_term = (sensor_conc_total * (1 - delta_0/2)) / sensor_kd
        
        # Goldstein-Barrett depletion correction
        depletion_term = delta_0 / (1 - delta_0) if delta_0 < 0.999 else 1000  # Avoid division by zero
        
        # Combined correction factor
        correction_factor = 1 + cheng_prusoff_term + depletion_term
        
        # Predicted IC50 from this sensor Kd
        calculated_ic50 = rbd_ki * correction_factor
        
        # Return squared difference for minimization
        return (calculated_ic50 - observed_ic50)**2
        
    except (ZeroDivisionError, ValueError, OverflowError):
        return 1e10  # Large penalty for invalid values

def solve_for_sensor_kd(rbd_ki, observed_ic50, sensor_conc_total, target_conc, initial_guess=None):
    
    if initial_guess is None:
        # Smart initial guess: assume minimal correction factor
        initial_guess = observed_ic50 / rbd_ki
    
    # Define bounds for the search (must be positive)
    bounds = (0.001, 10000)  # 1 pM to 10 μM
    
    # Use scipy's minimize_scalar for robust optimization
    result = minimize_scalar(
        goldstein_barrett_equation,
        args=(rbd_ki, observed_ic50, sensor_conc_total, target_conc),
        bounds=bounds,
        method='bounded'
    )
    
    return result.x, result

def validate_solution(sensor_kd_solved, rbd_ki, observed_ic50, sensor_conc_total, target_conc):
    """Validate the solved sensor Kd by checking the calculation."""
    
    delta_0, bound_sensor, free_sensor = calculate_depletion_factor(
        sensor_conc_total, target_conc, sensor_kd_solved
    )
    
    # Calculate predicted IC50 using solved sensor Kd
    cheng_prusoff_term = (sensor_conc_total * (1 - delta_0/2)) / sensor_kd_solved
    depletion_term = delta_0 / (1 - delta_0) if delta_0 < 0.999 else float('inf')
    correction_factor = 1 + cheng_prusoff_term + depletion_term
    predicted_ic50 = rbd_ki * correction_factor
    
    error_percent = abs(predicted_ic50 - observed_ic50) / observed_ic50 * 100
    
    return {
        'delta_0': delta_0,
        'bound_sensor': bound_sensor,
        'free_sensor': free_sensor,
        'sensor_conc_total': sensor_conc_total,
        'cheng_prusoff_term': cheng_prusoff_term,
        'depletion_term': depletion_term,
        'correction_factor': correction_factor,
        'predicted_ic50': predicted_ic50,
        'error_percent': error_percent
    }

def plot_sensitivity_analysis(rbd_ki, observed_ic50, sensor_conc_total, target_conc, sensor_kd_solved):
    """Plot how the objective function varies with sensor Kd."""
    
    # Range around the solution
    kd_range = np.logspace(np.log10(sensor_kd_solved/10), np.log10(sensor_kd_solved*10), 200)
    objective_values = []
    ic50_predictions = []
    
    for kd in kd_range:
        obj_val = goldstein_barrett_equation(kd, rbd_ki, observed_ic50, sensor_conc_total, target_conc)
        objective_values.append(obj_val)
        
        # Also calculate predicted IC50 for plotting
        try:
            delta_0, _, _ = calculate_depletion_factor(sensor_conc_total, target_conc, kd)
            # Use TOTAL concentration
            cheng_prusoff_term = (sensor_conc_total * (1 - delta_0/2)) / kd
            depletion_term = delta_0 / (1 - delta_0) if delta_0 < 0.999 else 1000
            correction_factor = 1 + cheng_prusoff_term + depletion_term
            pred_ic50 = rbd_ki * correction_factor
            ic50_predictions.append(pred_ic50)
        except:
            ic50_predictions.append(np.nan)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Objective function
    ax1.loglog(kd_range, np.sqrt(objective_values), 'b-', linewidth=2)
    ax1.axvline(x=sensor_kd_solved, color='r', linestyle='--', label=f'Solution = {sensor_kd_solved:.2f} nM')
    ax1.set_xlabel('Sensor Kd (nM)')
    ax1.set_ylabel('|Predicted IC50 - Observed IC50| (nM)')
    ax1.set_title('Optimization Objective Function')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Predicted IC50 vs sensor Kd
    ax2.loglog(kd_range, ic50_predictions, 'g-', linewidth=2, label='Predicted IC50')
    ax2.axhline(y=observed_ic50, color='r', linestyle='--', label=f'Observed IC50 = {observed_ic50:.1f} nM')
    ax2.axvline(x=sensor_kd_solved, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Sensor Kd (nM)')
    ax2.set_ylabel('Predicted IC50 (nM)')
    ax2.set_title('IC50 Prediction vs Sensor Kd')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('sensor_kd_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("=" * 70)
    print("SOLVE FOR SENSOR Kd FROM COMPETITION ASSAY DATA")
    print("Using cheng-prusoff form from Hulme & Trevethick (2010)")
    print("=" * 70)
    
    print("\nFormula: Ki = IC50 / (1 + ([L_T] × (1-δ₀/2))/Kd + (δ₀/(1-δ₀)))")
    print("Where [L_T] = TOTAL ligand concentration")
    print("      δ₀ = fraction of ligand bound")
    
    print("\nThis script solves for the unknown sensor Kd given:")
    print("• Known RBD Ki (competitor affinity)")
    print("• Observed IC50 from competition experiment")
    print("• Experimental conditions (sensor & target concentrations)")
    
    # Input parameters
    print("\nEnter known parameters:")
    rbd_ki = float(input("RBD Ki (known competitor affinity, nM): "))
    observed_ic50 = float(input("Observed IC50 from experiment (nM): "))
    sensor_conc_total = float(input("TOTAL sensor concentration used (nM): "))
    target_conc = float(input("Target concentration used (nM): "))
    
    # Optional initial guess
    guess_input = input("Initial guess for sensor Kd (nM, press Enter for auto): ")
    initial_guess = float(guess_input) if guess_input.strip() else None
    
    print("\nSolving for sensor Kd...")
    
    # Solve for sensor Kd
    try:
        sensor_kd_solved, opt_result = solve_for_sensor_kd(
            rbd_ki, observed_ic50, sensor_conc_total, target_conc, initial_guess
        )
        
        if not opt_result.success:
            print("Warning: Optimization may not have converged properly")
        
        # Validate the solution
        validation = validate_solution(sensor_kd_solved, rbd_ki, observed_ic50, sensor_conc_total, target_conc)
        
        # Display results
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        print(f"\nSOLVED SENSOR Kd:")
        print(f"  Sensor Kd to target:           {sensor_kd_solved:.2f} nM")
        print(f"  Optimization success:          {opt_result.success}")
        print(f"  Final objective value:         {np.sqrt(opt_result.fun):.4f}")
        
        print(f"\nVALIDATION (using solved Kd):")
        print(f"  Total sensor concentration:    {validation['sensor_conc_total']:.1f} nM")
        print(f"  Bound sensor:                  {validation['bound_sensor']:.1f} nM")
        print(f"  Free sensor:                   {validation['free_sensor']:.1f} nM")
        print(f"  Depletion factor (δ₀):         {validation['delta_0']:.3f} ({validation['delta_0']*100:.1f}%)")
        
        print(f"\nCORRECTION TERMS:")
        print(f"  Cheng-Prusoff term:            {validation['cheng_prusoff_term']:.2f}")
        print(f"    = ([L_T] × (1-δ₀/2)) / Kd")
        print(f"    = ({sensor_conc_total:.1f} × {1 - validation['delta_0']/2:.3f}) / {sensor_kd_solved:.2f}")
        print(f"  Depletion term:                {validation['depletion_term']:.2f}")
        print(f"    = δ₀ / (1-δ₀)")
        print(f"    = {validation['delta_0']:.3f} / {1-validation['delta_0']:.3f}")
        print(f"  Total correction factor:       {validation['correction_factor']:.2f}")
        
        print(f"\nIC50 COMPARISON:")
        print(f"  Observed IC50:                 {observed_ic50:.1f} nM")
        print(f"  Predicted IC50 (solved Kd):    {validation['predicted_ic50']:.1f} nM")
        print(f"  Prediction error:              {validation['error_percent']:.2f}%")
        
        print(f"\nFORMULA USED:")
        print(f"  Ki = IC50 / (1 + ([L_T] × (1-δ₀/2))/Kd + (δ₀/(1-δ₀)))")
        print(f"  Reference: Hulme & Trevethick (2010)")
        print(f"  Solved iteratively for Kd given known Ki and IC50")
        
        # Generate plots
        print(f"\nGenerating sensitivity analysis plots...")
        plot_sensitivity_analysis(rbd_ki, observed_ic50, sensor_conc_total, target_conc, sensor_kd_solved)
        print(f"Plots saved as 'sensor_kd_analysis.png'")
        
    except Exception as e:
        print(f"\n Error during optimization: {e}")
        print("Check your input values and try again.")

if __name__ == "__main__":
    main()