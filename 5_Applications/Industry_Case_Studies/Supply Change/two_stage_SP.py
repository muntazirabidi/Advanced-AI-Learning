import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import os
import sys
import matplotlib.pyplot as plt
import time

# Function to find the GLPK executable
def find_glpk_executable():
    """
    Find the GLPK executable path on the system.
    Returns the path if found, None otherwise.
    """
    # Potential paths for macOS
    potential_paths = [
        os.path.join(os.environ.get('CONDA_PREFIX', ''), 'bin', 'glpsol'),  # Conda path
        "/opt/homebrew/bin/glpsol",  # Homebrew on Apple Silicon
        "/usr/local/bin/glpsol",     # Homebrew on Intel Macs
        # Add Windows paths if needed
        "C:\\glpk\\w64\\glpsol.exe",
        "C:\\Program Files\\glpk\\glpsol.exe",
        "C:\\Program Files (x86)\\glpk\\glpsol.exe"
    ]
    
    for path in potential_paths:
        if os.path.exists(path):
            print(f"Found GLPK executable at: {path}")
            return path
    
    print("GLPK executable not found in expected locations.")
    return None

def solve_two_stage_stochastic_program(num_scenarios=5, seed=42):
    """
    Solve a two-stage stochastic programming problem for a simple resource allocation example.
    
    This example models a production planning problem:
    - First stage: Decide how much to produce before knowing demand
    - Second stage: After observing demand, decide how much to sell or store
    
    Parameters:
    -----------
    num_scenarios : int
        Number of demand scenarios to generate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing the optimal production quantity, expected cost,
        and detailed information for each scenario
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Problem parameters
    production_cost = 10
    storage_cost = 2
    shortage_cost = 20
    selling_price = 15
    
    # Generate random demand scenarios
    base_demand = 100
    demand_scenarios = base_demand + np.random.normal(0, 20, num_scenarios)
    scenario_probs = np.ones(num_scenarios) / num_scenarios  # Equal probability
    
    # Create the Pyomo concrete model
    model = pyo.ConcreteModel()
    
    # First stage decision variable: how much to produce
    model.produce = pyo.Var(domain=pyo.NonNegativeReals)
    
    # Second stage variables - for each scenario
    model.scenarios = pyo.RangeSet(0, num_scenarios-1)
    model.sell = pyo.Var(model.scenarios, domain=pyo.NonNegativeReals)
    model.store = pyo.Var(model.scenarios, domain=pyo.NonNegativeReals)
    model.shortage = pyo.Var(model.scenarios, domain=pyo.NonNegativeReals)
    
    # First stage objective: minimize production cost
    model.first_stage_cost = pyo.Expression(expr=production_cost * model.produce)
    
    # Second stage objective: expected profit/cost across all scenarios
    def second_stage_rule(model, s):
        return (scenario_probs[s] * 
                (selling_price * model.sell[s] - 
                 storage_cost * model.store[s] - 
                 shortage_cost * model.shortage[s]))
    
    model.second_stage_obj = pyo.Expression(model.scenarios, rule=second_stage_rule)
    
    # Total objective: first stage cost + expected second stage cost
    model.obj = pyo.Objective(
        expr=model.first_stage_cost - sum(model.second_stage_obj[s] for s in model.scenarios),
        sense=pyo.minimize
    )
    
    # Constraints
    
    # For each scenario, amount sold + stored equals amount produced
    def balance_rule(model, s):
        return model.sell[s] + model.store[s] == model.produce
    
    model.balance = pyo.Constraint(model.scenarios, rule=balance_rule)
    
    # For each scenario, amount sold + shortage equals demand
    def demand_rule(model, s):
        return model.sell[s] + model.shortage[s] == demand_scenarios[s]
    
    model.demand = pyo.Constraint(model.scenarios, rule=demand_rule)
    
    # Set up the solver
    solver = None
    
    # Try to find GLPK executable
    glpk_path = find_glpk_executable()
    
    if glpk_path:
        # Create solver with explicit executable path
        try:
            solver = SolverFactory('glpk', executable=glpk_path)
            print("Using GLPK solver with explicit path")
        except:
            solver = None
    
    # If GLPK isn't available, try other solvers
    if solver is None:
        for solver_name in ['cbc', 'ipopt', 'glpk_direct', 'scip']:
            try:
                solver = SolverFactory(solver_name)
                if solver.available():
                    print(f"Using {solver_name} solver")
                    break
                else:
                    solver = None
            except:
                solver = None
                continue
    
    # If no solver is available, solve with a simple method instead
    if solver is None:
        print("WARNING: No solver available. Using a simple approximation method.")
        # Simplified solution: produce the average demand to minimize expected costs
        optimal_production = np.mean(demand_scenarios)
        expected_cost = production_cost * optimal_production
        
        results = {
            'optimal_production': optimal_production,
            'expected_cost': expected_cost,
            'scenarios': [],
            'demand_scenarios': demand_scenarios
        }
        
        for s in range(num_scenarios):
            sold = min(optimal_production, demand_scenarios[s])
            stored = max(0, optimal_production - demand_scenarios[s])
            shortage = max(0, demand_scenarios[s] - optimal_production)
            
            scenario_result = {
                'scenario': s,
                'demand': demand_scenarios[s],
                'sold': sold,
                'stored': stored,
                'shortage': shortage
            }
            results['scenarios'].append(scenario_result)
            
        return results
    
    # Solve the model with the available solver
    result = solver.solve(model)
    
    # Check if the solver found a solution
    if (result.solver.status == pyo.SolverStatus.ok and 
        result.solver.termination_condition == pyo.TerminationCondition.optimal):
        # Extract and return results
        results = {
            'optimal_production': pyo.value(model.produce),
            'expected_cost': pyo.value(model.obj),
            'scenarios': [],
            'demand_scenarios': demand_scenarios
        }
        
        for s in model.scenarios:
            scenario_result = {
                'scenario': s,
                'demand': demand_scenarios[s],
                'sold': pyo.value(model.sell[s]),
                'stored': pyo.value(model.store[s]),
                'shortage': pyo.value(model.shortage[s])
            }
            results['scenarios'].append(scenario_result)
            
        return results
    else:
        print(f"Solver status: {result.solver.status}")
        print(f"Termination condition: {result.solver.termination_condition}")
        raise Exception("The solver failed to find a solution.")

def visualize_scenario_analysis(scenario_counts=[10, 50, 100, 500, 1000], num_runs=5):
    """
    Run the stochastic program with different numbers of scenarios and visualize the results.
    
    Parameters:
    -----------
    scenario_counts : list
        List of scenario counts to test
    num_runs : int
        Number of runs for each scenario count to assess stability
    """
    # Store results for each scenario count and run
    all_results = {count: [] for count in scenario_counts}
    computation_times = []
    
    # Run the model for each scenario count multiple times
    for count in scenario_counts:
        print(f"\nRunning with {count} scenarios...")
        times = []
        
        for run in range(num_runs):
            start_time = time.time()
            try:
                # Use a different seed for each run
                seed = 42 + run
                results = solve_two_stage_stochastic_program(num_scenarios=count, seed=seed)
                all_results[count].append(results)
                
                end_time = time.time()
                run_time = end_time - start_time
                times.append(run_time)
                
                print(f"Run {run+1}: Optimal production = {results['optimal_production']:.2f}, "
                      f"Expected cost = {results['expected_cost']:.2f}, "
                      f"Time = {run_time:.2f} seconds")
            except Exception as e:
                print(f"Error in run {run+1} with {count} scenarios: {str(e)}")
        
        avg_time = np.mean(times) if times else 0
        computation_times.append(avg_time)
    
    # Create visualizations
    
    # 1. Plot optimal production quantities by scenario count
    plt.figure(figsize=(12, 8))
    
    # Box plot for production quantities
    production_data = []
    for count in scenario_counts:
        prod_values = [res['optimal_production'] for res in all_results[count] if 'optimal_production' in res]
        if prod_values:
            production_data.append(prod_values)
        else:
            production_data.append([])
    
    plt.subplot(2, 2, 1)
    plt.boxplot(production_data, labels=scenario_counts)
    plt.title('Optimal Production Quantity by Scenario Count')
    plt.xlabel('Number of Scenarios')
    plt.ylabel('Production Quantity')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Plot expected costs by scenario count
    cost_data = []
    for count in scenario_counts:
        cost_values = [res['expected_cost'] for res in all_results[count] if 'expected_cost' in res]
        if cost_values:
            cost_data.append(cost_values)
        else:
            cost_data.append([])
    
    plt.subplot(2, 2, 2)
    plt.boxplot(cost_data, labels=scenario_counts)
    plt.title('Expected Cost by Scenario Count')
    plt.xlabel('Number of Scenarios')
    plt.ylabel('Expected Cost')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Plot computational time by scenario count
    plt.subplot(2, 2, 3)
    plt.plot(scenario_counts, computation_times, 'o-', linewidth=2)
    plt.title('Computation Time by Scenario Count')
    plt.xlabel('Number of Scenarios')
    plt.ylabel('Average Computation Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 4. Plot demand distribution for the last run
    plt.subplot(2, 2, 4)
    for count in scenario_counts:
        if all_results[count] and 'demand_scenarios' in all_results[count][-1]:
            plt.hist(all_results[count][-1]['demand_scenarios'], bins=20, 
                     alpha=0.3, label=f'{count} scenarios')
    
    plt.title('Demand Distribution')
    plt.xlabel('Demand')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('stochastic_programming_analysis.png')
    plt.show()
    
    # Calculate and display summary statistics
    print("\nSummary Statistics:")
    print("-------------------")
    print("Scenario Count | Avg Production | Std Dev Production | Avg Expected Cost | Std Dev Cost")
    print("----------------------------------------------------------------------------")
    
    for count in scenario_counts:
        prod_values = [res['optimal_production'] for res in all_results[count] if 'optimal_production' in res]
        cost_values = [res['expected_cost'] for res in all_results[count] if 'expected_cost' in res]
        
        if prod_values and cost_values:
            avg_prod = np.mean(prod_values)
            std_prod = np.std(prod_values)
            avg_cost = np.mean(cost_values)
            std_cost = np.std(cost_values)
            
            print(f"{count:14d} | {avg_prod:14.2f} | {std_prod:18.2f} | {avg_cost:17.2f} | {std_cost:11.2f}")
    
    return all_results

# Example usage
if __name__ == "__main__":
    print("Python executable:", sys.executable)
    print("PATH environment variable:", os.environ.get('PATH'))
    
    # Run scenario analysis
    scenario_counts = [10, 50, 100, 500, 1000]
    num_runs = 3  # Number of runs per scenario count to assess stability
    
    try:
        all_results = visualize_scenario_analysis(scenario_counts, num_runs)
        
        # Additional analysis: Find the most stable solution
        stability_measure = {}
        for count in scenario_counts:
            prod_values = [res['optimal_production'] for res in all_results[count] if 'optimal_production' in res]
            if prod_values:
                # Coefficient of variation (lower means more stable)
                cv = np.std(prod_values) / np.mean(prod_values) if np.mean(prod_values) != 0 else float('inf')
                stability_measure[count] = cv
        
        # Find the most stable solution
        most_stable = min(stability_measure.items(), key=lambda x: x[1])
        print(f"\nMost stable solution: {most_stable[0]} scenarios (CV = {most_stable[1]:.4f})")
        
    except Exception as e:
        print(f"Error in scenario analysis: {str(e)}")
        
        # Run individual scenario counts if the full analysis fails
        print("\nRunning individual scenario analyses:")
        for num_scenarios in [100, 500]:
            print(f"\nSolving with {num_scenarios} scenarios:")
            try:
                results = solve_two_stage_stochastic_program(num_scenarios=num_scenarios)
                print(f"Optimal production quantity: {results['optimal_production']:.2f}")
                print(f"Expected total cost: {results['expected_cost']:.2f}")
                
                # Print details for first 3 scenarios
                for i, scenario in enumerate(results['scenarios'][:3]):
                    print(f"\nScenario {i}:")
                    print(f"  Demand: {scenario['demand']:.2f}")
                    print(f"  Sold: {scenario['sold']:.2f}")
                    print(f"  Stored: {scenario['stored']:.2f}")
                    print(f"  Shortage: {scenario['shortage']:.2f}")
            except Exception as sub_e:
                print(f"Error: {str(sub_e)}")
                
                # Check available solvers
                try:
                    print("\nChecking available solvers:")
                    for solver_name in ['glpk', 'cbc', 'ipopt', 'glpk_direct']:
                        solver = SolverFactory(solver_name)
                        print(f"Solver {solver_name} available: {solver.available()}")
                except:
                    print("Could not check solvers.")