import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt
import time
import os

def fashion_multistage_stochastic_program(num_scenarios=10, seed=42):
    """
    Multi-stage stochastic programming model for fashion industry production planning.
    Optimizes decisions across three stages: knitting, dyeing, and final product manufacturing
    under demand uncertainty.
    
    Parameters:
    -----------
    num_scenarios : int
        Number of demand scenarios to generate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing optimal decisions and results
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # ====================== PROBLEM PARAMETERS ======================
    
    # Product and material specifications
    num_fabric_types = 3
    num_products = 3
    
    # Time periods (weeks)
    planning_horizon = 10  # Total planning horizon
    knitting_lead_time = 3  # Weeks from knitting to dyeing
    dyeing_lead_time = 2    # Weeks from dyeing to product
    total_lead_time = knitting_lead_time + dyeing_lead_time
    
    # Production capacities per week
    knitting_capacity = 150
    dyeing_capacity = 130
    final_production_capacity = 120
    
    # Cost parameters
    knitting_cost = np.array([10, 12, 11])  # Cost per unit for each fabric type
    dyeing_cost = np.array([8, 9, 10])     # Additional cost for dyeing each fabric type
    production_cost = np.array([15, 16, 14])  # Additional cost for final production
    
    holding_cost_raw = 1.0     # Cost of holding raw fabric per week
    holding_cost_dyed = 1.5    # Cost of holding dyed fabric per week
    holding_cost_final = 2.0   # Cost of holding final product per week
    
    shortage_cost = np.array([40, 45, 35])  # Penalty for not meeting demand for each product
    
    # Revenue per unit sold
    sales_price = np.array([100, 120, 90])
    
    # Material requirements
    # How much raw fabric is needed for each product (fabric_type → product)
    fabric_requirements = np.array([
        [1.0, 0.8, 0.0],  # Fabric 1 requirements for products 1-3
        [0.0, 0.2, 0.7],  # Fabric 2 requirements for products 1-3
        [0.2, 0.0, 1.0]   # Fabric 3 requirements for products 1-3
    ])
    
    # ====================== SCENARIO GENERATION ======================
    
    # Base demand for each product in each time period
    base_demand = np.array([
        [30, 35, 40, 45, 50, 55, 50, 45, 40, 35],  # Product 1 demand over time
        [25, 25, 30, 35, 40, 45, 40, 35, 30, 25],  # Product 2 demand over time
        [20, 25, 30, 35, 30, 25, 20, 20, 25, 30]   # Product 3 demand over time
    ])
    
    # Generate scenarios using a mixture of trends and normal variations
    demand_scenarios = np.zeros((num_scenarios, num_products, planning_horizon))
    
    # Different scenario types to represent forecast patterns (upside, downside, volatile)
    scenario_types = np.random.choice(['base', 'upside', 'downside', 'volatile'], 
                                     size=num_scenarios, 
                                     p=[0.4, 0.2, 0.2, 0.2])
    
    for s in range(num_scenarios):
        scenario_type = scenario_types[s]
        
        # Base pattern
        if scenario_type == 'base':
            for p in range(num_products):
                # Moderate variation around base demand
                noise = np.random.normal(0, 0.1 * base_demand[p], planning_horizon)
                demand_scenarios[s, p] = base_demand[p] * (1 + noise)
                
        # Upside pattern - demand exceeds forecast
        elif scenario_type == 'upside':
            for p in range(num_products):
                # Positive trend bias
                trend = np.linspace(0, 0.25, planning_horizon)
                noise = np.random.normal(0, 0.1, planning_horizon)
                demand_scenarios[s, p] = base_demand[p] * (1 + trend + noise)
                
        # Downside pattern - demand below forecast
        elif scenario_type == 'downside':
            for p in range(num_products):
                # Negative trend bias
                trend = np.linspace(0, -0.25, planning_horizon)
                noise = np.random.normal(0, 0.08, planning_horizon)
                demand_scenarios[s, p] = base_demand[p] * (1 + trend + noise)
                
        # Volatile pattern - high variability
        elif scenario_type == 'volatile':
            for p in range(num_products):
                # Higher variability
                noise = np.random.normal(0, 0.25, planning_horizon)
                demand_scenarios[s, p] = base_demand[p] * (1 + noise)
    
    # Ensure no negative demand
    demand_scenarios = np.maximum(demand_scenarios, 0)
    
    # Equal probability for each scenario
    scenario_probs = np.ones(num_scenarios) / num_scenarios
    
    # ====================== MODEL CREATION ======================
    model = pyo.ConcreteModel()
    
    # Sets
    model.SCENARIOS = pyo.RangeSet(0, num_scenarios-1)
    model.PRODUCTS = pyo.RangeSet(0, num_products-1)
    model.FABRICS = pyo.RangeSet(0, num_fabric_types-1)
    model.TIME = pyo.RangeSet(0, planning_horizon-1)
    
    # First stage decision variables (knitting decisions)
    model.knit = pyo.Var(model.FABRICS, model.TIME, domain=pyo.NonNegativeReals)
    
    # Second stage decision variables (dyeing decisions)
    model.dye = pyo.Var(model.FABRICS, model.TIME, model.SCENARIOS, domain=pyo.NonNegativeReals)
    
    # Third stage decision variables (production decisions)
    model.produce = pyo.Var(model.PRODUCTS, model.TIME, model.SCENARIOS, domain=pyo.NonNegativeReals)
    
    # Inventory variables
    model.raw_inventory = pyo.Var(model.FABRICS, model.TIME, model.SCENARIOS, domain=pyo.NonNegativeReals)
    model.dyed_inventory = pyo.Var(model.FABRICS, model.TIME, model.SCENARIOS, domain=pyo.NonNegativeReals)
    model.final_inventory = pyo.Var(model.PRODUCTS, model.TIME, model.SCENARIOS, domain=pyo.NonNegativeReals)
    
    # Sales and shortage variables
    model.sales = pyo.Var(model.PRODUCTS, model.TIME, model.SCENARIOS, domain=pyo.NonNegativeReals)
    model.shortage = pyo.Var(model.PRODUCTS, model.TIME, model.SCENARIOS, domain=pyo.NonNegativeReals)
    
    # ====================== CONSTRAINTS ======================
    
    # Knitting capacity constraint
    def knitting_capacity_rule(model, t):
        return sum(model.knit[f, t] for f in model.FABRICS) <= knitting_capacity
    model.KnittingCapacity = pyo.Constraint(model.TIME, rule=knitting_capacity_rule)
    
    # Dyeing capacity constraint
    def dyeing_capacity_rule(model, t, s):
        if t >= knitting_lead_time:  # Can only dye after knitting lead time
            return sum(model.dye[f, t, s] for f in model.FABRICS) <= dyeing_capacity
        else:
            return sum(model.dye[f, t, s] for f in model.FABRICS) <= 0  # No dyeing in early periods
    model.DyeingCapacity = pyo.Constraint(model.TIME, model.SCENARIOS, rule=dyeing_capacity_rule)
    
    # Production capacity constraint
    def production_capacity_rule(model, t, s):
        if t >= total_lead_time:  # Can only produce after total lead time
            return sum(model.produce[p, t, s] for p in model.PRODUCTS) <= final_production_capacity
        else:
            return sum(model.produce[p, t, s] for p in model.PRODUCTS) <= 0  # No production in early periods
    model.ProductionCapacity = pyo.Constraint(model.TIME, model.SCENARIOS, rule=production_capacity_rule)
    
    # Raw fabric inventory balance
    def raw_inventory_balance_rule(model, f, t, s):
        if t == 0:
            # Initial raw inventory = initial knitting (assume starting with no inventory)
            return model.raw_inventory[f, t, s] == model.knit[f, t] - model.dye[f, t, s]
        else:
            # Raw inventory = previous inventory + new knitting - dyeing
            return model.raw_inventory[f, t, s] == model.raw_inventory[f, t-1, s] + model.knit[f, t] - model.dye[f, t, s]
    model.RawInventoryBalance = pyo.Constraint(model.FABRICS, model.TIME, model.SCENARIOS, rule=raw_inventory_balance_rule)
    
    # Dyed fabric inventory balance
    def dyed_inventory_balance_rule(model, f, t, s):
        if t == 0:
            # Initial dyed inventory (assume starting with no inventory)
            return model.dyed_inventory[f, t, s] == model.dye[f, t, s] - sum(fabric_requirements[f, p] * model.produce[p, t, s] for p in model.PRODUCTS)
        else:
            # Dyed inventory = previous inventory + new dyeing - used in production
            return model.dyed_inventory[f, t, s] == model.dyed_inventory[f, t-1, s] + model.dye[f, t, s] - sum(fabric_requirements[f, p] * model.produce[p, t, s] for p in model.PRODUCTS)
    model.DyedInventoryBalance = pyo.Constraint(model.FABRICS, model.TIME, model.SCENARIOS, rule=dyed_inventory_balance_rule)
    
    # Final product inventory balance
    def final_inventory_balance_rule(model, p, t, s):
        if t == 0:
            # Initial final inventory (assume starting with no inventory)
            return model.final_inventory[p, t, s] == model.produce[p, t, s] - model.sales[p, t, s]
        else:
            # Final inventory = previous inventory + new production - sales
            return model.final_inventory[p, t, s] == model.final_inventory[p, t-1, s] + model.produce[p, t, s] - model.sales[p, t, s]
    model.FinalInventoryBalance = pyo.Constraint(model.PRODUCTS, model.TIME, model.SCENARIOS, rule=final_inventory_balance_rule)
    
    # Demand satisfaction
    def demand_satisfaction_rule(model, p, t, s):
        # Sales + shortage must equal demand
        return model.sales[p, t, s] + model.shortage[p, t, s] == demand_scenarios[s, p, t]
    model.DemandSatisfaction = pyo.Constraint(model.PRODUCTS, model.TIME, model.SCENARIOS, rule=demand_satisfaction_rule)
    
    # Sales cannot exceed inventory
    def sales_limit_rule(model, p, t, s):
        return model.sales[p, t, s] <= model.final_inventory[p, t, s] + model.produce[p, t, s]
    model.SalesLimit = pyo.Constraint(model.PRODUCTS, model.TIME, model.SCENARIOS, rule=sales_limit_rule)
    
    # ====================== OBJECTIVE FUNCTION ======================
    
    def objective_rule(model):
        # Expected profit across all scenarios
        total_profit = 0
        
        for s in model.SCENARIOS:
            # Revenue from sales
            revenue = sum(sales_price[p] * model.sales[p, t, s] 
                         for p in model.PRODUCTS for t in model.TIME)
            
            # Knitting costs (first stage)
            knitting_costs = sum(knitting_cost[f] * model.knit[f, t] 
                               for f in model.FABRICS for t in model.TIME)
            
            # Dyeing costs (second stage)
            dyeing_costs = sum(dyeing_cost[f] * model.dye[f, t, s] 
                             for f in model.FABRICS for t in model.TIME)
            
            # Production costs (third stage)
            production_costs = sum(production_cost[p] * model.produce[p, t, s] 
                                 for p in model.PRODUCTS for t in model.TIME)
            
            # Holding costs
            holding_costs = (
                sum(holding_cost_raw * model.raw_inventory[f, t, s] for f in model.FABRICS for t in model.TIME) +
                sum(holding_cost_dyed * model.dyed_inventory[f, t, s] for f in model.FABRICS for t in model.TIME) +
                sum(holding_cost_final * model.final_inventory[p, t, s] for p in model.PRODUCTS for t in model.TIME)
            )
            
            # Shortage costs
            shortage_costs = sum(shortage_cost[p] * model.shortage[p, t, s] 
                               for p in model.PRODUCTS for t in model.TIME)
            
            # Total scenario profit
            scenario_profit = revenue - knitting_costs - dyeing_costs - production_costs - holding_costs - shortage_costs
            
            # Add to expected profit
            total_profit += scenario_probs[s] * scenario_profit
        
        return total_profit
    
    model.Objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)
    
    # ====================== SOLVE THE MODEL ======================
    
    # Try to find the best available solver
    solver = None
    for solver_name in ['cbc', 'glpk', 'ipopt', 'glpk_direct']:
        try:
            solver = SolverFactory(solver_name)
            if solver.available():
                print(f"Using {solver_name} solver")
                break
            else:
                solver = None
        except:
            solver = None
    
    if solver is None:
        raise Exception("No solver available. Please install a supported solver.")
    
    # Solve the model
    start_time = time.time()
    result = solver.solve(model, tee=True)
    solve_time = time.time() - start_time
    
    # ====================== EXTRACT RESULTS ======================
    
    if (result.solver.status == pyo.SolverStatus.ok and 
        result.solver.termination_condition == pyo.TerminationCondition.optimal):
        
        # Extract first-stage decisions (knitting plan)
        knitting_plan = np.zeros((num_fabric_types, planning_horizon))
        for f in model.FABRICS:
            for t in model.TIME:
                knitting_plan[f, t] = pyo.value(model.knit[f, t])
        
        # Extract second-stage decisions (dyeing plan) - average across scenarios
        dyeing_plan = np.zeros((num_fabric_types, planning_horizon))
        for f in model.FABRICS:
            for t in model.TIME:
                dyeing_plan[f, t] = np.mean([pyo.value(model.dye[f, t, s]) for s in model.SCENARIOS])
        
        # Extract third-stage decisions (production plan) - average across scenarios
        production_plan = np.zeros((num_products, planning_horizon))
        for p in model.PRODUCTS:
            for t in model.TIME:
                production_plan[p, t] = np.mean([pyo.value(model.produce[p, t, s]) for s in model.SCENARIOS])
        
        # Calculate expected profit
        expected_profit = pyo.value(model.Objective)
        
        # Prepare detailed scenario results
        scenario_results = []
        for s in model.SCENARIOS:
            # Calculate profit for this scenario
            revenue = sum(sales_price[p] * pyo.value(model.sales[p, t, s]) 
                         for p in model.PRODUCTS for t in model.TIME)
            
            knitting_costs = sum(knitting_cost[f] * pyo.value(model.knit[f, t]) 
                               for f in model.FABRICS for t in model.TIME)
            
            dyeing_costs = sum(dyeing_cost[f] * pyo.value(model.dye[f, t, s]) 
                             for f in model.FABRICS for t in model.TIME)
            
            production_costs = sum(production_cost[p] * pyo.value(model.produce[p, t, s]) 
                                 for p in model.PRODUCTS for t in model.TIME)
            
            holding_costs = (
                sum(holding_cost_raw * pyo.value(model.raw_inventory[f, t, s]) for f in model.FABRICS for t in model.TIME) +
                sum(holding_cost_dyed * pyo.value(model.dyed_inventory[f, t, s]) for f in model.FABRICS for t in model.TIME) +
                sum(holding_cost_final * pyo.value(model.final_inventory[p, t, s]) for p in model.PRODUCTS for t in model.TIME)
            )
            
            shortage_costs = sum(shortage_cost[p] * pyo.value(model.shortage[p, t, s]) 
                               for p in model.PRODUCTS for t in model.TIME)
            
            scenario_profit = revenue - knitting_costs - dyeing_costs - production_costs - holding_costs - shortage_costs
            
            # Service level
            total_demand = sum(demand_scenarios[s, p, t] for p in model.PRODUCTS for t in model.TIME)
            total_sales = sum(pyo.value(model.sales[p, t, s]) for p in model.PRODUCTS for t in model.TIME)
            service_level = total_sales / total_demand if total_demand > 0 else 1.0
            
            scenario_results.append({
                'scenario': s,
                'scenario_type': scenario_types[s],
                'profit': scenario_profit,
                'revenue': revenue,
                'knitting_costs': knitting_costs,
                'dyeing_costs': dyeing_costs,
                'production_costs': production_costs,
                'holding_costs': holding_costs,
                'shortage_costs': shortage_costs,
                'service_level': service_level
            })
        
        return {
            'status': 'optimal',
            'expected_profit': expected_profit,
            'knitting_plan': knitting_plan,
            'dyeing_plan': dyeing_plan,
            'production_plan': production_plan,
            'scenario_results': scenario_results,
            'demand_scenarios': demand_scenarios,
            'scenario_types': scenario_types,
            'solve_time': solve_time
        }
    
    else:
        print(f"Solver status: {result.solver.status}")
        print(f"Termination condition: {result.solver.termination_condition}")
        raise Exception("The solver failed to find an optimal solution.")

def visualize_results(results):
    """Visualize the results of the multi-stage stochastic program."""
    
    # Plot settings
    plt.style.use('ggplot')
    plt.figure(figsize=(15, 12))
    
    # Extract data
    knitting_plan = results['knitting_plan']
    dyeing_plan = results['dyeing_plan']
    production_plan = results['production_plan']
    expected_profit = results['expected_profit']
    scenario_results = results['scenario_results']
    demand_scenarios = results['demand_scenarios']
    scenario_types = results['scenario_types']
    
    # Time periods for x-axis
    time_periods = range(knitting_plan.shape[1])
    
    # 1. Production Plans
    plt.subplot(3, 2, 1)
    fabric_labels = [f"Fabric {i+1}" for i in range(knitting_plan.shape[0])]
    for i in range(knitting_plan.shape[0]):
        plt.plot(time_periods, knitting_plan[i], 'o-', label=fabric_labels[i])
    plt.title('First Stage: Knitting Plan')
    plt.xlabel('Week')
    plt.ylabel('Units to Knit')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 2)
    for i in range(dyeing_plan.shape[0]):
        plt.plot(time_periods, dyeing_plan[i], 'o-', label=fabric_labels[i])
    plt.title('Second Stage: Dyeing Plan (Average)')
    plt.xlabel('Week')
    plt.ylabel('Units to Dye')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 2, 3)
    product_labels = [f"Product {i+1}" for i in range(production_plan.shape[0])]
    for i in range(production_plan.shape[0]):
        plt.plot(time_periods, production_plan[i], 'o-', label=product_labels[i])
    plt.title('Third Stage: Production Plan (Average)')
    plt.xlabel('Week')
    plt.ylabel('Units to Produce')
    plt.legend()
    plt.grid(True)
    
    # 2. Demand Scenarios
    plt.subplot(3, 2, 4)
    scenario_colors = {'base': 'blue', 'upside': 'green', 'downside': 'red', 'volatile': 'purple'}
    
    # Plot demand for product 1 across all scenarios
    for s in range(demand_scenarios.shape[0]):
        scenario_type = scenario_types[s]
        plt.plot(time_periods, demand_scenarios[s, 0], alpha=0.3, 
                 color=scenario_colors[scenario_type])
    
    # Plot the average demand
    avg_demand = np.mean(demand_scenarios[:, 0, :], axis=0)
    plt.plot(time_periods, avg_demand, 'k-', linewidth=2, label='Average Demand')
    
    # Plot the production plan for product 1
    plt.plot(time_periods, production_plan[0], 'r--', linewidth=2, label='Production Plan')
    
    plt.title('Demand Scenarios vs Production Plan (Product 1)')
    plt.xlabel('Week')
    plt.ylabel('Units')
    plt.legend()
    plt.grid(True)
    
    # 3. Scenario Analysis
    plt.subplot(3, 2, 5)
    
    # Group scenarios by type
    scenario_grouped = {}
    for scenario in scenario_results:
        scenario_type = scenario['scenario_type']
        if scenario_type not in scenario_grouped:
            scenario_grouped[scenario_type] = []
        scenario_grouped[scenario_type].append(scenario['profit'])
    
    # Extract data for box plot
    scenario_profits = [scenario_grouped[t] for t in ['base', 'upside', 'downside', 'volatile'] 
                        if t in scenario_grouped]
    scenario_labels = [t for t in ['base', 'upside', 'downside', 'volatile'] 
                      if t in scenario_grouped]
    
    plt.boxplot(scenario_profits, labels=scenario_labels)
    plt.axhline(y=expected_profit, color='r', linestyle='-', label='Expected Profit')
    plt.title('Profit Distribution by Scenario Type')
    plt.xlabel('Scenario Type')
    plt.ylabel('Profit')
    plt.legend()
    plt.grid(True)
    
    # 4. Service Level Analysis
    plt.subplot(3, 2, 6)
    service_levels = [s['service_level'] for s in scenario_results]
    scenario_types_list = [s['scenario_type'] for s in scenario_results]
    
    # Create groups for the scatter plot
    type_to_num = {'base': 0, 'upside': 1, 'downside': 2, 'volatile': 3}
    x = [type_to_num[t] for t in scenario_types_list]
    
    plt.scatter(x, service_levels, c=[scenario_colors[t] for t in scenario_types_list], alpha=0.7)
    
    # Add mean service level by scenario type
    for t, num in type_to_num.items():
        if t in scenario_types_list:
            mean_service = np.mean([s['service_level'] for i, s in enumerate(scenario_results) 
                                   if scenario_types_list[i] == t])
            plt.plot(num, mean_service, 'ko', markersize=10, label=f'{t} mean')
    
    plt.axhline(y=np.mean(service_levels), color='r', linestyle='-', label='Overall Mean')
    plt.xticks([0, 1, 2, 3], ['Base', 'Upside', 'Downside', 'Volatile'])
    plt.title('Service Level by Scenario Type')
    plt.xlabel('Scenario Type')
    plt.ylabel('Service Level (% of Demand Fulfilled)')
    plt.ylim(0, 1.05)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fashion_multistage_results.png')
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("===================")
    print(f"Expected Profit: ${expected_profit:.2f}")
    print(f"Solve Time: {results['solve_time']:.2f} seconds")
    
    # Service level statistics
    mean_service = np.mean([s['service_level'] for s in scenario_results])
    print(f"\nOverall Service Level: {mean_service:.2%}")
    
    # Service level by scenario type
    print("\nService Level by Scenario Type:")
    for t in ['base', 'upside', 'downside', 'volatile']:
        if t in scenario_types_list:
            type_service = np.mean([s['service_level'] for i, s in enumerate(scenario_results) 
                                  if scenario_types_list[i] == t])
            print(f"  {t.capitalize()}: {type_service:.2%}")
    
    # Profit by scenario type
    print("\nProfit by Scenario Type:")
    for t, profits in scenario_grouped.items():
        print(f"  {t.capitalize()}: ${np.mean(profits):.2f} (min: ${min(profits):.2f}, max: ${max(profits):.2f})")
    
    return

# Run the model with different numbers of scenarios
if __name__ == "__main__":
    try:
        # Run the model with 20 scenarios
        results = fashion_multistage_stochastic_program(num_scenarios=20)
        
        # Visualize the results
        visualize_results(results)
        
    except Exception as e:
        print(f"Error: {str(e)}")