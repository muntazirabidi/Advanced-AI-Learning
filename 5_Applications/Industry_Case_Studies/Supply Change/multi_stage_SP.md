# Multi-Stage Stochastic Programming for Fashion Industry Production Planning

## Problem Definition

The fashion industry faces a complex production planning challenge that requires making sequential decisions under uncertainty. This problem involves a three-stage production process:

1. **Knitting** (raw fabric production)
2. **Dyeing** (fabric treatment)
3. **Final Product Manufacturing**

**Key Challenges:**

- Long lead times between stages (knitting → dyeing → final product)
- Uncertain demand that follows various patterns
- Limited production capacity at each stage
- Multiple products with different material requirements
- Balancing inventory costs, production costs, and shortage penalties

The ultimate goal is to determine the optimal production quantities at each stage to maximize expected profit while maintaining adequate service levels across different possible demand scenarios.

## Stochastic Programming Approach

Stochastic programming is ideal for this problem because it explicitly handles uncertainty by:

1. Modeling multiple possible future demand scenarios
2. Making initial commitments (knitting decisions) before full information is available
3. Allowing recourse decisions (dyeing and production) that can adapt to revealed information
4. Optimizing for the expected outcome across all scenarios

## Model Implementation

### 1. Scenario Generation

We model four distinct demand patterns that represent different forecast error behaviors:

- **Base Scenarios**: Moderate variations around the forecast
- **Upside Scenarios**: Demand trending higher than forecast
- **Downside Scenarios**: Demand trending lower than forecast
- **Volatile Scenarios**: High variations regardless of trend

```python
# Generate different scenario types to represent forecast patterns
scenario_types = np.random.choice(['base', 'upside', 'downside', 'volatile'],
                                 size=num_scenarios,
                                 p=[0.4, 0.2, 0.2, 0.2])

for s in range(num_scenarios):
    scenario_type = scenario_types[s]

    # Different logic for each scenario type
    if scenario_type == 'base':
        noise = np.random.normal(0, 0.1 * base_demand[p], planning_horizon)
        demand_scenarios[s, p] = base_demand[p] * (1 + noise)
    # Other scenario types similarly defined...
```

### 2. Decision Variables

The model defines decision variables for each stage of the production process:

```python
# First stage decision variables (knitting decisions)
model.knit = pyo.Var(model.FABRICS, model.TIME, domain=pyo.NonNegativeReals)

# Second stage decision variables (dyeing decisions)
model.dye = pyo.Var(model.FABRICS, model.TIME, model.SCENARIOS, domain=pyo.NonNegativeReals)

# Third stage decision variables (production decisions)
model.produce = pyo.Var(model.PRODUCTS, model.TIME, model.SCENARIOS, domain=pyo.NonNegativeReals)
```

Note that:

- First-stage variables (knitting) don't depend on scenarios - these decisions are made before scenario outcomes are known
- Second and third-stage variables include scenario indices - these decisions can adapt to each scenario

### 3. Constraints

The model incorporates several types of constraints:

#### Capacity Constraints

Each production stage has limited capacity:

```python
# Knitting capacity constraint
def knitting_capacity_rule(model, t):
    return sum(model.knit[f, t] for f in model.FABRICS) <= knitting_capacity
```

#### Lead Time Constraints

Production at each stage can only occur after the required lead time:

```python
# Production capacity constraint with lead time
def production_capacity_rule(model, t, s):
    if t >= total_lead_time:  # Can only produce after total lead time
        return sum(model.produce[p, t, s] for p in model.PRODUCTS) <= final_production_capacity
    else:
        return sum(model.produce[p, t, s] for p in model.PRODUCTS) <= 0  # No production in early periods
```

#### Inventory Balance Constraints

Track material flow through the production process:

```python
# Raw fabric inventory balance
def raw_inventory_balance_rule(model, f, t, s):
    if t == 0:
        return model.raw_inventory[f, t, s] == model.knit[f, t] - model.dye[f, t, s]
    else:
        return model.raw_inventory[f, t, s] == model.raw_inventory[f, t-1, s] + model.knit[f, t] - model.dye[f, t, s]
```

#### Demand Satisfaction

Sales plus shortages must equal demand:

```python
def demand_satisfaction_rule(model, p, t, s):
    return model.sales[p, t, s] + model.shortage[p, t, s] == demand_scenarios[s, p, t]
```

### 4. Objective Function

The objective function maximizes expected profit across all scenarios:

```python
def objective_rule(model):
    # Expected profit across all scenarios
    total_profit = 0

    for s in model.SCENARIOS:
        # Revenue from sales
        revenue = sum(sales_price[p] * model.sales[p, t, s]
                     for p in model.PRODUCTS for t in model.TIME)

        # Various costs...

        # Total scenario profit
        scenario_profit = revenue - knitting_costs - dyeing_costs - production_costs -
                          holding_costs - shortage_costs

        # Add to expected profit (weighted by scenario probability)
        total_profit += scenario_probs[s] * scenario_profit

    return total_profit
```

## Results Interpretation

### Analysis of the Output Graphs

#### 1. First Stage: Knitting Plan

- Production starts in week 3-4 (allowing time for the entire pipeline)
- Fabric 1 has the highest production volume, corresponding to product requirements
- There's a buildup of knitting in weeks 6-8, then a decrease in the final periods

#### 2. Second Stage: Dyeing Plan

- Dyeing follows the knitting pattern with a time lag of approximately 3 weeks (the knitting lead time)
- The quantities closely match the knitting quantities, indicating minimal raw inventory holding

#### 3. Third Stage: Production Plan

- Production begins at week 5 (after the total lead time of 5 weeks)
- Product 2 initially sees the highest production volume in week 5
- Products 1 and 2 have similar production volumes in the middle of the horizon
- Production decreases toward the end of the planning horizon

#### 4. Demand Scenarios vs. Production Plan

- The blue lines show individual demand scenarios, which vary significantly
- The black line shows average demand
- The red dashed line shows the production plan for Product 1
- The production plan is much lower than many demand scenarios, indicating the model is conservative due to high shortage costs

#### 5. Profit Distribution by Scenario Type

- "Upside" scenarios yield the highest profits
- "Base" scenarios show high variability in profit
- "Downside" scenarios have lower but more consistent profits
- The red line shows expected profit across all scenarios

#### 6. Service Level by Scenario Type

- Service levels (% of demand fulfilled) vary significantly by scenario type
- Overall service level is around 40% (red line)
- Upside scenarios have better service levels than base scenarios
- The low overall service level suggests the model is balancing the costs of production against the costs of shortages

### Key Insights

1. **Conservative Production Strategy**: The model recommends a cautious production plan that doesn't try to meet all possible demand scenarios. This suggests the costs of overproduction outweigh the costs of potential shortages in this particular parameter setting.

2. **Just-in-Time Production**: The close alignment between knitting, dyeing, and production plans suggests minimal inventory holding between stages, approaching a just-in-time production strategy.

3. **Scenario-Specific Adaptation**: The model performs differently across scenario types, with better outcomes in upside and volatile scenarios compared to base scenarios.

4. **Lead-Time Planning**: The production plans respect the lead times between stages, with production starting only after sufficient lead time has passed for the entire production pipeline.

## Practical Applications

This model can help fashion industry planners:

1. Determine optimal initial knitting quantities when demand is uncertain
2. Plan dyeing and production quantities that adapt to different demand scenarios
3. Analyze the trade-offs between service level and production costs
4. Evaluate the impact of different demand patterns on profitability
5. Assess the benefits of additional production capacity at different stages

## Potential Model Extensions

The model could be extended to include:

1. Seasonal demand patterns specific to fashion
2. Color and size variations within products
3. Alternative sourcing options with different lead times
4. Markdown pricing for excess inventory
5. Learning mechanisms to update demand forecasts over time

## Conclusion

Multi-stage stochastic programming provides a powerful framework for addressing the complex production planning challenges in the fashion industry. By explicitly modeling uncertainty through scenarios and allowing for recourse decisions at later stages, the approach enables more robust decision-making that balances the competing objectives of cost minimization, profit maximization, and service level targets.
