"""
Smart Bus Demand Prediction System - Optimizer Module

This module converts predicted passenger demand into actionable bus
allocation recommendations. It determines how many buses should be
running to handle predicted demand efficiently.
"""

import logging
import math
from typing import Dict, Literal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_bus_allocation(
    predicted_passengers: float,
    bus_capacity: int = 40,
    current_buses_running: int = 1,
) -> Dict[str, any]:
    """
    Convert predicted passenger demand into bus allocation recommendations.
    
    Args:
        predicted_passengers: Number of passengers expected (from predictor)
        bus_capacity: Passenger capacity per bus (default 40)
        current_buses_running: Number of buses currently scheduled
        
    Returns:
        Dictionary with:
        {
            "predicted_passengers": int,
            "bus_capacity": int,
            "current_buses": int,
            "required_buses": int,
            "action": str ("add_buses", "reduce_buses", "maintain"),
            "additional_buses_needed": int (positive to add, negative to remove),
            "utilization_percentage": float (% of capacity being used),
            "recommendation": str (human-readable explanation)
        }
        
    Logic:
        1. Calculate required buses: ceil(predicted_passengers / bus_capacity)
        2. Compare with current buses running
        3. If required > current: recommend adding buses
        4. If required < current: recommend reducing buses
        5. Else: recommend maintaining current schedule
        
    Utilization:
        - Shows percentage of bus capacity that will be used
        - Helps identify if buses will be overcrowded or underutilized
    """
    
    # Input validation
    if predicted_passengers < 0:
        logger.warning(f"Negative passenger count: {predicted_passengers}, setting to 0")
        predicted_passengers = 0
    
    if bus_capacity <= 0:
        raise ValueError(f"Bus capacity must be positive, got {bus_capacity}")
    
    if current_buses_running < 0:
        logger.warning(f"Negative bus count: {current_buses_running}, setting to 0")
        current_buses_running = 0
    
    # Calculate required buses to handle predicted demand
    required_buses = math.ceil(predicted_passengers / bus_capacity)
    
    # Handle edge case: if no buses needed but some are running, might reduce
    if required_buses == 0 and current_buses_running > 0:
        required_buses = 0
    # Ensure at least 1 bus if there's any demand
    elif predicted_passengers > 0 and required_buses == 0:
        required_buses = 1
    
    # Calculate difference
    buses_difference = required_buses - current_buses_running
    
    # Determine action
    if buses_difference > 0:
        action: Literal["add_buses", "reduce_buses", "maintain"] = "add_buses"
        recommendation = (
            f"Add {buses_difference} bus(es) to handle predicted demand. "
            f"{required_buses} buses needed vs {current_buses_running} running."
        )
    elif buses_difference < 0:
        action = "reduce_buses"
        recommendation = (
            f"Reduce {abs(buses_difference)} bus(es) to save resources. "
            f"Only {required_buses} buses needed vs {current_buses_running} running."
        )
    else:
        action = "maintain"
        recommendation = (
            f"Maintain current schedule with {current_buses_running} buses. "
            f"Exactly meets predicted demand."
        )
    
    # Calculate utilization percentage
    if required_buses > 0:
        total_capacity = required_buses * bus_capacity
        utilization = (predicted_passengers / total_capacity) * 100
    else:
        utilization = 0.0
    
    # Log recommendation
    logger.info(
        f"Optimization: {action.upper()} | "
        f"Current: {current_buses_running} buses, Required: {required_buses} buses, "
        f"Passengers: {predicted_passengers:.0f} (utilization: {utilization:.1f}%)"
    )
    
    return {
        "predicted_passengers": int(predicted_passengers),
        "bus_capacity": bus_capacity,
        "current_buses": current_buses_running,
        "required_buses": required_buses,
        "action": action,
        "additional_buses_needed": buses_difference,
        "utilization_percentage": round(utilization, 2),
        "recommendation": recommendation,
    }


def batch_optimize(
    predictions: list,
    bus_capacity: int = 40,
) -> Dict[str, any]:
    """
    Optimize bus allocation for multiple stops/routes.
    
    Args:
        predictions: List of dicts with keys:
            - "stop_name": str
            - "predicted_demand": int
            - "current_buses": int
            
        bus_capacity: Passenger capacity per bus
        
    Returns:
        Dictionary with:
        {
            "total_passengers": int,
            "total_current_buses": int,
            "total_required_buses": int,
            "total_change": int,
            "optimizations": [dict, ...] (results for each stop)
            "summary": str
        }
        
    Use Case:
        When you have predictions for multiple stops and want an
        overall optimization strategy.
    """
    if not predictions:
        logger.warning("Empty prediction list for batch optimization")
        return {
            "total_passengers": 0,
            "total_current_buses": 0,
            "total_required_buses": 0,
            "total_change": 0,
            "optimizations": [],
            "summary": "No predictions to optimize",
        }
    
    optimizations = []
    total_passengers = 0
    total_current = 0
    total_required = 0
    
    for pred in predictions:
        result = optimize_bus_allocation(
            predicted_passengers=pred.get("predicted_demand", 0),
            bus_capacity=bus_capacity,
            current_buses_running=pred.get("current_buses", 1),
        )
        result["stop_name"] = pred.get("stop_name", "Unknown")
        optimizations.append(result)
        
        total_passengers += result["predicted_passengers"]
        total_current += result["current_buses"]
        total_required += result["required_buses"]
    
    total_change = total_required - total_current
    
    # Generate summary
    if total_change > 0:
        summary = f"Network needs {total_change} additional buses to handle {total_passengers} passengers"
    elif total_change < 0:
        summary = f"Network can reduce {abs(total_change)} buses while handling {total_passengers} passengers"
    else:
        summary = f"Current bus allocation is optimal for {total_passengers} passengers"
    
    logger.info(summary)
    
    return {
        "total_passengers": total_passengers,
        "total_current_buses": total_current,
        "total_required_buses": total_required,
        "total_change": total_change,
        "optimizations": optimizations,
        "summary": summary,
    }


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 70)
    print("Bus Allocation Optimizer - Example Runs")
    print("=" * 70)
    
    # Example 1: Need to add buses
    print("\n[Example 1: High Demand - Add Buses]")
    result1 = optimize_bus_allocation(
        predicted_passengers=120,
        bus_capacity=40,
        current_buses_running=2,
    )
    print(f"  Predicted Passengers: {result1['predicted_passengers']}")
    print(f"  Current Buses: {result1['current_buses']}")
    print(f"  Required Buses: {result1['required_buses']}")
    print(f"  Action: {result1['action'].upper()}")
    print(f"  Additional Buses Needed: {result1['additional_buses_needed']}")
    print(f"  Utilization: {result1['utilization_percentage']}%")
    print(f"  → {result1['recommendation']}")
    
    # Example 2: Can reduce buses
    print("\n[Example 2: Low Demand - Reduce Buses]")
    result2 = optimize_bus_allocation(
        predicted_passengers=35,
        bus_capacity=40,
        current_buses_running=3,
    )
    print(f"  Predicted Passengers: {result2['predicted_passengers']}")
    print(f"  Current Buses: {result2['current_buses']}")
    print(f"  Required Buses: {result2['required_buses']}")
    print(f"  Action: {result2['action'].upper()}")
    print(f"  Additional Buses Needed: {result2['additional_buses_needed']}")
    print(f"  Utilization: {result2['utilization_percentage']}%")
    print(f"  → {result2['recommendation']}")
    
    # Example 3: Optimal allocation
    print("\n[Example 3: Just Right - Maintain Schedule]")
    result3 = optimize_bus_allocation(
        predicted_passengers=80,
        bus_capacity=40,
        current_buses_running=2,
    )
    print(f"  Predicted Passengers: {result3['predicted_passengers']}")
    print(f"  Current Buses: {result3['current_buses']}")
    print(f"  Required Buses: {result3['required_buses']}")
    print(f"  Action: {result3['action'].upper()}")
    print(f"  Additional Buses Needed: {result3['additional_buses_needed']}")
    print(f"  Utilization: {result3['utilization_percentage']}%")
    print(f"  → {result3['recommendation']}")
    
    # Example 4: Batch optimization for multiple stops
    print("\n[Example 4: Batch Optimization - Multiple Stops]")
    predictions = [
        {"stop_name": "Central Station", "predicted_demand": 150, "current_buses": 3},
        {"stop_name": "Downtown Hub", "predicted_demand": 85, "current_buses": 3},
        {"stop_name": "Suburb Stop A", "predicted_demand": 45, "current_buses": 2},
        {"stop_name": "Suburb Stop B", "predicted_demand": 200, "current_buses": 4},
    ]
    batch_result = batch_optimize(predictions, bus_capacity=40)
    
    print(f"  Total Passengers: {batch_result['total_passengers']}")
    print(f"  Total Current Buses: {batch_result['total_current_buses']}")
    print(f"  Total Required Buses: {batch_result['total_required_buses']}")
    print(f"  Total Change: {batch_result['total_change']:+d}")
    print(f"\n  Summary: {batch_result['summary']}")
    
    print(f"\n  Detailed Recommendations:")
    for opt in batch_result['optimizations']:
        print(f"    • {opt['stop_name']}: {opt['action']} ({opt['additional_buses_needed']:+d}) "
              f"→ {opt['utilization_percentage']}% capacity")
