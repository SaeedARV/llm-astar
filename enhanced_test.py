import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.style.use('default')  # Use default style instead of seaborn
import numpy as np
import openai
import os
import time
import math
import json
import random
from llmastar.pather import AStar, LLMAStar
from tabulate import tabulate

# Load dataset
def load_scenarios():
    with open('dataset/environment_50_30.json', 'r') as f:
        data = json.load(f)
    
    # Select three random scenarios from the dataset
    selected_scenarios = random.sample(data, 3)
    scenarios = []
    
    for i, scenario_data in enumerate(selected_scenarios):
        # Take the first start_goal pair from each scenario
        start_goal = scenario_data['start_goal'][0]
        
        scenario = {
            "name": f"Dataset Map {scenario_data['id']}",
            "query": {
                "start": start_goal[0],
                "goal": start_goal[1],
                "size": [51, 31],
                "horizontal_barriers": scenario_data['horizontal_barriers'],
                "vertical_barriers": scenario_data['vertical_barriers'],
                "range_x": scenario_data['range_x'],
                "range_y": scenario_data['range_y']
            }
        }
        scenarios.append(scenario)
    
    return scenarios

# Use dataset scenarios
scenarios = load_scenarios()

# Define different LLM configurations to test
llm_configs = [
    {
        "name": "Llama Standard",
        "llm": "llama",
        "prompt": "standard",
        "use_improved_astar": False
    },
    {
        "name": "Llama CoT",
        "llm": "llama",
        "prompt": "cot",
        "use_improved_astar": False
    },
    {
        "name": "Llama Repetitive",
        "llm": "llama",
        "prompt": "repe",
        "use_improved_astar": False
    },
    {
        "name": "Llama ReAct",
        "llm": "llama",
        "prompt": "react",
        "use_improved_astar": True
    },
    {
        "name": "Llama Step-Back",
        "llm": "llama",
        "prompt": "step_back",
        "use_improved_astar": True
    },
    {
        "name": "Llama Tree-of-Thoughts",
        "llm": "llama",
        "prompt": "tot",
        "use_improved_astar": True
    },
    # Add improved versions with bidirectional A*
    {
        "name": "Llama Standard (Improved)",
        "llm": "llama",
        "prompt": "standard",
        "use_improved_astar": True
    },
    {
        "name": "Llama CoT (Improved)",
        "llm": "llama",
        "prompt": "cot",
        "use_improved_astar": True
    },
    {
        "name": "Llama Repetitive (Improved)",
        "llm": "llama",
        "prompt": "repe",
        "use_improved_astar": True
    }
]

def run_path_planning(algorithm, query, config=None, scenario_name=None):
    start_time = time.time()
    try:
        if config:
            # Create LLMAStar instance with the improved flag if specified
            if "use_improved_astar" in config:
                path_planner = LLMAStar(
                    llm=config["llm"], 
                    prompt=config["prompt"], 
                    use_improved_astar=config.get("use_improved_astar", False)
                )
            else:
                path_planner = LLMAStar(llm=config["llm"], prompt=config["prompt"])
                
            filepath = f"{config['name'].lower().replace(' ', '_')}_{scenario_name.lower().replace(' ', '_')}.png"
        else:
            path_planner = AStar()
            filepath = f"astar_{scenario_name.lower().replace(' ', '_')}.png"
        
        # Use the search method which will select the appropriate algorithm based on the configuration
        if hasattr(path_planner, 'search') and config and "use_improved_astar" in config:
            result = path_planner.search(query=query, filepath=filepath)
        else:
            result = path_planner.searching(query=query, filepath=filepath)
            
        end_time = time.time()
        
        # Extract metrics from the result
        operation_count = result.get("operation", 0)
        storage_used = result.get("storage", 0)
        path_length = result.get("length", float('inf'))
        
        # Check if path can be extracted
        path = []
        if "llm_output" in result:
            path = result["llm_output"]
        
        # Check if the path is valid
        valid_path = is_valid_path(path, query)
        
        return {
            "path": path,
            "time": end_time - start_time,
            "operation_count": operation_count,
            "storage_used": storage_used,
            "path_length": path_length,
            "valid_path": valid_path
        }
    except Exception as e:
        print(f"Error in {algorithm}: {str(e)}")
        return {
            "path": [],
            "time": time.time() - start_time,
            "operation_count": 0,
            "storage_used": 0,
            "path_length": float('inf'),
            "valid_path": False
        }

def is_valid_path(path, query):
    """Check if the path is valid (collision-free and connects start to goal)"""
    if not path or len(path) < 2:
        return False
    
    # Check if path starts at start and ends at goal
    start = tuple(query['start'])
    goal = tuple(query['goal'])
    
    if path[0] != start or path[-1] != goal:
        return False
    
    # Create simple obstacle checking
    horizontal_barriers = query['horizontal_barriers']
    vertical_barriers = query['vertical_barriers']
    
    # Check each segment for collisions
    for i in range(len(path) - 1):
        segment = [path[i], path[i+1]]
        
        # Check horizontal barriers
        for barrier in horizontal_barriers:
            y, x_start, x_end = barrier
            barrier_segment = [[x_start, y], [x_end, y]]
            if segments_intersect(segment, barrier_segment):
                return False
        
        # Check vertical barriers
        for barrier in vertical_barriers:
            x, y_start, y_end = barrier
            barrier_segment = [[x, y_start], [x, y_end]]
            if segments_intersect(segment, barrier_segment):
                return False
    
    return True

def segments_intersect(seg1, seg2):
    """
    Check if two line segments intersect
    Implementation based on the orientation method from computational geometry
    """
    def orientation(p, q, r):
        # Calculate orientation of triplet (p, q, r)
        # Returns: 0 if collinear, 1 if clockwise, 2 if counterclockwise
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # collinear
        return 1 if val > 0 else 2  # clockwise or counterclockwise
    
    def on_segment(p, q, r):
        # Check if point q lies on line segment 'pr'
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    p1, q1 = seg1
    p2, q2 = seg2
    
    # Calculate the four orientations needed
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    # General case
    if o1 != o2 and o3 != o4:
        return True
    
    # Special Cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    
    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    
    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    
    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    
    # No intersection
    return False

def create_comparison_table(results, scenario):
    """Create a comparison table of metrics across different algorithms"""
    # Get A* result as baseline
    a_star_result = results.get("A*", {})
    a_star_ops = a_star_result.get("operation_count", 1)
    a_star_storage = a_star_result.get("storage_used", 1)
    a_star_path_length = a_star_result.get("path_length", 1)
    
    table_data = []
    
    # Add A* as baseline (100%)
    table_data.append([
        "A*", "-", "-", "-",
        "100", "100", "100", 
        "100" if a_star_result.get("valid_path", False) else "0"
    ])
    
    # Add Dynamic WA* (w=2) row with placeholder values
    # These are example values based on the image provided
    table_data.append([
        "Dynamic WA* (w = 2)", "-", "-", "-",
        "60.91", "78.53", "100.24",
        "100"
    ])
    
    # Group results by algorithm type (standard vs improved)
    standard_results = {name: result for name, result in results.items() 
                      if "Llama" in name and "Improved" not in name}
    improved_results = {name: result for name, result in results.items() 
                      if "Llama" in name and "Improved" in name}
    
    # Add standard LLM-A* results
    for name, result in standard_results.items():
        prompt_type = name.replace("Llama ", "")
        
        # Calculate relative metrics
        op_ratio = result.get("operation_count", 0) / a_star_ops * 100 if a_star_ops else 0
        storage_ratio = result.get("storage_used", 0) / a_star_storage * 100 if a_star_storage else 0
        path_length_ratio = result.get("path_length", float('inf')) / a_star_path_length * 100 if a_star_path_length else 0
        valid_path_ratio = 100 if result.get("valid_path", False) else 0
        
        table_data.append([
            "LLM-A* (Original)", "Llama", prompt_type, "No",
            f"{op_ratio:.2f}", f"{storage_ratio:.2f}", f"{path_length_ratio:.2f}",
            f"{valid_path_ratio:.2f}"
        ])
    
    # Add improved LLM-A* results
    for name, result in improved_results.items():
        prompt_type = name.replace("Llama ", "").replace(" (Improved)", "")
        
        # Calculate relative metrics
        op_ratio = result.get("operation_count", 0) / a_star_ops * 100 if a_star_ops else 0
        storage_ratio = result.get("storage_used", 0) / a_star_storage * 100 if a_star_storage else 0
        path_length_ratio = result.get("path_length", float('inf')) / a_star_path_length * 100 if a_star_path_length else 0
        valid_path_ratio = 100 if result.get("valid_path", False) else 0
        
        table_data.append([
            "LLM-A* (Improved)", "Llama", prompt_type, "Yes",
            f"{op_ratio:.2f}", f"{storage_ratio:.2f}", f"{path_length_ratio:.2f}",
            f"{valid_path_ratio:.2f}"
        ])
    
    # Create the table
    headers = [
        "Methodology", "Base Model", "Prompt Approach", "Bidirectional",
        "Operation Ratio ↓ (%)", "Storage Ratio ↓ (%)", 
        "Relative Path Length ↓ (%)", "Valid Path Ratio ↑ (%)"
    ]
    
    table = tabulate(table_data, headers=headers, tablefmt="pipe")
    return table

def main():
        
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        results = {}
        
        # Run traditional A*
        print("Running A*...")
        a_star_result = run_path_planning("A*", scenario["query"], scenario_name=scenario["name"])
        results["A*"] = a_star_result
        
        # Run different LLM configurations
        for config in llm_configs:
            print(f"Running {config['name']}...")
            try:
                llm_result = run_path_planning("LLM-A*", scenario["query"], config, scenario_name=scenario["name"])
                results[config["name"]] = llm_result
            except Exception as e:
                print(f"Error running {config['name']}: {str(e)}")
                continue
        
        # Create and print comparison table
        table = create_comparison_table(results, scenario)
        print(f"\nComparison Table for {scenario['name']}:")
        print(table)
        
        # Print detailed results
        print("\nDetailed Results:")
        for name, result in results.items():
            path_status = "Valid" if result.get("valid_path", False) else "Invalid"
            path_length = result.get("path_length", float('inf'))
            if path_length == float('inf'):
                path_length = "N/A"
            else:
                path_length = f"{path_length:.2f}"
            
            print(f"{name}:")
            print(f"  Time: {result.get('time', 0):.2f}s")
            print(f"  Operations: {result.get('operation_count', 0)}")
            print(f"  Storage: {result.get('storage_used', 0)}")
            print(f"  Path Length: {path_length}")
            print(f"  Path Status: {path_status}")
            print("")

if __name__ == "__main__":
    main() 