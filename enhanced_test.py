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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import partial
from llmastar.pather import AStar, LLMAStar
from tabulate import tabulate

# Set multiprocessing start method to 'spawn' to avoid CUDA initialization issues
multiprocessing.set_start_method('spawn', force=True)

# Load a random grid from the dataset
def load_random_grid():
    with open('/content/llm-astar/dataset/environment_50_30.json', 'r') as f:
        data = json.load(f)
    
    # Make sure we have grids to choose from
    if not data:
        raise ValueError("No grids found in the dataset. Please run generate_complex_grid.py first.")
    
    # Pick a random grid from the dataset
    random_grid = random.choice(data)
    
    # Create a scenario from it
    scenario = {
        "name": f"Complex Grid {random_grid['id']}",
        "query": {
            "start": random_grid['start_goal'][0][0],
            "goal": random_grid['start_goal'][0][1],
            "size": [random_grid['range_x'][1], random_grid['range_y'][1]],
            "horizontal_barriers": random_grid['horizontal_barriers'],
            "vertical_barriers": random_grid['vertical_barriers'],
            "range_x": random_grid['range_x'],
            "range_y": random_grid['range_y']
        }
    }
    
    print(f"Loaded random complex grid with ID {random_grid['id']}")
    print(f"Grid dimensions: {random_grid['range_x'][1]}x{random_grid['range_y'][1]}")
    print(f"Start: {random_grid['start_goal'][0][0]}, Goal: {random_grid['start_goal'][0][1]}")
    
    return [scenario]  # Return as a list to maintain compatibility with existing code

# Use a random grid from the dataset
scenarios = load_random_grid()

# Define different LLM configurations to test
llm_configs = [
    {
        "name": "Llama ReAct (Improved)",
        "llm": "llama",
        "prompt": "react",
        "use_improved_astar": True
    },
    {
        "name": "Llama Step-Back (Improved)",
        "llm": "llama",
        "prompt": "step_back",
        "use_improved_astar": True
    },
    {
        "name": "Llama Tree-of-Thoughts (Improved)",
        "llm": "llama",
        "prompt": "tot",
        "use_improved_astar": True
    },
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

# Function to run A* on a single segment (for parallel processing)
def process_segment(start, end, query, use_improved_astar=False):
    """Process a single path segment using A* or improved A*"""
    # Create a new query with the segment's start and end points
    segment_query = query.copy()
    segment_query['start'] = start
    segment_query['goal'] = end
    
    # Create a new A* instance for this segment
    if use_improved_astar:
        path_planner = LLMAStar(llm="llama", prompt="standard", use_improved_astar=True)
        result = path_planner.searching_improved(segment_query)
    else:
        path_planner = AStar()
        result = path_planner.searching(segment_query)
    
    return {
        'start': start,
        'end': end,
        'path': result.get('path', []),
        'operation_count': result.get('operation', 0),
        'storage_used': result.get('storage', 0),
        'length': result.get('length', 0)
    }

# Function to run the pathfinding in parallel across all waypoints
def parallel_pathfinding(waypoints, query, use_improved_astar=False, max_workers=4):
    """Process pathfinding between waypoints in parallel"""
    if not waypoints or len(waypoints) < 2:
        return {'operation_count': 0, 'storage_used': 0, 'length': 0, 'path': []}
    
    # Create the segment pairs
    segments = []
    for i in range(len(waypoints) - 1):
        segments.append((waypoints[i], waypoints[i+1]))
    
    # Process segments in parallel
    start_time = time.time()
    results = []
    
    # Create a partial function with the query and improved flag
    process_func = partial(process_segment, 
                          query=query, 
                          use_improved_astar=use_improved_astar)
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid CUDA issues
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all segment tasks
        future_to_segment = {
            executor.submit(process_func, start, end): (start, end) 
            for start, end in segments
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_segment):
            segment = future_to_segment[future]
            # try:
            result = future.result()
            results.append(result)
            # except Exception as e:
            #     print(f"Error processing segment {segment}: {e}")
    
    # Sort results by start position to maintain order
    # We need to define a key function that matches the original segment order
    segment_order = {tuple(segments[i]): i for i in range(len(segments))}
    results.sort(key=lambda x: segment_order.get((tuple(x['start']), tuple(x['end'])), float('inf')))
    
    # Combine paths (removing duplicates at junctions)
    combined_path = [waypoints[0]]  # Start with the first point
    total_ops = 0
    total_storage = 0
    total_length = 0
    
    for result in results:
        # Add the path segment (excluding the start point)
        path_segment = result.get('path', [])
        if path_segment and len(path_segment) > 1:
            combined_path.extend(path_segment[1:])  # Skip the first point to avoid duplicates
        
        # Accumulate metrics
        total_ops += result.get('operation_count', 0)
        total_storage += result.get('storage_used', 0)
        total_length += result.get('length', 0)
    
    execution_time = time.time() - start_time
    
    return {
        'path': combined_path,
        'operation_count': total_ops,
        'storage_used': total_storage,
        'length': total_length,
        'time': execution_time
    }

# Sequential version of pathfinding for cases where parallel execution fails
def sequential_pathfinding(waypoints, query, use_improved_astar=False):
    """Process pathfinding between waypoints sequentially as a fallback"""
    if not waypoints or len(waypoints) < 2:
        return {'operation_count': 0, 'storage_used': 0, 'length': 0, 'path': []}
    
    start_time = time.time()
    combined_path = [waypoints[0]]  # Start with the first point
    total_ops = 0
    total_storage = 0
    total_length = 0
    
    # Process each segment sequentially
    for i in range(len(waypoints) - 1):
        start = waypoints[i]
        end = waypoints[i+1]
        
        # Create a new query with the segment's start and end points
        segment_query = query.copy()
        segment_query['start'] = start
        segment_query['goal'] = end
        
        # Create a new A* instance for this segment
        if use_improved_astar:
            path_planner = LLMAStar(llm="llama", prompt="standard", use_improved_astar=True)
            result = path_planner.searching_improved(segment_query)
        else:
            path_planner = AStar()
            result = path_planner.searching(segment_query)
        
        # Add the path segment (excluding the start point)
        path_segment = result.get('path', [])
        if path_segment and len(path_segment) > 1:
            combined_path.extend(path_segment[1:])  # Skip the first point to avoid duplicates
        
        # Accumulate metrics
        total_ops += result.get('operation', 0)
        total_storage += result.get('storage', 0)
        total_length += result.get('length', 0)
    
    execution_time = time.time() - start_time
    
    return {
        'path': combined_path,
        'operation_count': total_ops,
        'storage_used': total_storage,
        'length': total_length,
        'time': execution_time
    }

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
            
            # Use the search method which will select the appropriate algorithm based on the configuration
            if hasattr(path_planner, 'search') and config and "use_improved_astar" in config:
                result = path_planner.search(query=query, filepath=filepath)
            else:
                result = path_planner.searching(query=query, filepath=filepath)
                
            # Extract the waypoints from the LLM output
            waypoints = []
            if "llm_output" in result:
                waypoints = result["llm_output"]
                
                # Print LLM response details
                print("\n=== LLM RESPONSE DETAILS ===")
                print(f"Config: {config['name']}")
                print(f"Raw LLM output: {result.get('llm_output', 'None')}")
                print(f"Number of waypoints: {len(waypoints)}")
                print(f"Waypoints: {waypoints}")
                if len(waypoints) == 2:
                    print("Only start and goal points were returned by LLM")
                elif len(waypoints) < 2:
                    print("Warning: Insufficient waypoints returned by LLM")
                else:
                    print(f"Intermediate waypoints: {waypoints[1:-1]}")
                print("============================\n")
                
                # Process waypoints in parallel if we have valid waypoints
                if waypoints and len(waypoints) >= 2:
                    print(f"Processing {len(waypoints)-1} path segments in parallel...")
                    print(query)
                    
                    try:
                        # Try parallel execution first
                        parallel_result = parallel_pathfinding(
                            waypoints=waypoints,
                            query=query,
                            use_improved_astar=config.get("use_improved_astar", False)
                        )
                    except Exception as e:
                        # Fall back to sequential execution if parallel fails
                        print(f"Parallel execution failed: {e}")
                        print("Falling back to sequential processing...")
                        parallel_result = sequential_pathfinding(
                            waypoints=waypoints,
                            query=query,
                            use_improved_astar=config.get("use_improved_astar", False)
                        )
                    
                    # Update the result with processing metrics
                    result["operation"] = parallel_result["operation_count"]
                    result["storage"] = parallel_result["storage_used"]
                    result["length"] = parallel_result["length"]
                    result["parallel_path"] = parallel_result["path"]
            
        else:
            # Standard A* without LLM
            path_planner = AStar()
            filepath = f"astar_{scenario_name.lower().replace(' ', '_')}.png"
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
            
            # If we have a parallel path, use that for validation
            if "parallel_path" in result:
                path = result["parallel_path"]
        
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
    # Create a results directory
    results_dir = 'test_results'
    os.makedirs(results_dir, exist_ok=True)
        
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
        
        # Save table to file
        table_file = os.path.join(results_dir, f"{scenario['name'].lower().replace(' ', '_')}_comparison.txt")
        with open(table_file, 'w') as f:
            f.write(f"Comparison Table for {scenario['name']}:\n")
            f.write(table)
        
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
        
        # Save detailed results to file
        details_file = os.path.join(results_dir, f"{scenario['name'].lower().replace(' ', '_')}_details.txt")
        with open(details_file, 'w') as f:
            f.write(f"Detailed Results for {scenario['name']}:\n\n")
            for name, result in results.items():
                path_status = "Valid" if result.get("valid_path", False) else "Invalid"
                path_length = result.get("path_length", float('inf'))
                if path_length == float('inf'):
                    path_length = "N/A"
                else:
                    path_length = f"{path_length:.2f}"
                
                f.write(f"{name}:\n")
                f.write(f"  Time: {result.get('time', 0):.2f}s\n")
                f.write(f"  Operations: {result.get('operation_count', 0)}\n")
                f.write(f"  Storage: {result.get('storage_used', 0)}\n")
                f.write(f"  Path Length: {path_length}\n")
                f.write(f"  Path Status: {path_status}\n\n")
        
        print(f"Results saved to {results_dir} directory.")

if __name__ == "__main__":
    main() 