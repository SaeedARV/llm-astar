import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.style.use('default')  # Use default style instead of seaborn
import numpy as np
import openai
import os
import time
import math
from llmastar.pather import AStar, LLMAStar

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define different test scenarios
scenarios = [
    {
        "name": "Simple Maze",
        "query": {
            "start": [5, 5],
            "goal": [27, 15],
            "size": [51, 31],
            "horizontal_barriers": [[10, 0, 25], [15, 30, 50]],
            "vertical_barriers": [[25, 10, 22]],
            "range_x": [0, 51],
            "range_y": [0, 31]
        }
    },
    {
        "name": "Complex Maze",
        "query": {
            "start": [5, 5],
            "goal": [35, 5],
            "size": [51, 31],
            "horizontal_barriers": [
                [10, 0, 25], [15, 30, 50],
                [20, 0, 20], [25, 30, 50]
            ],
            "vertical_barriers": [
                [25, 10, 22], [30, 15, 25],
                [35, 5, 15], [40, 20, 30]
            ],
            "range_x": [0, 51],
            "range_y": [0, 31]
        }
    }
]

# Define different LLM configurations to test
llm_configs = [
    {
        "name": "GPT-3.5 Standard",
        "llm": "gpt",
        "prompt": "standard"
    },
    {
        "name": "GPT-3.5 CoT",
        "llm": "gpt",
        "prompt": "cot"
    },
    {
        "name": "GPT-3.5 Repetitive",
        "llm": "gpt",
        "prompt": "repe"
    },
    {
        "name": "GPT-3.5 ReAct",
        "llm": "gpt",
        "prompt": "react"
    },
    {
        "name": "GPT-3.5 Step-Back",
        "llm": "gpt",
        "prompt": "step_back"
    },
    {
        "name": "GPT-3.5 Tree-of-Thoughts",
        "llm": "gpt",
        "prompt": "tot"
    }
]

def run_path_planning(algorithm, query, config=None, scenario_name=None):
    start_time = time.time()
    try:
        if config:
            path_planner = LLMAStar(llm=config["llm"], prompt=config["prompt"])
            filepath = f"{config['name'].lower().replace(' ', '_')}_{scenario_name.lower().replace(' ', '_')}.png"
        else:
            path_planner = AStar()
            filepath = f"astar_{scenario_name.lower().replace(' ', '_')}.png"
        
        result = path_planner.searching(query=query, filepath=filepath)
        end_time = time.time()
        
        # Get path from result
        if isinstance(result, dict):
            if config:  # For LLM results
                path = result.get("llm_output", [])
            else:  # For A* results
                path = result.get("path", [])
        else:
            path = result
        
        return {
            "path": path,
            "time": end_time - start_time,
            "nodes_explored": len(path_planner.CLOSED) if hasattr(path_planner, 'CLOSED') else None
        }
    except Exception as e:
        print(f"Error in {algorithm}: {str(e)}")
        return {
            "path": [],
            "time": time.time() - start_time,
            "nodes_explored": None
        }

def calculate_metrics(path, query):
    if not path:
        return {
            "path_length": float('inf'),
            "path_smoothness": float('inf'),
            "safety_margin": float('inf'),
            "efficiency": float('inf')
        }
    
    # Calculate path length
    path_length = sum(math.sqrt((path[i][0] - path[i+1][0])**2 + (path[i][1] - path[i+1][1])**2) 
                     for i in range(len(path)-1))
    
    # Calculate path smoothness (average angle between consecutive segments)
    angles = []
    for i in range(1, len(path)-1):
        v1 = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
        v2 = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
        norm_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        norm_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        angle = math.acos(dot_product / (norm_v1 * norm_v2))
        angles.append(angle)
    path_smoothness = sum(angles) / len(angles) if angles else 0
    
    # Calculate safety margin (minimum distance to obstacles)
    safety_margin = float('inf')
    for i in range(len(path)):
        point = path[i]
        # Check distance to horizontal barriers
        for barrier in query['horizontal_barriers']:
            y, x_start, x_end = barrier
            if x_start <= point[0] <= x_end:
                dist = abs(point[1] - y)
                if dist < safety_margin:
                    safety_margin = dist
        # Check distance to vertical barriers
        for barrier in query['vertical_barriers']:
            x, y_start, y_end = barrier
            if y_start <= point[1] <= y_end:
                dist = abs(point[0] - x)
                if dist < safety_margin:
                    safety_margin = dist
    
    # Calculate efficiency (path length / straight-line distance)
    start = query['start']
    goal = query['goal']
    straight_line_dist = math.sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
    efficiency = path_length / straight_line_dist if straight_line_dist > 0 else float('inf')
    
    return {
        "path_length": path_length,
        "path_smoothness": path_smoothness,
        "safety_margin": safety_margin,
        "efficiency": efficiency
    }

def plot_comparison(results, scenario):
    # The plotting is now handled by the AStar and LLMAStar classes
    pass

def main():
    print("Note: Make sure to set your OPENAI_API_KEY environment variable!")
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not set!")
        return
        
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        results = {}
        
        # Run traditional A*
        print("Running A*...")
        a_star_result = run_path_planning("A*", scenario["query"], scenario_name=scenario["name"])
        path = a_star_result["path"]
        a_star_metrics = calculate_metrics(path, scenario["query"])
        results["A*"] = {
            "time": a_star_result["time"],
            "nodes_explored": a_star_result["nodes_explored"],
            "metrics": a_star_metrics,
            "path": path
        }
        
        # Run different LLM configurations
        for config in llm_configs:
            print(f"Running {config['name']}...")
            try:
                llm_result = run_path_planning("LLM-A*", scenario["query"], config, scenario_name=scenario["name"])
                path = llm_result["path"]
                llm_metrics = calculate_metrics(path, scenario["query"])
                results[config["name"]] = {
                    "time": llm_result["time"],
                    "nodes_explored": llm_result["nodes_explored"],
                    "metrics": llm_metrics,
                    "path": path
                }
            except Exception as e:
                print(f"Error running {config['name']}: {str(e)}")
                continue
        
        # Plot comparison
        plot_comparison(results, scenario)
        
        # Print detailed results
        print("\nDetailed Results:")
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Computation time: {result['time']:.2f}s")
            print(f"  Path length: {result['metrics']['path_length']:.2f}")
            print(f"  Path smoothness: {result['metrics']['path_smoothness']:.2f}")
            print(f"  Safety margin: {result['metrics']['safety_margin']:.2f}")
            print(f"  Efficiency: {result['metrics']['efficiency']:.2f}")
            if result['nodes_explored']:
                print(f"  Nodes explored: {result['nodes_explored']}")
        
        # Add a separator between scenarios
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main() 