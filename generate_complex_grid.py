#!/usr/bin/env python3
"""
Generate complex grid maps for testing A* algorithms.
This script creates maze-like grid environments that are more challenging 
than the standard environments, similar to those used in improved A* algorithms.
The grids are saved in the dataset/environment_50_30.json file to be used by enhanced_test.py.
"""

import argparse
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt

def generate_complex_gridmap(width=50, height=30, obstacle_density=0.3):
    """
    Generate a complex grid-based environment with maze-like structures.
    
    Args:
        width: Width of the grid
        height: Height of the grid
        obstacle_density: Percentage of grid cells to be filled with obstacles (0.0-1.0)
            
    Returns:
        A dictionary containing the grid data
    """
    # Initialize an empty grid (0 = passable, 1 = obstacle)
    grid = np.zeros((height, width), dtype=int)
    
    # Add boundary walls
    grid[0, :] = 1  # Top wall
    grid[height-1, :] = 1  # Bottom wall
    grid[:, 0] = 1  # Left wall
    grid[:, width-1] = 1  # Right wall
        
    # Randomly generate obstacles based on density
    for i in range(1, height-1):
        for j in range(1, width-1):
            if random.random() < obstacle_density:
                grid[i, j] = 1
                
    # Add maze-like patterns - horizontal walls with gaps
    for i in range(3, height-3, 8):
        # Horizontal walls with gaps
        wall_length = random.randint(width//3, width-5)
        start_pos = random.randint(1, width-wall_length-1)
        grid[i, start_pos:start_pos+wall_length] = 1
        # Create 1-2 gaps in the wall
        for _ in range(random.randint(1, 2)):
            gap_pos = random.randint(start_pos, start_pos+wall_length-1)
            grid[i, gap_pos] = 0
            
    # Add maze-like patterns - vertical walls with gaps
    for j in range(3, width-3, 8):
        # Vertical walls with gaps
        wall_length = random.randint(height//3, height-5)
        start_pos = random.randint(1, height-wall_length-1)
        grid[start_pos:start_pos+wall_length, j] = 1
        # Create 1-2 gaps in the wall
        for _ in range(random.randint(1, 2)):
            gap_pos = random.randint(start_pos, start_pos+wall_length-1)
            grid[gap_pos, j] = 0
            
    # Create some corridors to ensure the maze is solvable
    for _ in range(max(width, height)//5):
        x, y = random.randint(1, width-2), random.randint(1, height-2)
        path_length = random.randint(10, max(width, height)//2)
        
        for _ in range(path_length):
            # Clear the current cell
            grid[y, x] = 0
            # Move in a random direction
            dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
            x, y = x + dx, y + dy
            
            # Keep within bounds
            x = max(1, min(width-2, x))
            y = max(1, min(height-2, y))
    
    # Select start and goal points from opposite corners of the grid
    quadrants = [
        (1, height//4, 1, width//4),  # Top-left
        (1, height//4, width*3//4, width-2),  # Top-right
        (height*3//4, height-2, 1, width//4),  # Bottom-left
        (height*3//4, height-2, width*3//4, width-2)  # Bottom-right
    ]
    
    # Pick two different quadrants for start and goal
    start_quadrant, goal_quadrant = random.sample(quadrants, 2)
    
    # Find open spaces in each quadrant
    def find_open_point(y_min, y_max, x_min, x_max):
        candidates = []
        for i in range(y_min, y_max):
            for j in range(x_min, x_max):
                if grid[i, j] == 0:
                    candidates.append((j, i))  # Note: (x,y) format for A*
        
        return random.choice(candidates) if candidates else None
    
    start = find_open_point(*start_quadrant)
    goal = find_open_point(*goal_quadrant)
    
    # If we couldn't find points in the chosen quadrants, create some
    if start is None:
        x, y = random.randint(1, width//3), random.randint(1, height//3)
        grid[y, x] = 0
        start = (x, y)
        
    if goal is None:
        x, y = random.randint(2*width//3, width-2), random.randint(2*height//3, height-2)
        grid[y, x] = 0
        goal = (x, y)
    
    # Convert grid to horizontal and vertical barriers format
    horizontal_barriers, vertical_barriers = grid_to_barriers(grid)
    
    # Create environment data
    env_data = {
        "id": random.randint(1000, 9999),  # Use a random ID for the new environment
        "range_x": [0, width],
        "range_y": [0, height],
        "horizontal_barriers": horizontal_barriers,
        "vertical_barriers": vertical_barriers,
        "start_goal": [[start, goal]]  # Format compatible with existing dataset
    }
    
    # Save as visualization
    save_gridmap_visualization(grid, start, goal, f"dataset/complex_grid_{env_data['id']}.png")
    
    return env_data, grid

def grid_to_barriers(grid):
    """Convert a grid representation to horizontal and vertical barriers."""
    height, width = grid.shape
    horizontal_barriers = []
    vertical_barriers = []
    
    # Find horizontal barriers (rows of obstacles)
    for i in range(height):
        j = 0
        while j < width:
            # Skip non-obstacles
            if grid[i, j] == 0:
                j += 1
                continue
            
            # Found an obstacle, look for adjacent obstacles
            start = j
            while j < width and grid[i, j] == 1:
                j += 1
            
            # If more than one adjacent obstacle, add as a barrier
            if j - start > 1:
                horizontal_barriers.append([i, start, j-1])
            elif j - start == 1:
                # Single obstacle, add as a horizontal barrier
                horizontal_barriers.append([i, start, start])
    
    # Find vertical barriers (columns of obstacles)
    for j in range(width):
        i = 0
        while i < height:
            # Skip non-obstacles
            if grid[i, j] == 0:
                i += 1
                continue
            
            # Found an obstacle, look for adjacent obstacles
            start = i
            while i < height and grid[i, j] == 1:
                i += 1
            
            # If more than one adjacent obstacle, add as a barrier
            if i - start > 1:
                vertical_barriers.append([j, start, i-1])
            elif i - start == 1:
                # Single obstacle, add as a vertical barrier
                vertical_barriers.append([j, start, start])
    
    return horizontal_barriers, vertical_barriers

def save_gridmap_visualization(grid, start, goal, filepath):
    """Save a visualization of the grid with start and goal points."""
    height, width = grid.shape
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()  # Invert Y-axis to match grid coordinates
    
    # Plot the grid
    for i in range(height):
        for j in range(width):
            if grid[i, j] == 1:  # Obstacle
                rect = plt.Rectangle((j, i), 1, 1, facecolor='black', edgecolor='black')
                ax.add_patch(rect)
            else:  # Open space
                rect = plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='lightgray')
                ax.add_patch(rect)
    
    # Mark start and goal positions
    start_x, start_y = start
    goal_x, goal_y = goal
    
    start_rect = plt.Rectangle((start_x, start_y), 1, 1, facecolor='green', edgecolor='black')
    goal_rect = plt.Rectangle((goal_x, goal_y), 1, 1, facecolor='red', edgecolor='black')
    ax.add_patch(start_rect)
    ax.add_patch(goal_rect)
    
    # Save the figure
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate complex grid maps for A* testing')
    parser.add_argument('--width', type=int, default=50, help='Width of the grid')
    parser.add_argument('--height', type=int, default=30, help='Height of the grid')
    parser.add_argument('--density', type=float, default=0.3, help='Obstacle density (0.0-1.0)')
    parser.add_argument('--count', type=int, default=1, help='Number of grid maps to generate')
    
    args = parser.parse_args()
    
    # Create dataset directories if they don't exist
    os.makedirs('dataset', exist_ok=True)
    os.makedirs('dataset/A*', exist_ok=True)
    
    # Load existing environments if available
    dataset_path = 'dataset/environment_50_30.json'
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r') as f:
            environments = json.load(f)
    else:
        environments = []
    
    print(f"Generating {args.count} complex grid maps...")
    
    for i in range(args.count):
        # Generate complex grid map
        print(f"Generating grid map {i+1}/{args.count}...")
        env_data, grid = generate_complex_gridmap(
            width=args.width,
            height=args.height,
            obstacle_density=args.density
        )
        
        # Add to environments list
        environments.append(env_data)
        
        print(f"Grid map {i+1} generated:")
        print(f"  - Size: {args.width}x{args.height}")
        print(f"  - ID: {env_data['id']}")
        print(f"  - Start: {env_data['start_goal'][0][0]}")
        print(f"  - Goal: {env_data['start_goal'][0][1]}")
        print(f"  - Number of obstacles: {sum(sum(row) for row in grid)}")
        print("")
    
    # Save the updated environments
    with open(dataset_path, 'w') as f:
        json.dump(environments, f, indent=4)
    
    print(f"Done! All grid maps saved to {dataset_path}")
    print(f"Last generated grid ID: {environments[-1]['id']}")

if __name__ == "__main__":
    main() 