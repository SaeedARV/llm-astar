import random
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from llmastar.env.search import env as env_search, plotting as plotting_search
import json, os
import inquirer
import numpy as np


class Dataset:
    def __init__(self):
        self.MAP = [(50, 30)]
        self.unique_env = 100
        self.unique_sg = 10
    
    def generate_environment_Astar(self):
        for map in self.MAP:
            x_range, y_range = (0, map[0]+1), (0, map[1]+1)
            with open('dataset/A*/environment_50_30.json', 'r') as file:
                environments = json.load(file)
            
            for i in range(len(environments), self.unique_env):
                decision = False
                while not decision:
                    num_h = round(random.uniform(1, 4))
                    num_v = round(random.uniform(1, 4))
                    data = {'id': i}
                    data.update(self._generate_random_obstacles_and_points_Astar(x_range, y_range, num_h, num_v))
                    self.plot_grid_Astar(data['start_goal'][0][0], data['start_goal'][0][1], data['range_x'], data['range_y'], data['horizontal_barriers'], data['vertical_barriers'], show=False)
                    action_planner = [
                        inquirer.List(
                            'approach',
                            message=f"Choose your approach on {i}",
                            choices=[('Bad', False), ('Good', True)],
                            default=False
                        )
                    ]
                    decision = inquirer.prompt(action_planner)['approach']
                environments.append(data)
                
        with open('dataset/A*/environment_50_30.json', 'w') as f:
            json.dump(environments, f, indent=4)
        
        for i in range(len(environments)):
            data = environments[i]
            for index in range(len(data['start_goal'])): 
                sg = data['start_goal'][index]
                if not os.path.exists(f"dataset/A*/environment_{x_range[1]}_{y_range[1]}_maps/map_{i}"):
                    os.makedirs(f"dataset/A*/environment_{x_range[1]}_{y_range[1]}_maps/map_{i}")
                self.plot_grid_Astar(sg[0], sg[1], data['range_x'], data['range_y'], data['horizontal_barriers'], data['vertical_barriers'], f"A* {i}-{index}", f"dataset/A*/environment_{x_range[1]}_{y_range[1]}_maps/map_{i}/{index}.png")
    
    def generate_complex_gridmap(self, grid_size=(50, 50), obstacle_density=0.3, save_path='complex_gridmaps'):
        """
        Generate a complex grid-based environment with maze-like structures, similar to those used in improved A*.
        
        Args:
            grid_size: Tuple (width, height) of the grid
            obstacle_density: Percentage of grid cells to be filled with obstacles (0.0-1.0)
            save_path: Directory to save the generated gridmaps
            
        Returns:
            A dictionary containing the grid data
        """
        width, height = grid_size
        
        # Create directories if they don't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
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
                    
        # Create some structured maze-like patterns
        self._add_maze_patterns(grid, width, height)
        
        # Make sure there are clear paths by opening up some areas
        self._create_corridors(grid, width, height)
        
        # Select start and goal points that are far from each other
        start, goal = self._select_distant_points(grid, width, height)
        
        # Convert the grid to the format expected by our A* implementation
        horizontal_barriers, vertical_barriers = self._grid_to_barriers(grid)
        
        # Create the environment data
        env_data = {
            "grid": grid.tolist(),
            "start": start,
            "goal": goal,
            "range_x": (0, width),
            "range_y": (0, height),
            "horizontal_barriers": horizontal_barriers,
            "vertical_barriers": vertical_barriers
        }
        
        # Save the grid as a visualization
        self._save_gridmap_visualization(grid, start, goal, save_path)
        
        # Generate dataset queries
        query = f"""design a path from [{start[0]}, {start[1]}] to [{goal[0]}, {goal[1]}] on a {width} by {height} grid with complex obstacle patterns in a maze-like structure."""
        
        # Save to JSON
        json_path = os.path.join(save_path, "complex_gridmap.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []
            
        env_data["id"] = len(existing_data)
        env_data["query"] = query
        existing_data.append(env_data)
        
        with open(json_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
            
        # Also save in the format expected by the original A* implementation
        sg_list = [(start, goal)]
        a_star_data = {
            "id": len(existing_data) - 1,
            "range_x": (0, width),
            "range_y": (0, height),
            "horizontal_barriers": horizontal_barriers,
            "vertical_barriers": vertical_barriers,
            "start_goal": sg_list
        }
        
        # Visualize using the existing plotting function
        self.plot_grid_Astar(start, goal, (0, width), (0, height), horizontal_barriers, 
                             vertical_barriers, f"Complex Gridmap {a_star_data['id']}", 
                             os.path.join(save_path, f"gridmap_{a_star_data['id']}.png"), 
                             show=True)
            
        return env_data
    
    def _add_maze_patterns(self, grid, width, height):
        """Add maze-like patterns to the grid to make navigation more challenging."""
        # Add some horizontal and vertical walls with small gaps
        for i in range(3, height-3, 8):
            # Horizontal walls with gaps
            wall_length = random.randint(width//3, width-5)
            start_pos = random.randint(1, width-wall_length-1)
            grid[i, start_pos:start_pos+wall_length] = 1
            # Create 1-2 gaps in the wall
            for _ in range(random.randint(1, 2)):
                gap_pos = random.randint(start_pos, start_pos+wall_length-1)
                grid[i, gap_pos] = 0
                
        for j in range(3, width-3, 8):
            # Vertical walls with gaps
            wall_length = random.randint(height//3, height-5)
            start_pos = random.randint(1, height-wall_length-1)
            grid[start_pos:start_pos+wall_length, j] = 1
            # Create 1-2 gaps in the wall
            for _ in range(random.randint(1, 2)):
                gap_pos = random.randint(start_pos, start_pos+wall_length-1)
                grid[gap_pos, j] = 0
                
        # Add some diagonal obstacle patterns
        for _ in range(width//10):
            x, y = random.randint(5, width-6), random.randint(5, height-6)
            length = random.randint(3, 8)
            direction = random.choice([(1, 1), (1, -1), (-1, 1), (-1, -1)])
            
            for i in range(length):
                nx, ny = x + i * direction[0], y + i * direction[1]
                if 0 < nx < width-1 and 0 < ny < height-1:
                    grid[ny, nx] = 1
    
    def _create_corridors(self, grid, width, height):
        """Create some guaranteed corridors to ensure the maze is solvable."""
        # Create some random walking paths to ensure there are navigable routes
        for _ in range(max(width, height)//5):
            x, y = random.randint(1, width-2), random.randint(1, height-2)
            path_length = random.randint(10, max(width, height)//2)
            
            for _ in range(path_length):
                # Clear the current cell and its neighbors to create wider corridors
                grid[y, x] = 0
                if 0 < x-1 < width-1:
                    grid[y, x-1] = 0  # Left
                if 0 < x+1 < width-1:
                    grid[y, x+1] = 0  # Right
                if 0 < y-1 < height-1:
                    grid[y-1, x] = 0  # Up
                if 0 < y+1 < height-1:
                    grid[y+1, x] = 0  # Down
                
                # Move in a random direction
                dx, dy = random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)])
                x, y = x + dx, y + dy
                
                # Keep within bounds
                x = max(1, min(width-2, x))
                y = max(1, min(height-2, y))
    
    def _select_distant_points(self, grid, width, height):
        """Select start and goal points that are far from each other and not obstacles."""
        # Try to find points at opposite corners or sides
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
        
        # If we couldn't find points in the chosen quadrants, try random positions
        attempts = 0
        while (start is None or goal is None) and attempts < 100:
            if start is None:
                x, y = random.randint(1, width-2), random.randint(1, height-2)
                if grid[y, x] == 0:
                    start = (x, y)
            
            if goal is None:
                x, y = random.randint(1, width-2), random.randint(1, height-2)
                if grid[y, x] == 0 and (x, y) != start:
                    goal = (x, y)
                    
            attempts += 1
        
        # If still no valid points, force open some spaces
        if start is None:
            x, y = random.randint(1, width//3), random.randint(1, height//3)
            grid[y, x] = 0
            start = (x, y)
            
        if goal is None:
            x, y = random.randint(2*width//3, width-2), random.randint(2*height//3, height-2)
            grid[y, x] = 0
            goal = (x, y)
        
        return start, goal
    
    def _grid_to_barriers(self, grid):
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
                    # Single obstacle, add both horizontal and vertical barrier
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
                elif i - start == 1 and [start, j, j] not in horizontal_barriers:
                    # Single obstacle, add as vertical barrier if not already horizontal
                    vertical_barriers.append([j, start, start])
        
        return horizontal_barriers, vertical_barriers
    
    def _save_gridmap_visualization(self, grid, start, goal, save_path):
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
        
        # Add gridlines
        ax.grid(which='both', linestyle='-', linewidth=0.2)
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        
        # Save the figure
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, 'grid_visualization.png'), dpi=100, bbox_inches='tight')
        plt.close()

    def _generate_random_obstacles_and_points_Astar(self, x_range, y_range, num_h_obstacles, num_v_obstacles):
        def generate_horizontal_obstacles(num_h_obstacles, x_range, y_range, existing_obstacles):
            horizontal_obstacles = []
            for _ in range(num_h_obstacles):
                while True:
                    y = round(random.uniform(y_range[0], y_range[1]))
                    x_start = round(random.uniform(x_range[0], x_range[1]))
                    x_end = round(random.uniform(x_start, x_range[1]))
                    horizontal = LineString([(x_start, y), (x_end, y)])
                    horizontal_obstacles.append([y, x_start, x_end])
                    existing_obstacles.append(horizontal)
                    break
            return horizontal_obstacles
        
        def generate_vertical_obstacles(num_v_obstacles, x_range, y_range, existing_obstacles):
            vertical_obstacles = []
            for _ in range(num_v_obstacles):
                while True:
                    x = round(random.uniform(x_range[0], x_range[1]))
                    y_start = round(random.uniform(y_range[0], y_range[1]))
                    y_end = round(random.uniform(y_start, y_range[1]))
                    vertical = LineString([(x, y_start), (x, y_end)])
                    vertical_obstacles.append([x, y_start, y_end])
                    existing_obstacles.append(vertical)
                    break
            return vertical_obstacles
        
        def generate_random_point(x_range, y_range, existing_obstacles):
            while True:
                x = round(random.uniform(x_range[0], x_range[1] - 2))
                y = round(random.uniform(y_range[0], y_range[1] - 2))
                point = Point(x, y)
                if not any(point.intersects(ob) for ob in existing_obstacles):
                    return [x, y]
        
        existing_obstacles = []
        for x in x_range:
            existing_obstacles.append(LineString([(x, y_range[0]), (x, y_range[1])]))
        for y in y_range:
            existing_obstacles.append(LineString([(x_range[0], y), (x_range[1], y)]))
            
        horizontal_barriers = generate_horizontal_obstacles(num_h_obstacles, x_range, y_range, existing_obstacles)
        vertical_barriers = generate_vertical_obstacles(num_v_obstacles, x_range, y_range, existing_obstacles)
        
        sg_list = []
        while len(sg_list) < self.unique_sg:
            start = generate_random_point(x_range, y_range, existing_obstacles)
            goal = generate_random_point(x_range, y_range, existing_obstacles)
            if any(LineString([start, goal]).intersects(ob) for ob in existing_obstacles):
                sg_list.append((start, goal))
        
        environment = {
            "range_x": x_range,
            "range_y": y_range,
            "horizontal_barriers": horizontal_barriers,
            "vertical_barriers": vertical_barriers,
            "start_goal": sg_list
        }
        print(environment)

        return environment

    def add_query_Astar(self, filepath='dataset/A*/environment_50_30.json'):
        with open(filepath) as f:
            data = json.load(f)
        
        for environment in data:
            for sg in environment['start_goal']:
                start, goal = sg[0], sg[1]
                x_range = environment['range_x']
                y_range = environment['range_y']
                horizontal_barriers = environment['horizontal_barriers']
                vertical_barriers = environment['vertical_barriers']
                query = f"""design a path from [{start[0]}, {start[1]}] to [{goal[0]}, {goal[1]}] on a {x_range[1]} by {y_range[1]} grid that avoids horizontal barriers centered at {horizontal_barriers} and vertical barriers at {vertical_barriers}."""
                sg.append(query)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
    def plot_grid_Astar(self, s_start, s_goal, range_x, range_y, horizontal_barriers, vertical_barriers, name='A*', path="temp.png", show=False):
        Env = env_search.Env(range_x[1], range_y[1], horizontal_barriers, vertical_barriers)  # class Env
        plot = plotting_search.Plotting(s_start, s_goal, Env)
        plot.plot_map(name, path, show)