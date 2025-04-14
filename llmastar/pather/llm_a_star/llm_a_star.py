import json
import math
import heapq

from llmastar.env.search import env, plotting
from llmastar.model import ChatGPT, Llama3
from llmastar.utils import is_lines_collision, list_parse
from .prompt import *

# Node class for the improved bidirectional A* algorithm
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # The cost from the starting point to the current node
        self.h = 0  # The heuristic estimated cost from the current node to the target node
        self.f = 0  # Total cost g + h

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)

class LLMAStar:
    """LLM-A* algorithm with cost + heuristics as the priority."""
    
    GPT_METHOD = "PARSE"
    GPT_LLMASTAR_METHOD = "LLM-A*"

    def __init__(self, llm='gpt', prompt='standard', use_improved_astar=False):
        self.llm = llm
        if self.llm == 'gpt':
            self.parser = ChatGPT(method=self.GPT_METHOD, sysprompt=sysprompt_parse, example=example_parse)
            self.model = ChatGPT(method=self.GPT_LLMASTAR_METHOD, sysprompt="", example=None)
        elif self.llm == 'llama':
            self.model = Llama3()
        else:
            raise ValueError("Invalid LLM model. Choose 'gpt' or 'llama'.")
        
        assert prompt in ['standard', 'cot', 'repe', 'react', 'step_back', 'tot'], "Invalid prompt type. Choose 'standard', 'cot', 'repe', 'react', 'step_back', or 'tot'."
        self.prompt = prompt
        self.use_improved_astar = use_improved_astar

    def search(self, query, filepath='temp.png'):
        """
        Main search method that uses either original A* or improved A* depending on the configuration.
        :param query: The search query containing environment information
        :param filepath: Path to save the visualization
        :return: Path and search metrics
        """
        if self.use_improved_astar:
            return self.searching_improved(query, filepath)
        else:
            return self.searching(query, filepath)

    def _parse_query(self, query):
        """Parse input query using the specified LLM model."""
        if isinstance(query, str):
            if self.llm == 'gpt':
                response = self.parser.chat(query)
                print(response)
                return json.loads(response)
            elif self.llm == 'llama':
                response = self.model.ask(parse_llama.format(query=query))
                print(response)
                return json.loads(response)
            else:
                raise ValueError("Invalid LLM model.")
        return query

    def _initialize_parameters(self, input_data):
        """Initialize environment parameters from input data."""
        self.s_start = tuple(input_data['start'])
        self.s_goal = tuple(input_data['goal'])
        self.horizontal_barriers = input_data['horizontal_barriers']
        self.vertical_barriers = input_data['vertical_barriers']
        self.range_x = input_data['range_x']
        self.range_y = input_data['range_y']
        self.Env = env.Env(self.range_x[1], self.range_y[1], self.horizontal_barriers, self.vertical_barriers)
        self.plot = plotting.Plotting(self.s_start, self.s_goal, self.Env)
        # Adjust range limits
        self.range_x[1] -= 1
        self.range_y[1] -= 1
        self.u_set = self.Env.motions
        self.obs = self.Env.obs
        self.OPEN = []
        self.CLOSED = []
        self.PARENT = dict()
        self.g = dict()

    def _initialize_llm_paths(self):
        """Initialize paths using LLM suggestions."""
        start, goal = list(self.s_start), list(self.s_goal)
        query = self._generate_llm_query(start, goal)

        if self.llm == 'gpt':
            response = self.model.ask(prompt=query, max_tokens=1000)
        elif self.llm == 'llama':
            response = self.model.ask(prompt=query)
        else:
            raise ValueError("Invalid LLM model.")

        nodes = list_parse(response)
        self.target_list = self._filter_valid_nodes(nodes)

        if not self.target_list or self.target_list[0] != self.s_start:
            self.target_list.insert(0, self.s_start)
        if not self.target_list or self.target_list[-1] != self.s_goal:
            self.target_list.append(self.s_goal)
        print(self.target_list)
        self.i = 1
        self.s_target = self.target_list[1]
        print(self.target_list[0], self.s_target)

    def _generate_llm_query(self, start, goal):
        """Generate the query for the LLM."""
        if self.llm == 'gpt':
            return gpt_prompt[self.prompt].format(start=start, goal=goal,
                                horizontal_barriers=self.horizontal_barriers,
                                vertical_barriers=self.vertical_barriers)
        elif self.llm == 'llama':
            return llama_prompt[self.prompt].format(start=start, goal=goal,
                                    horizontal_barriers=self.horizontal_barriers,
                                    vertical_barriers=self.vertical_barriers)

    def _filter_valid_nodes(self, nodes):
        """Filter out invalid nodes based on environment constraints."""
        return [(node[0], node[1]) for node in nodes
                if (node[0], node[1]) not in self.obs
                and self.range_x[0] + 1 < node[0] < self.range_x[1] - 1
                and self.range_y[0] + 1 < node[1] < self.range_y[1] - 1]

    def searching(self, query, filepath='temp.png'):
        """
        A* searching algorithm.
        :return: Path and search metrics.
        """
        self.filepath = filepath
        print(query)
        input_data = self._parse_query(query)
        self._initialize_parameters(input_data)
        self._initialize_llm_paths()
        
        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        heapq.heappush(self.OPEN, (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  # stop condition
                break

            for s_n in self.get_neighbor(s):
                if s_n == self.s_target and self.s_goal != self.s_target :
                    self._update_target()
                    self._update_queue()
                    print(s_n, self.s_target)
                    
                if s_n in self.CLOSED:
                    continue

                new_cost = self.g[s] + self.cost(s, s_n)
                if s_n not in self.g:
                    self.g[s_n] = math.inf
                    
                if new_cost < self.g[s_n]:  # conditions for updating Cost
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        path = self.extract_path(self.PARENT)
        visited = self.CLOSED
        result = {
            "operation": len(self.CLOSED),
            "storage": len(self.g),
            "length": sum(self._euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1)),
            "llm_output": self.target_list
        }
        print(result)
        self.plot.animation(path, visited, True, "LLM-A*", self.filepath)
        return result

    def searching_improved(self, query, filepath='temp.png'):
        """
        Improved bidirectional A* searching algorithm following the implementation in A_star_improved2.py.
        :return: Path and search metrics.
        """
        self.filepath = filepath
        print(query)
        input_data = self._parse_query(query)
        self._initialize_parameters(input_data)
        self._initialize_llm_paths()
        
        # Create a grid representation for the improved A* algorithm
        grid_size_x = self.range_x[1] + 2
        grid_size_y = self.range_y[1] + 2
        grid = [[0 for _ in range(grid_size_y)] for _ in range(grid_size_x)]
        
        # Mark obstacles in the grid
        for pos in self.obs:
            if 0 <= pos[0] < grid_size_x and 0 <= pos[1] < grid_size_y:
                grid[pos[0]][pos[1]] = 1
        
        # Debug output of grid and obstacles
        print(f"Grid size: {grid_size_x}x{grid_size_y}")
        print(f"Start: {self.s_start}, Goal: {self.s_goal}")
        print(f"Number of obstacles: {len(self.obs)}")
        
        # Initialize bidirectional A* components
        open_list_start = []  # Priority queue in the direction of the starting point
        open_list_goal = []   # Priority queue in the direction of the endpoint
        closed_list_start_positions = set()  # Set of positions (not nodes) that have been explored from start
        closed_list_goal_positions = set()   # Set of positions (not nodes) that have been explored from goal
        closed_list_start = {}  # Dictionary mapping positions to nodes from start direction
        closed_list_goal = {}   # Dictionary mapping positions to nodes from goal direction
        
        start_node = Node(self.s_start)
        goal_node = Node(self.s_goal)
        
        # Calculate initial heuristics
        start_node.h = self._manhattan_distance(self.s_start, self.s_goal)
        start_node.f = start_node.g + start_node.h
        goal_node.h = self._manhattan_distance(self.s_goal, self.s_start)
        goal_node.f = goal_node.g + goal_node.h
        
        # Add start and goal nodes to the open lists
        heapq.heappush(open_list_start, start_node)
        heapq.heappush(open_list_goal, goal_node)
        
        # Track explored paths
        search_path_start = []
        search_path_goal = []
        
        # Helper function to build a path from a node to its origin
        def build_path(node, reverse=False):
            path = []
            current = node
            while current:
                path.append(current.position)
                current = current.parent
            
            if reverse:
                return path  # Path is already from goal to start
            else:
                return path[::-1]  # Reverse to get from start to goal
                
        # Function to check if the bidirectional search has found a meeting point
        def find_meeting_point():
            # Check for positions that have been explored from both directions
            common_positions = closed_list_start_positions.intersection(closed_list_goal_positions)
            
            if common_positions:
                # Found at least one common position, find the best one based on path length
                best_meet_pos = None
                best_length = float('inf')
                
                for pos in common_positions:
                    # Get nodes from both directions
                    start_node = closed_list_start[pos]
                    goal_node = closed_list_goal[pos]
                    
                    # Calculate total path length
                    path_length = start_node.g + goal_node.g
                    
                    if path_length < best_length:
                        best_length = path_length
                        best_meet_pos = pos
                
                if best_meet_pos:
                    # Build paths from both directions
                    path_from_start = build_path(closed_list_start[best_meet_pos], reverse=False)
                    path_from_goal = build_path(closed_list_goal[best_meet_pos], reverse=True)
                    
                    # Combine paths, avoiding duplicate meeting point
                    return path_from_start + path_from_goal[1:]
            
            # No meeting point found
            return None
        
        # Helper function to get neighbors with grid bounds and obstacle checking
        def get_valid_neighbors(pos):
            neighbors = []
            
            # 8-direction adjacency
            for dx, dy in self.u_set:
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < grid_size_x and 0 <= ny < grid_size_y:
                    # Skip obstacles
                    if (nx, ny) in self.obs:
                        continue
                    
                    # Skip if there's a collision with a barrier
                    if self.is_collision(pos, (nx, ny)):
                        continue
                        
                    neighbors.append((nx, ny))
            
            return neighbors
        
        # Maximum iterations to prevent infinite loops
        max_iterations = 2000
        iteration_count = 0
        
        # Main bidirectional search loop
        while open_list_start and open_list_goal and iteration_count < max_iterations:
            iteration_count += 1
            
            # Process from start direction
            if open_list_start:
                current_node_start = heapq.heappop(open_list_start)
                current_pos_start = current_node_start.position
                
                # Skip if already processed
                if current_pos_start in closed_list_start_positions:
                    continue
                
                # Add to closed lists
                closed_list_start_positions.add(current_pos_start)
                closed_list_start[current_pos_start] = current_node_start
                search_path_start.append(current_pos_start)
                
                # Check if we've reached the goal directly
                if current_pos_start == self.s_goal:
                    # Direct path from start to goal found
                    path = build_path(current_node_start, reverse=False)
                    result = {
                        "operation": len(closed_list_start) + len(closed_list_goal),
                        "storage": len(closed_list_start) + len(closed_list_goal),
                        "length": sum(self._euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1)),
                        "llm_output": self.target_list
                    }
                    print("Path found directly from start to goal!")
                    print(result)
                    visited = list(closed_list_start.keys()) + list(closed_list_goal.keys())
                    self.plot.animation(path, visited, True, "LLM-A* Improved", self.filepath)
                    return result
                    
                # Check if this node is also in the closed list from the goal direction
                # This means the two searches have met
                if current_pos_start in closed_list_goal_positions:
                    # Bidirectional search has met
                    merged_path = find_meeting_point()
                    result = {
                        "operation": len(closed_list_start) + len(closed_list_goal),
                        "storage": len(closed_list_start) + len(closed_list_goal),
                        "length": sum(self._euclidean_distance(merged_path[i], merged_path[i+1]) for i in range(len(merged_path)-1)),
                        "llm_output": self.target_list
                    }
                    print("Path found through bidirectional search meeting!")
                    print(result)
                    visited = list(closed_list_start.keys()) + list(closed_list_goal.keys())
                    self.plot.animation(merged_path, visited, True, "LLM-A* Improved", self.filepath)
                    return result
                
                # Expand neighbors from start direction
                for neighbor_pos in get_valid_neighbors(current_pos_start):
                    # Skip if already processed
                    if neighbor_pos in closed_list_start_positions:
                        continue
                    
                    # Create neighbor node
                    neighbor = Node(neighbor_pos, current_node_start)
                    
                    # Calculate costs
                    move_cost = self._euclidean_distance(current_pos_start, neighbor_pos)
                    neighbor.g = current_node_start.g + move_cost
                    neighbor.h = self._manhattan_distance(neighbor_pos, self.s_goal)
                    neighbor.f = neighbor.g + neighbor.h
                    
                    # Check if already in open list with better cost
                    should_add = True
                    for i, open_node in enumerate(open_list_start):
                        if open_node.position == neighbor_pos:
                            if neighbor.f < open_node.f:
                                # Better path found, replace
                                open_list_start[i] = neighbor
                                heapq.heapify(open_list_start)
                            should_add = False
                            break
                    
                    # Add to open list if not already there or if better path
                    if should_add:
                        heapq.heappush(open_list_start, neighbor)
            
            # Process from goal direction
            if open_list_goal:
                current_node_goal = heapq.heappop(open_list_goal)
                current_pos_goal = current_node_goal.position
                
                # Skip if already processed
                if current_pos_goal in closed_list_goal_positions:
                    continue
                
                # Add to closed lists
                closed_list_goal_positions.add(current_pos_goal)
                closed_list_goal[current_pos_goal] = current_node_goal
                search_path_goal.append(current_pos_goal)
                
                # Check if we've reached the start directly
                if current_pos_goal == self.s_start:
                    # Direct path from goal to start found
                    path = build_path(current_node_goal, reverse=True)
                    result = {
                        "operation": len(closed_list_start) + len(closed_list_goal),
                        "storage": len(closed_list_start) + len(closed_list_goal),
                        "length": sum(self._euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1)),
                        "llm_output": self.target_list
                    }
                    print("Path found directly from goal to start!")
                    print(result)
                    visited = list(closed_list_start.keys()) + list(closed_list_goal.keys())
                    self.plot.animation(path, visited, True, "LLM-A* Improved", self.filepath)
                    return result
                
                # Check if this node is also in the closed list from the start direction
                # This means the two searches have met
                if current_pos_goal in closed_list_start_positions:
                    # Bidirectional search has met
                    merged_path = find_meeting_point()
                    result = {
                        "operation": len(closed_list_start) + len(closed_list_goal),
                        "storage": len(closed_list_start) + len(closed_list_goal),
                        "length": sum(self._euclidean_distance(merged_path[i], merged_path[i+1]) for i in range(len(merged_path)-1)),
                        "llm_output": self.target_list
                    }
                    print("Path found through bidirectional search meeting!")
                    print(result)
                    visited = list(closed_list_start.keys()) + list(closed_list_goal.keys())
                    self.plot.animation(merged_path, visited, True, "LLM-A* Improved", self.filepath)
                    return result
                
                # Expand neighbors from goal direction
                for neighbor_pos in get_valid_neighbors(current_pos_goal):
                    # Skip if already processed
                    if neighbor_pos in closed_list_goal_positions:
                        continue
                    
                    # Create neighbor node
                    neighbor = Node(neighbor_pos, current_node_goal)
                    
                    # Calculate costs
                    move_cost = self._euclidean_distance(current_pos_goal, neighbor_pos)
                    neighbor.g = current_node_goal.g + move_cost
                    neighbor.h = self._manhattan_distance(neighbor_pos, self.s_start)
                    neighbor.f = neighbor.g + neighbor.h
                    
                    # Check if already in open list with better cost
                    should_add = True
                    for i, open_node in enumerate(open_list_goal):
                        if open_node.position == neighbor_pos:
                            if neighbor.f < open_node.f:
                                # Better path found, replace
                                open_list_goal[i] = neighbor
                                heapq.heapify(open_list_goal)
                            should_add = False
                            break
                    
                    # Add to open list if not already there or if better path
                    if should_add:
                        heapq.heappush(open_list_goal, neighbor)
            
            # Check for meeting point at the end of each iteration
            if iteration_count % 100 == 0:
                merged_path = find_meeting_point()
                if merged_path:
                    # Found a meeting point
                    result = {
                        "operation": len(closed_list_start) + len(closed_list_goal),
                        "storage": len(closed_list_start) + len(closed_list_goal),
                        "length": sum(self._euclidean_distance(merged_path[i], merged_path[i+1]) for i in range(len(merged_path)-1)),
                        "llm_output": self.target_list
                    }
                    print(f"Path found through bidirectional search at iteration {iteration_count}!")
                    print(result)
                    visited = list(closed_list_start.keys()) + list(closed_list_goal.keys())
                    self.plot.animation(merged_path, visited, True, "LLM-A* Improved", self.filepath)
                    return result
                
        # If bidirectional search failed, try regular A*
        print("Bidirectional search exhausted after", iteration_count, "iterations")
        print("Closed list sizes - Start:", len(closed_list_start), "Goal:", len(closed_list_goal))
        print("Fallback to regular A* search...")
        
        # Reset for regular A* search
        self.PARENT = {self.s_start: self.s_start}
        self.g = {self.s_start: 0}
        self.OPEN = []
        self.CLOSED = []
        
        # Add start node to open list with f-score as priority
        heapq.heappush(self.OPEN, (self._manhattan_distance(self.s_start, self.s_goal), self.s_start))
        
        # Regular A* search
        a_star_iterations = 0
        while self.OPEN and a_star_iterations < 10000:  # Added iteration limit
            a_star_iterations += 1
            
            # Get node with lowest f-score
            _, current = heapq.heappop(self.OPEN)
            
            # Add to closed list
            self.CLOSED.append(current)
            
            # Check if we've reached the goal
            if current == self.s_goal:
                # Extract path
                path = self.extract_path(self.PARENT)
                
                # Report success
                result = {
                    "operation": len(self.CLOSED) + len(closed_list_start) + len(closed_list_goal),
                    "storage": len(self.g) + len(closed_list_start) + len(closed_list_goal),
                    "length": sum(self._euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1)),
                    "llm_output": self.target_list
                }
                print("Path found with regular A* fallback!")
                print(result)
                visited = list(closed_list_start.keys()) + list(closed_list_goal.keys()) + self.CLOSED
                self.plot.animation(path, visited, True, "LLM-A* Improved (A* Fallback)", self.filepath)
                return result
            
            # Expand neighbors
            for neighbor in get_valid_neighbors(current):
                # Skip if already processed
                if neighbor in self.CLOSED:
                    continue
                
                # Calculate new cost
                new_cost = self.g[current] + self._euclidean_distance(current, neighbor)
                
                # Initialize cost if not seen before
                if neighbor not in self.g:
                    self.g[neighbor] = float('inf')
                
                # Update cost if better path found
                if new_cost < self.g[neighbor]:
                    self.g[neighbor] = new_cost
                    self.PARENT[neighbor] = current
                    f_value = new_cost + self._manhattan_distance(neighbor, self.s_goal)
                    heapq.heappush(self.OPEN, (f_value, neighbor))
        
        # If still no path found
        print("No path found after exhaustive search:", a_star_iterations, "A* iterations")
        result = {
            "operation": len(closed_list_start) + len(closed_list_goal) + len(self.CLOSED),
            "storage": len(closed_list_start) + len(closed_list_goal) + len(self.g),
            "length": 0,
            "llm_output": self.target_list
        }
        print(result)
        
        # Visualize the search process even if no path is found
        visited = list(closed_list_start.keys()) + list(closed_list_goal.keys()) + self.CLOSED
        self.plot.animation([], visited, True, "LLM-A* Improved (No Path)", self.filepath)
        
        return result

    @staticmethod
    def _euclidean_distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    @staticmethod
    def _manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _update_queue(self):
        queue = []
        for _, s in self.OPEN:
            heapq.heappush(queue, (self.f_value(s), s))
        self.OPEN = queue

    def _update_target(self):
        """Update the current target in the path."""
        self.i += 1
        if self.i < len(self.target_list):
            self.s_target = self.target_list[self.i]

    def get_neighbor(self, s):
        """Find neighbors of state s that are not in obstacles."""
        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        """Calculate cost for the motion from s_start to s_goal."""
        return math.inf if self.is_collision(s_start, s_goal) else math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        """Check if the line segment (s_start, s_end) collides with any barriers."""
        line1 = [s_start, s_end]
        return any(is_lines_collision(line1, [[h[1], h[0]], [h[2], h[0]]]) for h in self.horizontal_barriers) or \
                any(is_lines_collision(line1, [[v[0], v[1]], [v[0], v[2]]]) for v in self.vertical_barriers) or \
                any(is_lines_collision(line1, [[x, self.range_y[0]], [x, self.range_y[1]]]) for x in self.range_x) or \
                any(is_lines_collision(line1, [[self.range_x[0], y], [self.range_x[1], y]]) for y in self.range_y)

    def f_value(self, s):
        """Compute the f-value for state s."""
        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
        """Extract the path based on the PARENT set."""
        path = [self.s_goal]
        while path[-1] != self.s_start:
            path.append(PARENT[path[-1]])
        return path[::-1]

    def heuristic(self, s):
        """Calculate heuristic value."""
        return math.hypot(self.s_goal[0] - s[0], self.s_goal[1] - s[1]) + math.hypot(self.s_target[0] - s[0], self.s_target[1] - s[1])

