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
        
        # Initialize bidirectional A* components exactly as in A_star_improved2.py
        open_list_start = []  # Priority queue in the direction of the starting point
        open_list_goal = []   # Priority queue in the direction of the endpoint
        closed_list_start = set()  # The explored node in the direction of the starting point
        closed_list_goal = set()   # The explored node in the end direction
        start_node = Node(self.s_start)
        goal_node = Node(self.s_goal)
        
        # Use the manhattan distance heuristic as in A_star_improved2.py
        start_node.h = self._manhattan_distance(self.s_start, self.s_goal)
        start_node.f = start_node.g + start_node.h
        goal_node.h = self._manhattan_distance(self.s_goal, self.s_start)
        goal_node.f = goal_node.g + goal_node.h
        
        heapq.heappush(open_list_start, start_node)
        heapq.heappush(open_list_goal, goal_node)
        
        search_path_start = []  # The search path in the direction of the starting point
        search_path_goal = []   # The search path in the direction of the end point
        
        # Helper functions from A_star_improved2.py
        def build_path(meet_node, start_direction=True):
            path = []
            current = meet_node
            while current:
                path.append(current.position)
                current = current.parent
            if not start_direction:
                path = path[::-1]
            return path
        
        def meet_in_the_middle():
            meet_node = None
            goal_meet_node = None
            start_meet_node = None
            for node in closed_list_start:
                if node in closed_list_goal:
                    meet_node = node
                    break
            if meet_node:
                for node in closed_list_start:
                    if node == meet_node:
                        start_meet_node = node
                        break
                for node in closed_list_goal:
                    if node == meet_node:
                        goal_meet_node = node
                        break
            if meet_node:
                path_start = build_path(start_meet_node)
                path_goal = build_path(goal_meet_node, start_direction=False)
                # Combine the paths, removing duplicates at meeting point
                return path_start[:-1] + path_goal
            return None
        
        # Helper functions for neighbor selection from A_star_improved2.py
        def get_angle(start, end):
            # Calculate the angle from the start point to the end point
            delta_y = -(end[0] - start[0])
            delta_x = end[1] - start[1]
            
            angle = math.degrees(math.atan2(delta_y, delta_x))
            if angle < 0:
                angle += 360
            return angle
        
        def filter_neighbors(neighbors, angle, mode=8):
            # Filter neighbor nodes based on direction and mode (8 adjacency).
            direction_map = {
                (22.5, 67.5): ([0, 1, 2, 6, 7], [3, 4, 5]),
                (67.5, 112.5): ([0, 1, 2, 3, 7], [4, 5, 6]),
                (112.5, 157.5): ([0, 1, 2, 3, 4], [5, 6, 7]),
                (157.5, 202.5): ([1, 2, 3, 4, 5], [0, 6, 7]),
                (202.5, 247.5): ([2, 3, 4, 5, 6], [0, 1, 7]),
                (247.5, 292.5): ([3, 4, 5, 6, 7], [0, 1, 2]),
                (292.5, 337.5): ([0, 4, 5, 6, 7], [1, 2, 3]),
                (337.5, 360): ([0, 1, 5, 6, 7], [2, 3, 4]),
                (0, 22.5): ([0, 1, 5, 6, 7], [2, 3, 4])
            }
            
            for (low, high), (keep, _) in direction_map.items():
                if low <= angle <= high:
                    filtered_neighbors = [neighbors[i] for i in keep if i < len(neighbors)]
                    return filtered_neighbors
            
            return neighbors
        
        def get_neighbors(node, target_pos, allow_diagonal=True):
            """Get neighbor nodes with directional filtering as in A_star_improved2.py"""
            # 8-direction adjacency
            directions = [
                (-1, 1), (-1, 0), (-1, -1),
                (0, -1), (1, -1), (1, 0),
                (1, 1), (0, 1)
            ]
            
            if not allow_diagonal:
                # 4-direction adjacency if diagonals not allowed
                directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
            
            neighbors = []
            for dx, dy in directions:
                x, y = node.position[0] + dx, node.position[1] + dy
                # Check if the position is within grid bounds and not an obstacle
                if 0 <= x < grid_size_x and 0 <= y < grid_size_y and grid[x][y] == 0:
                    new_node = Node((x, y), node)
                    neighbors.append(new_node)
            
            # Apply directional filtering based on angle to target
            angle = get_angle(node.position, target_pos)
            filtered_neighbors = filter_neighbors(neighbors, angle)
            
            return filtered_neighbors
        
        # Main loop similar to A_star_improved2.py
        while open_list_start and open_list_goal:
            # Process from start direction
            current_node_start = heapq.heappop(open_list_start)
            closed_list_start.add(current_node_start)
            search_path_start.append(current_node_start.position)
            
            # Process from goal direction
            current_node_goal = heapq.heappop(open_list_goal)
            closed_list_goal.add(current_node_goal)
            search_path_goal.append(current_node_goal.position)
            
            # Check termination conditions
            if (current_node_start.position == self.s_goal or 
                current_node_goal.position == self.s_start or 
                current_node_start in closed_list_goal):
                
                # Check if we've met in the middle
                merged_path = meet_in_the_middle()
                if merged_path:
                    # Return metrics
                    result = {
                        "operation": len(closed_list_start) + len(closed_list_goal),
                        "storage": len(closed_list_start) + len(closed_list_goal),
                        "length": sum(self._euclidean_distance(merged_path[i], merged_path[i+1]) for i in range(len(merged_path)-1)),
                        "llm_output": self.target_list
                    }
                    print(result)
                    
                    # Visualize the path and search process
                    visited = [node.position for node in closed_list_start] + [node.position for node in closed_list_goal]
                    self.plot.animation(merged_path, visited, True, "LLM-A* Improved", self.filepath)
                    
                    return result
            
            # Process neighbors from start direction
            start_neighbors = get_neighbors(current_node_start, self.s_goal)
            for neighbor in start_neighbors:
                if any(existing_node.position == neighbor.position for existing_node in closed_list_start):
                    continue
                
                # Compute costs
                move_cost = self._euclidean_distance(current_node_start.position, neighbor.position)
                tentative_g = current_node_start.g + move_cost
                
                # Use Manhattan distance to goal as heuristic
                neighbor.g = tentative_g
                neighbor.h = self._manhattan_distance(neighbor.position, self.s_goal)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current_node_start
                
                # Check if this is a better path if the node is already in open list
                add_to_open = True
                for i, open_node in enumerate(open_list_start):
                    if open_node.position == neighbor.position:
                        if neighbor.f < open_node.f:
                            # Replace with better path
                            open_list_start[i] = neighbor
                            # Reorder the heap
                            heapq.heapify(open_list_start)
                        add_to_open = False
                        break
                
                if add_to_open:
                    heapq.heappush(open_list_start, neighbor)
                    search_path_start.append(neighbor.position)
            
            # Process neighbors from goal direction
            goal_neighbors = get_neighbors(current_node_goal, self.s_start)
            for neighbor in goal_neighbors:
                if any(existing_node.position == neighbor.position for existing_node in closed_list_goal):
                    continue
                
                # Compute costs
                move_cost = self._euclidean_distance(current_node_goal.position, neighbor.position)
                tentative_g = current_node_goal.g + move_cost
                
                # Use Manhattan distance to start as heuristic
                neighbor.g = tentative_g
                neighbor.h = self._manhattan_distance(neighbor.position, self.s_start)
                neighbor.f = neighbor.g + neighbor.h
                neighbor.parent = current_node_goal
                
                # Check if this is a better path if the node is already in open list
                add_to_open = True
                for i, open_node in enumerate(open_list_goal):
                    if open_node.position == neighbor.position:
                        if neighbor.f < open_node.f:
                            # Replace with better path
                            open_list_goal[i] = neighbor
                            # Reorder the heap
                            heapq.heapify(open_list_goal)
                        add_to_open = False
                        break
                
                if add_to_open:
                    heapq.heappush(open_list_goal, neighbor)
                    search_path_goal.append(neighbor.position)
        
        # If no path found, try the optimization step from A_star_improved2.py
        def optimize_path(path):
            """Optimize the path by removing unnecessary waypoints"""
            if not path or len(path) <= 2:
                return path
                
            optimized_path = [path[0]]
            current_idx = 0
            
            while current_idx < len(path) - 1:
                # Try to find the furthest point we can directly reach
                furthest_idx = current_idx + 1
                for i in range(current_idx + 2, len(path)):
                    if not self.is_collision(path[current_idx], path[i]):
                        furthest_idx = i
                
                optimized_path.append(path[furthest_idx])
                current_idx = furthest_idx
                
            return optimized_path
        
        # Fallback to regular A* if bidirectional search fails
        print("Bidirectional search failed, trying regular A*...")
        # Reset search parameters
        self.PARENT = {self.s_start: self.s_start}
        self.g = {self.s_start: 0}
        self.OPEN = [(self._manhattan_distance(self.s_start, self.s_goal), self.s_start)]
        self.CLOSED = []
        
        # Regular A* search
        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)
            
            if s == self.s_goal:
                path = self.extract_path(self.PARENT)
                # Apply path optimization
                optimized_path = optimize_path(path)
                result = {
                    "operation": len(self.CLOSED) + len(closed_list_start) + len(closed_list_goal),
                    "storage": len(self.g) + len(closed_list_start) + len(closed_list_goal),
                    "length": sum(self._euclidean_distance(optimized_path[i], optimized_path[i+1]) for i in range(len(optimized_path)-1)),
                    "llm_output": self.target_list
                }
                print("Found path using fallback A* with optimization")
                print(result)
                visited = [node.position for node in closed_list_start] + [node.position for node in closed_list_goal] + self.CLOSED
                self.plot.animation(optimized_path, visited, True, "LLM-A* Improved (Fallback)", self.filepath)
                return result
            
            for s_n in self.get_neighbor(s):
                if s_n in self.CLOSED or s_n in self.obs:
                    continue
                
                new_cost = self.g[s] + self.cost(s, s_n)
                if s_n not in self.g:
                    self.g[s_n] = float('inf')
                
                if new_cost < self.g[s_n]:
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    f_value = new_cost + self._manhattan_distance(s_n, self.s_goal)
                    heapq.heappush(self.OPEN, (f_value, s_n))
        
        # If still no path found
        print("No path found after exhaustive search")
        result = {
            "operation": len(closed_list_start) + len(closed_list_goal) + len(self.CLOSED),
            "storage": len(closed_list_start) + len(closed_list_goal) + len(self.g),
            "length": 0,
            "llm_output": self.target_list
        }
        print(result)
        
        # Visualize the search process even if no path is found
        visited = [node.position for node in closed_list_start] + [node.position for node in closed_list_goal] + self.CLOSED
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

