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
        Improved bidirectional A* searching algorithm.
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
        
        # Initialize bidirectional A* components
        open_list_start = []
        open_list_goal = []
        closed_list_start = set()
        closed_list_goal = set()
        start_node = Node(self.s_start)
        goal_node = Node(self.s_goal)
        
        # Set initial target from LLM suggestions
        if len(self.target_list) > 1:
            waypoint_node = Node(self.target_list[1])
            # Adjust start node's heuristic to include the waypoint
            start_node.h = self._manhattan_distance(self.s_start, self.target_list[1]) + self._manhattan_distance(self.target_list[1], self.s_goal)
            start_node.f = start_node.g + start_node.h
        else:
            waypoint_node = goal_node
            start_node.h = self._manhattan_distance(self.s_start, self.s_goal)
            start_node.f = start_node.g + start_node.h
        
        # Initialize goal node's heuristic as well
        goal_node.h = self._manhattan_distance(self.s_goal, self.s_start)
        goal_node.f = goal_node.g + goal_node.h
        
        heapq.heappush(open_list_start, start_node)
        heapq.heappush(open_list_goal, goal_node)
        
        search_path_start = []
        search_path_goal = []
        
        # Path construction functions for bidirectional search
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
            # Check if any node from start side meets any node from goal side
            for start_node in closed_list_start:
                if any(goal_node.position == start_node.position for goal_node in closed_list_goal):
                    meet_position = start_node.position
                    # Find the corresponding nodes in both directions
                    start_meet_node = next(node for node in closed_list_start if node.position == meet_position)
                    goal_meet_node = next(node for node in closed_list_goal if node.position == meet_position)
                    
                    # Build paths from both directions
                    path_start = build_path(start_meet_node)
                    path_goal = build_path(goal_meet_node, start_direction=False)
                    
                    # Remove duplicated meeting point
                    return path_start[:-1] + path_goal
            
            # If no meeting point found
            return None
        
        # Process LLM waypoints for guidance
        waypoints = self.target_list[1:-1]  # Exclude start and goal
        current_waypoint_idx = 0
        current_waypoint = waypoints[current_waypoint_idx] if waypoints else None
        
        # Counter to limit iterations and prevent infinite loops
        max_iterations = min(1000, grid_size_x * grid_size_y * 2)
        iteration_count = 0
        
        while open_list_start and open_list_goal and iteration_count < max_iterations:
            iteration_count += 1
            
            # Process from start direction
            if open_list_start:
                current_node_start = heapq.heappop(open_list_start)
                closed_list_start.add(current_node_start)
                search_path_start.append(current_node_start.position)
                
                # Check if we've reached the goal directly
                if current_node_start.position == self.s_goal:
                    path = build_path(current_node_start)
                    result = {
                        "operation": len(closed_list_start) + len(closed_list_goal),
                        "storage": len(closed_list_start) + len(closed_list_goal),
                        "length": sum(self._euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1)),
                        "llm_output": self.target_list
                    }
                    print(result)
                    visited = list(node.position for node in closed_list_start) + list(node.position for node in closed_list_goal)
                    self.plot.animation(path, visited, True, "LLM-A* Improved", self.filepath)
                    return result
            
            # Process from goal direction
            if open_list_goal:
                current_node_goal = heapq.heappop(open_list_goal)
                closed_list_goal.add(current_node_goal)
                search_path_goal.append(current_node_goal.position)
                
                # Check if we've reached the start directly
                if current_node_goal.position == self.s_start:
                    path = build_path(current_node_goal, start_direction=False)
                    result = {
                        "operation": len(closed_list_start) + len(closed_list_goal),
                        "storage": len(closed_list_start) + len(closed_list_goal),
                        "length": sum(self._euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1)),
                        "llm_output": self.target_list
                    }
                    print(result)
                    visited = list(node.position for node in closed_list_start) + list(node.position for node in closed_list_goal)
                    self.plot.animation(path, visited, True, "LLM-A* Improved", self.filepath)
                    return result
            
            # Check if we've reached the waypoint and update to next waypoint
            if current_waypoint and current_node_start.position == current_waypoint:
                current_waypoint_idx += 1
                if current_waypoint_idx < len(waypoints):
                    current_waypoint = waypoints[current_waypoint_idx]
                else:
                    current_waypoint = None
            
            # Check if paths from both directions have met
            merged_path = meet_in_the_middle()
            if merged_path:
                # Return combined metrics
                result = {
                    "operation": len(closed_list_start) + len(closed_list_goal),
                    "storage": len(closed_list_start) + len(closed_list_goal),
                    "length": sum(self._euclidean_distance(merged_path[i], merged_path[i+1]) for i in range(len(merged_path)-1)),
                    "llm_output": self.target_list
                }
                print(result)
                
                # Visualize the path and search process
                visited = list(node.position for node in closed_list_start) + list(node.position for node in closed_list_goal)
                self.plot.animation(merged_path, visited, True, "LLM-A* Improved", self.filepath)
                
                return result
            
            # Process neighbors in start direction
            if open_list_start:
                for dx, dy in self.u_set:
                    neighbor_pos = (current_node_start.position[0] + dx, current_node_start.position[1] + dy)
                    
                    # Check if position is valid
                    if (neighbor_pos in self.obs or 
                        neighbor_pos[0] < 0 or neighbor_pos[0] >= grid_size_x or 
                        neighbor_pos[1] < 0 or neighbor_pos[1] >= grid_size_y):
                        continue
                    
                    # Skip if this neighbor is already processed
                    if any(node.position == neighbor_pos for node in closed_list_start):
                        continue
                    
                    neighbor = Node(neighbor_pos, current_node_start)
                    
                    # Calculate cost
                    move_cost = self._euclidean_distance(current_node_start.position, neighbor_pos)
                    neighbor.g = current_node_start.g + move_cost
                    
                    # LLM-guided heuristic that considers waypoints
                    if current_waypoint:
                        # Consider both the next waypoint and the final goal
                        waypoint_dist = self._manhattan_distance(neighbor_pos, current_waypoint)
                        goal_dist = self._manhattan_distance(current_waypoint, self.s_goal)
                        neighbor.h = waypoint_dist + goal_dist
                    else:
                        # Direct path to goal
                        neighbor.h = self._manhattan_distance(neighbor_pos, self.s_goal)
                    
                    neighbor.f = neighbor.g + neighbor.h
                    
                    # Check if this is a better path if the node is already in open list
                    existing_node_idx = None
                    for i, open_node in enumerate(open_list_start):
                        if open_node.position == neighbor_pos:
                            existing_node_idx = i
                            break
                    
                    if existing_node_idx is not None:
                        if neighbor.f < open_list_start[existing_node_idx].f:
                            # Replace with better path
                            open_list_start[existing_node_idx] = neighbor
                            # Reorder the heap after modification
                            heapq.heapify(open_list_start)
                    else:
                        # Add new node to open list
                        heapq.heappush(open_list_start, neighbor)
                        search_path_start.append(neighbor_pos)
            
            # Process neighbors in goal direction
            if open_list_goal:
                for dx, dy in self.u_set:
                    neighbor_pos = (current_node_goal.position[0] + dx, current_node_goal.position[1] + dy)
                    
                    # Check if position is valid
                    if (neighbor_pos in self.obs or 
                        neighbor_pos[0] < 0 or neighbor_pos[0] >= grid_size_x or 
                        neighbor_pos[1] < 0 or neighbor_pos[1] >= grid_size_y):
                        continue
                    
                    # Skip if this neighbor is already processed
                    if any(node.position == neighbor_pos for node in closed_list_goal):
                        continue
                    
                    neighbor = Node(neighbor_pos, current_node_goal)
                    
                    # Calculate cost
                    move_cost = self._euclidean_distance(current_node_goal.position, neighbor_pos)
                    neighbor.g = current_node_goal.g + move_cost
                    
                    # Calculate heuristic to start
                    neighbor.h = self._manhattan_distance(neighbor_pos, self.s_start)
                    neighbor.f = neighbor.g + neighbor.h
                    
                    # Check if this is a better path if the node is already in open list
                    existing_node_idx = None
                    for i, open_node in enumerate(open_list_goal):
                        if open_node.position == neighbor_pos:
                            existing_node_idx = i
                            break
                    
                    if existing_node_idx is not None:
                        if neighbor.f < open_list_goal[existing_node_idx].f:
                            # Replace with better path
                            open_list_goal[existing_node_idx] = neighbor
                            # Reorder the heap after modification
                            heapq.heapify(open_list_goal)
                    else:
                        # Add new node to open list
                        heapq.heappush(open_list_goal, neighbor)
                        search_path_goal.append(neighbor_pos)
        
        # If no path found
        result = {
            "operation": len(closed_list_start) + len(closed_list_goal),
            "storage": len(closed_list_start) + len(closed_list_goal),
            "length": 0,
            "llm_output": self.target_list
        }
        print("No path found.")
        print(result)
        
        # Visualize the search process even if no path is found
        visited = list(node.position for node in closed_list_start) + list(node.position for node in closed_list_goal)
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

