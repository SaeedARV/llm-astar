import json
import math
import heapq
import sys
import os

from llmastar.env.search import env, plotting
from llmastar.model import ChatGPT, Llama3
from llmastar.utils import is_lines_collision, list_parse
from .prompt import *

# Import the improved A* implementation directly
IMPROVED_ASTAR_AVAILABLE = False
try:
    # Add the improved_a_star directory to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../improved_a_star')))
    
    # Try to import the necessary functions from A_star_improved2.py
    from A_star_improved2 import a_star_bidirectional, optimize_path, Node as ImprovedNode
    
    # Mark import as successful
    IMPROVED_ASTAR_AVAILABLE = True
    print("Successfully imported A_star_improved2 module")
except Exception as e:
    # Catch any import errors or file not found errors
    print(f"Warning: Could not import from A_star_improved2.py. Error: {str(e)}")
    print("Using fallback implementation for improved A* algorithm.")
    IMPROVED_ASTAR_AVAILABLE = False

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
        
        # Parse query and initialize parameters
        input_data = self._parse_query(query)
        self._initialize_parameters(input_data)
        self._initialize_llm_paths()
        
        # Initialize bidirectional A* components
        open_list_start = [(0, self.s_start)]  # Priority queue for start
        open_list_goal = [(0, self.s_goal)]    # Priority queue for goal
        
        # Use dictionaries for faster lookup
        closed_list_start = {}  # Visited nodes from start
        closed_list_goal = {}   # Visited nodes from goal
        
        # Path tracking
        g_start = {self.s_start: 0}  # Cost from start to current
        g_goal = {self.s_goal: 0}    # Cost from goal to current
        
        parent_start = {self.s_start: None}  # Parent map for start path
        parent_goal = {self.s_goal: None}    # Parent map for goal path
        
        # For path extraction
        meeting_point = None
        
        # Maximum iterations to prevent infinite loops
        max_iter = 2000
        iter_count = 0
        
        # Store visited nodes for visualization
        visited_nodes = set()
        
        # Helper function to calculate heuristic (Manhattan distance)
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # Helper function to find neighboring points
        def get_neighbors(node):
            neighbors = []
            for dx, dy in self.u_set:  # u_set contains possible movements
                neighbor = (node[0] + dx, node[1] + dy)
                # Check if neighbor is within bounds
                if (0 <= neighbor[0] <= self.range_x[1] and 
                    0 <= neighbor[1] <= self.range_y[1]):
                    neighbors.append(neighbor)
            return neighbors
        
        # Helper function to build path from meeting point
        def build_path(meeting_node):
            # Build path from start to meeting point
            path_from_start = []
            current = meeting_node
            while current is not None:
                path_from_start.append(current)
                current = parent_start[current]
            path_from_start.reverse()
            
            # Build path from meeting point to goal
            path_from_goal = []
            current = meeting_node
            current = parent_goal[current]  # Skip the meeting node itself
            while current is not None:
                path_from_goal.append(current)
                current = parent_goal[current]
            
            # Combine paths (meeting_node is included only once, in path_from_start)
            complete_path = path_from_start + path_from_goal
            return complete_path
        
        # Helper function to find a meeting point
        def find_meeting_point():
            # Check all nodes in closed_list_start for presence in closed_list_goal
            for node in closed_list_start:
                if node in closed_list_goal:
                    return node
            return None
        
        # Main search loop
        while open_list_start and open_list_goal and iter_count < max_iter:
            iter_count += 1
            
            # Process from start direction
            if open_list_start:
                _, current_start = heapq.heappop(open_list_start)
                
                # Skip if already processed
                if current_start in closed_list_start:
                    continue
                
                # Add to closed list
                closed_list_start[current_start] = True
                visited_nodes.add(current_start)
                
                # Check if we've reached a node processed from goal
                if current_start in closed_list_goal:
                    meeting_point = current_start
                    break
                
                # Process neighbors
                for neighbor in get_neighbors(current_start):
                    # Skip if already processed
                    if neighbor in closed_list_start:
                        continue
                    
                    # Check for collisions
                    if self.is_collision(current_start, neighbor):
                        continue
                    
                    # Calculate new cost
                    tentative_g = g_start[current_start] + self._euclidean_distance(current_start, neighbor)
                    
                    # Update if better path found
                    if neighbor not in g_start or tentative_g < g_start[neighbor]:
                        g_start[neighbor] = tentative_g
                        f_value = tentative_g + heuristic(neighbor, self.s_goal)
                        heapq.heappush(open_list_start, (f_value, neighbor))
                        parent_start[neighbor] = current_start
            
            # Process from goal direction
            if open_list_goal:
                _, current_goal = heapq.heappop(open_list_goal)
                
                # Skip if already processed
                if current_goal in closed_list_goal:
                    continue
                
                # Add to closed list
                closed_list_goal[current_goal] = True
                visited_nodes.add(current_goal)
                
                # Check if we've reached a node processed from start
                if current_goal in closed_list_start:
                    meeting_point = current_goal
                    break
                
                # Process neighbors
                for neighbor in get_neighbors(current_goal):
                    # Skip if already processed
                    if neighbor in closed_list_goal:
                        continue
                    
                    # Check for collisions
                    if self.is_collision(current_goal, neighbor):
                        continue
                    
                    # Calculate new cost
                    tentative_g = g_goal[current_goal] + self._euclidean_distance(current_goal, neighbor)
                    
                    # Update if better path found
                    if neighbor not in g_goal or tentative_g < g_goal[neighbor]:
                        g_goal[neighbor] = tentative_g
                        f_value = tentative_g + heuristic(neighbor, self.s_start)
                        heapq.heappush(open_list_goal, (f_value, neighbor))
                        parent_goal[neighbor] = current_goal
            
            # Check for meeting point
            if iter_count % 10 == 0:  # Periodically check to reduce overhead
                meeting_point = find_meeting_point()
                if meeting_point:
                    break
        
        # If we found a meeting point, construct the path
        if meeting_point:
            path = build_path(meeting_point)
            
            # Calculate result metrics
            result = {
                "operation": len(closed_list_start) + len(closed_list_goal),
                "storage": len(g_start) + len(g_goal),
                "length": sum(self._euclidean_distance(path[i], path[i+1]) for i in range(len(path)-1)),
                "path": path,
                "llm_output": self.target_list
            }
            print("Path found with bidirectional A*!")
            print(result)
            
            # Visualize the path
            self.plot.animation(path, list(visited_nodes), True, "LLM-A* Improved (Bidirectional)", self.filepath)
            
            return result
        
        # If no path is found using bidirectional A*, fall back to regular A*
        print("No path found with bidirectional A*. Falling back to regular A*.")
        return self.searching(query, filepath)

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

