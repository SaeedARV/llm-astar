import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


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

    def __str__(self):
        return f"Node(position: {self.position}, g: {self.g}, h: {self.h}, f: {self.f})"


# def heuristic(node, goal):
#     # Use Manhattan distance as a heuristic function
#     return abs(node.position[0] - goal.position[0]) + abs(node.position[1] - goal.position[1])
def heuristic(node, goal):
    # Use Manhattan distance as a heuristic function
    return ((node.position[0] - goal.position[0]) ** 2 + (node.position[1] - goal.position[1]) ** 2) ** 0.5


def get_angle(start, end):
    # Calculate the angle from the start point to the end point
    delta_y = -(end[0] - start[0])
    delta_x = end[1] - start[1]

    angle = math.degrees(math.atan2(delta_y, delta_x))
    if angle < 0:
        angle += 360
    return angle


def filter_neighbors(neighbors, angle, mode):
    # Filter neighbor nodes based on direction and mode (8 or 16 adjacency).
    if mode == 8:
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
    else:  # mode == 16
        direction_map = {
            (11.25, 33.75): ([0, 1, 2, 12, 13, 14, 15], [3, 4, 5, 6, 7, 8, 9, 10, 11]),
            (33.75, 56.25): ([0, 1, 2, 3, 13, 14, 15], [4, 5, 6, 7, 8, 9, 10, 11, 12]),
            (56.25, 78.75): ([0, 1, 2, 3, 4, 14, 15], [5, 6, 7, 8, 9, 10, 11, 12, 13]),
            (78.75, 101.25): ([0, 1, 2, 3, 4, 5, 15], [6, 7, 8, 9, 10, 11, 12, 13, 14]),
            (101.25, 123.75): ([0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12, 13, 14, 15]),
            (123.75, 146.25): ([1, 2, 3, 4, 5, 6, 7], [0, 8, 9, 10, 11, 12, 13, 14, 15]),
            (146.25, 168.75): ([2, 3, 4, 5, 6, 7, 8], [0, 1, 9, 10, 11, 12, 13, 14, 15]),
            (168.75, 191.25): ([3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 10, 11, 12, 13, 14, 15]),
            (191.25, 213.75): ([4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 11, 12, 13, 14, 15]),
            (213.75, 236.25): ([5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 3, 4, 12, 13, 14, 15]),
            (236.75, 258.75): ([6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3, 4, 5, 13, 14, 15]),
            (258.75, 281.25): ([7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 5, 6, 14, 15]),
            (281.25, 303.75): ([8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 6, 7, 15]),
            (303.75, 326.25): ([9, 10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5, 6, 7, 8]),
            (326.25, 348.75): ([0, 10, 11, 12, 13, 14, 15], [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (348.75, 360): ([0, 1, 11, 12, 13, 14], [2, 3, 4, 5, 6, 7, 8, 9, 10]),
            (0, 11.25): ([0, 1, 11, 12, 13, 14, 15], [2, 3, 4, 5, 6, 7, 8, 9, 10])
        }

    for (low, high), (keep, remove) in direction_map.items():
        if low <= angle <= high:
            filtered_neighbors = [neighbors[i] for i in keep if i < len(neighbors)]
            return filtered_neighbors

    return neighbors


def get_neighbors(node, grid, goal, mode):
    """
    Get neighbor nodes, according to different adjacency patterns (8 or 16)。

    Parameter:
        node (Node): The current node
        grid (list): Grid map, 0 means passable and 1 means obstacle
        goal (tuple): The location of the target node
        mode (int): adjacency mode, 8 for 8 adjacency, 16 for 16 adjacency

    Return:
        list: a list of neighbor nodes
    """
    # 8-Direction vector in adjacency mode
    directions_8 = [
        (-1, 1), (-1, 0), (-1, -1),
        (0, -1), (1, -1), (1, 0),
        (1, 1), (0, 1)
    ]

    # 16-Direction vector in adjacency mode
    directions_16 = [
        (-2, 2), (-2, 1), (-2, 0), (-2, -1), (-2, -2),
        (-1, -2), (0, -2), (1, -2), (2, -2),
        (2, -1), (2, 0), (2, 1), (2, 2),
        (1, 2), (0, 2), (-1, 2)
    ]

    neighbors = []  # Stores neighbor nodes
    if mode == 8:
        directions = directions_8
    else:
        directions = directions_16

    has_obstacle = False  # Mark if there is an obstacle
    goal_in_neighbors = False  # Marks whether the target node is in a neighbor node
    obstacles_in_neighbors = []  # Store obstacle nodes

    for dx, dy in directions:
        x, y = node.position[0] + dx, node.position[1] + dy
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
            if (x, y) == goal:
                goal_in_neighbors = True
            if grid[x][y] == 0 or (x, y) == goal:
                if mode == 8 or (dx, dy) not in directions_8:
                    neighbors.append(Node((x, y), node))
            else:
                neighbors.append(Node((x, y), node))
                obstacles_in_neighbors.append(Node((x, y), node))
                has_obstacle = True
        else:
            neighbors.append(Node((x, y), node))
            obstacles_in_neighbors.append(Node((x, y), node))

    # If there are no obstacles in 8 adjacency mode and the target node is not in the neighbor node, switch to 16 adjacency mode
    if mode == 8 and not has_obstacle and not goal_in_neighbors:
        neighbors = get_neighbors(node, grid, goal, 16)
    # If the target node is found in the 8 adjacency mode or the target node is in the neighbor node, the neighbor nodes are filtered according to the direction
    elif mode == 8 or goal_in_neighbors:
        angle = get_angle(node.position, goal)
        neighbors = filter_neighbors(neighbors, angle, 8)
    # If in 16 adjacency mode, the neighbor nodes are filtered based on direction
    elif mode == 16:
        angle = get_angle(node.position, goal)
        neighbors = filter_neighbors(neighbors, angle, 16)

    # Filter out obstacle nodes or offending nodes
    result = [item for item in neighbors if item not in obstacles_in_neighbors]
    return result


def a_star(start, goal, grid, connectivity=8):
    """
    Use the A* algorithm to find the shortest path from the start to the end point in the grid.

    Parameter:
        start (tuple): start coordinate (x, y)
        goal (tuple): End coordinates (x, y)
        grid (list): Grid map, 0 means passable and 1 means obstacle
        connectivity (int): Adjacency mode, 4 indicates 4 adjacency, 8 indicates 8 adjacency, and 16 indicates 16 adjacency

    Return:
        tuple: The shortest path, if it exists, and the search path
    """
    open_list = []  # Priority queue (minimum heap) to store nodes to be explored
    closed_list = set()  # Collections, which are used to store the nodes that have been explored
    start_node = Node(start)  # Create a starting point node
    goal_node = Node(goal)  # Create an endpoint node
    heapq.heappush(open_list, start_node)  # Add the start node to the open_list
    search_path = []  # The path used to store the searched nodes

    while open_list:
        current_node = heapq.heappop(open_list)  # Remove the node with the smallest f-value in the open_list
        closed_list.add(current_node)  # Add the current node to the closed_list
        search_path.append(current_node.position)  # Adds the location of the current node to the search path

        if current_node == goal_node:
            # Find the path, build the path and return the path and search the path
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1], search_path  # Returns the reversed path and the search path

        # Obtain the neighbor node of the current node
        neighbors = get_neighbors(current_node, grid, goal, connectivity)
        for neighbor in neighbors:
            if neighbor in closed_list:
                continue  # If the neighbor node node is in the closed_list, it is skipped

            # Calculate the cost of moving from the current node to the neighbor node
            move_cost = 1 if abs(neighbor.position[0] - current_node.position[0]) + abs(
                neighbor.position[1] - current_node.position[1]) == 1 else 1.414
            tentative_g = current_node.g + move_cost  # Calculate the temporary g-value of neighbor nodes

            # If the neighbor node is in the open_list and the new g value is not better than the existing g value, it is skipped
            if any(open_node for open_node in open_list if neighbor == open_node and tentative_g >= open_node.g):
                continue

            # Update the g-value, h-value, f-value, and parent value of the neighbor node
            neighbor.g = tentative_g
            neighbor.h = heuristic(neighbor, goal_node)
            neighbor.f = neighbor.g + neighbor.h
            neighbor.parent = current_node
            heapq.heappush(open_list, neighbor)  # Add neighbor nodes to open_list
            search_path.append(neighbor.position)  # Add the location of the neighbor node to the search path

    return None, search_path  # Path not found, None and search path are returned


def plot_grid(grid, search, path, start, goal):
    # Draw a grid, search path, and final path
    START_POINT = -1
    END_POINT = -2

    map_data = np.array(grid, dtype=int)
    map_data[start[0], start[1]] = START_POINT
    map_data[goal[0], goal[1]] = END_POINT

    fig, ax = plt.subplots()
    ax.set_xlim(0, map_data.shape[1])
    ax.set_ylim(0, map_data.shape[0])
    ax.invert_yaxis()
    ax.set_aspect('equal')

    for i in range(map_data.shape[0]):
        for j in range(map_data.shape[1]):
            cell_value = map_data[i, j]
            if cell_value == 0:
                rect = plt.Rectangle((j, i), 1, 1, facecolor='white', edgecolor='black')
                ax.add_patch(rect)
            elif cell_value == 1:
                rect = plt.Rectangle((j, i), 1, 1, facecolor='black', edgecolor='black')
                rect.set_zorder(2)
                ax.add_patch(rect)
            elif cell_value == START_POINT:
                rect = plt.Rectangle((j, i), 1, 1, facecolor='red', edgecolor='black')
                ax.add_patch(rect)
            elif cell_value == END_POINT:
                rect = plt.Rectangle((j, i), 1, 1, facecolor='blue', edgecolor='black')
                ax.add_patch(rect)

    for p in search:
        rect = plt.Rectangle((p[1], p[0]), 1, 1, facecolor='grey', edgecolor='black', alpha=1)
        rect.set_zorder(1)
        ax.add_patch(rect)

    for p in path:
        rect = plt.Rectangle((p[1], p[0]), 1, 1, facecolor='green', edgecolor='black')
        ax.add_patch(rect)

    path_x = [p[1] + 0.5 for p in path]
    path_y = [p[0] + 0.5 for p in path]
    ax.plot(path_x, path_y, color='yellow', linewidth=2, linestyle='-')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.show()


def read_data_from_excel(file_path):
    # Read grid data, start and end points from an Excel file
    df = pd.read_excel(file_path, header=None, engine='openpyxl')
    grid = df.iloc[0:50, :].values.tolist()
    start = None
    goal = None
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if value == 'start':
                start = (i, j)
                grid[i][j] = 0
            elif value == 'end':
                goal = (i, j)
                grid[i][j] = 0
    if start is None or goal is None:
        raise ValueError("start or end not found")

    return grid, start, goal


# main function

file_path = 'GridMap1.xlsx'
grid, start, goal = read_data_from_excel(file_path)

# Record the start time
start_time = time.time()

path, search_path = a_star(start, goal, grid)

# Record the end time
end_time = time.time()

# Calculate the time difference
execution_time = end_time - start_time


# Calculate the path length
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


if path:
    path_length = sum(euclidean_distance(path[i], path[i - 1]) for i in range(1, len(path)))
else:
    path_length = 0

# Remove duplicate nodes from search_path
unique_search_path = []
visited = set()
for node in search_path:
    if node not in visited:
        unique_search_path.append(node)
        visited.add(node)

# output
print(f"Number of Expanded Nodes: {len(unique_search_path)}")
print(f"Path Length: {path_length:.6f}")
print(f"Number of Path Points: {len(path) if path else 0}")
print(f"Time (s): {execution_time:.6f}")

# Draw a grid diagram
plot_grid(grid, unique_search_path, path, start, goal)
