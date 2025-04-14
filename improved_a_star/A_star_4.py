import heapq
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import math
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)


def heuristic(node, goal):
    return abs(node.position[0] - goal.position[0]) + abs(node.position[1] - goal.position[1])

# def heuristic(node, goal):
#     # Use Euclidean distance as a heuristic function
#     return ((node.position[0] - goal.position[0]) ** 2 + (node.position[1] - goal.position[1]) ** 2) ** 0.5

def get_neighbors(node, grid, connectivity):
    neighbors = []
    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    elif connectivity == 16:
        directions = [
            (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
            (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
            (0, -2), (0, -1), (0, 1), (0, 2),
            (1, -2), (1, -1), (1, 0), (1, 1), (1, 2),
            (2, -2), (2, -1), (2, 0), (2, 1), (2, 2)
        ]
    else:
        raise ValueError("Invalid connectivity value. Choose 4, 8, or 16.")

    for dx, dy in directions:
        x, y = node.position[0] + dx, node.position[1] + dy
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0:
            neighbors.append(Node((x, y), node))
    return neighbors


def a_star(start, goal, grid, connectivity):
    open_list = []
    closed_list = set()
    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list, start_node)
    search_path = []  # To store nodes that have been searched

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node)
        search_path.append(current_node.position)

        if current_node == goal_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1], search_path  # Return reversed path and search path

        neighbors = get_neighbors(current_node, grid, connectivity)
        for neighbor in neighbors:
            if neighbor in closed_list:
                continue

            move_cost = 1 if abs(neighbor.position[0] - current_node.position[0]) + abs(neighbor.position[1] - current_node.position[1]) == 1 else 1.414
            tentative_g = current_node.g + move_cost

            if any(open_node for open_node in open_list if neighbor == open_node and tentative_g >= open_node.g):
                continue

            neighbor.g = tentative_g
            neighbor.h = heuristic(neighbor, goal_node)
            neighbor.f = neighbor.g + neighbor.h
            neighbor.parent = current_node
            if current_node.position == (1,21):
                print(neighbor.position,neighbor.f)
            heapq.heappush(open_list, neighbor)
            search_path.append(neighbor.position)

    return None, search_path  # No path found


def plot_grid(grid, search, path, start, goal):
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
        rect = plt.Rectangle((p[1], p[0]), 1, 1, facecolor='grey', edgecolor='black')
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
    df = pd.read_excel(file_path, header=None, engine='openpyxl')
    grid = df.iloc[:30, :].values.tolist()
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


file_path = 'GridMap1.xlsx'
start_time = time.time()

connectivity = 8  # Choose 4, 8, or 16 neighborhood
grid, start, goal = read_data_from_excel(file_path)
path, search_path = a_star(start, goal, grid, connectivity)

# end time
end_time = time.time()

# Calculate the time difference
execution_time = end_time - start_time

plot_grid(grid, search_path, path, start, goal)
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


if path:
    path_length = sum(euclidean_distance(path[i], path[i - 1]) for i in range(1, len(path)))
else:
    path_length = 0

# output 
print(f"Number of Expanded Nodes: {len(search_path)}")
print(f"Path Length: {path_length}")
print(f"Number of Path Points: {len(path) if path else 0}")
print(f"Time (s): {execution_time:.6f}")
