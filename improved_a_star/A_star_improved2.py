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


def heuristic(node, goal):
    # Use Manhattan distance as a heuristic function
    return abs(node.position[0] - goal.position[0]) + abs(node.position[1] - goal.position[1])
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
            (11.25, 33.75): ([0, 1, 2, 3, 11, 12, 13, 14, 15], [4, 5, 6, 7, 8, 9, 10]),
            (33.75, 56.25): ([0, 1, 2, 3, 4, 12, 13, 14, 15], [5, 6, 7, 8, 9, 10, 11]),
            (56.25, 78.75): ([0, 1, 2, 3, 4, 5, 13, 14, 15], [6, 7, 8, 9, 10, 11, 12]),
            (78.75, 101.25): ([0, 1, 2, 3, 4, 5, 6, 14, 15], [7, 8, 9, 10, 11, 12, 13]),
            (101.25, 123.75): ([0, 1, 2, 3, 4, 5, 6, 7, 15], [8, 9, 10, 11, 12, 13, 14]),
            (123.75, 146.25): ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15]),
            (146.25, 168.75): ([1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 10, 11, 12, 13, 14, 15]),
            (168.75, 191.25): ([2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 11, 12, 13, 14, 15]),
            (191.25, 213.75): ([3, 4, 5, 6, 7, 8, 9, 10, 11], [0, 1, 2, 12, 13, 14, 15]),
            (213.75, 236.25): ([4, 5, 6, 7, 8, 9, 10, 11, 12], [0, 1, 2, 3, 13, 14, 15]),
            (236.75, 258.75): ([5, 6, 7, 8, 9, 10, 11, 12, 13], [0, 1, 2, 3, 4, 14, 15]),
            (258.75, 281.25): ([6, 7, 8, 9, 10, 11, 12, 13, 14], [0, 1, 2, 3, 4, 5, 15]),
            (281.25, 303.75): ([7, 8, 9, 10, 11, 12, 13, 14, 15], [0, 1, 2, 3, 4, 5, 6]),
            (303.75, 326.25): ([0, 8, 9, 10, 11, 12, 13, 14, 15], [1, 2, 3, 4, 5, 6, 7]),
            (326.25, 348.75): ([0, 1, 9, 10, 11, 12, 13, 14, 15], [2, 3, 4, 5, 6, 7, 8]),
            (348.75, 360): ([0, 1, 2, 10, 11, 12, 13, 14], [3, 4, 5, 6, 7, 8, 9]),
            (0, 11.25): ([0, 1, 2, 10, 11, 12, 13, 14, 15], [3, 4, 5, 6, 7, 8, 9])
        }

    for (low, high), (keep, remove) in direction_map.items():
        if low <= angle <= high:
            filtered_neighbors = [neighbors[i] for i in keep if i < len(neighbors)]
            return filtered_neighbors

    return neighbors


def get_neighbors(node, grid, goal, mode):
    """
    Get neighbor node nodes, depending on the adjacency mode (8 or 16).

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


def a_star_bidirectional(start, goal, grid, connectivity=8):
    open_list_start = []  # Priority queue in the direction of the starting point
    open_list_goal = []  # Priority queue in the direction of the endpoint
    closed_list_start = set()  # The explored node in the direction of the starting point
    closed_list_goal = set()  # The explored node in the end direction
    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list_start, start_node)
    heapq.heappush(open_list_goal, goal_node)
    search_path_start = []  # The search path in the direction of the starting point
    search_path_goal = []  # The search path in the direction of the end point

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
            path_start = build_path(goal_meet_node)
            path_goal = build_path(start_meet_node, start_direction=False)
            return path_start, path_goal[:]  # Merge paths to avoid duplicating intermediate nodes

        return None, None

    while open_list_start and open_list_goal:
        current_node_start = heapq.heappop(open_list_start)
        current_node_goal = heapq.heappop(open_list_goal)
        closed_list_start.add(current_node_start)
        closed_list_goal.add(current_node_goal)
        search_path_start.append(current_node_start.position)
        search_path_goal.append(current_node_goal.position)

        if current_node_start == goal_node or current_node_goal == start_node or current_node_start in closed_list_goal:
            path1, path2 = meet_in_the_middle()
            if path1:
                return path1, path2, search_path_start, search_path_goal

        # The cost of updating the neighbor node in the direction of the origin
        neighbors_start = get_neighbors(current_node_start, grid, goal, connectivity)
        for neighbor in neighbors_start:
            if neighbor in closed_list_start:
                continue
            move_cost = math.sqrt((neighbor.position[0] - current_node_start.position[0]) ** 2 + (
                    neighbor.position[1] - current_node_start.position[1]) ** 2)
            tentative_g = current_node_start.g + move_cost
            heuristic_cost = min(
                heuristic(neighbor, node) for node in closed_list_goal) if closed_list_goal else heuristic(neighbor,
                                                                                                           goal_node)
            neighbor.g = tentative_g
            neighbor.h = heuristic_cost
            neighbor.f = neighbor.g + neighbor.h
            neighbor.parent = current_node_start
            search_path_start.append(neighbor.position)
            heapq.heappush(open_list_start, neighbor)

        # The cost of updating the neighbor node in the end direction
        neighbors_goal = get_neighbors(current_node_goal, grid, start, connectivity)
        for neighbor in neighbors_goal:
            if neighbor in closed_list_goal:
                continue
            move_cost = math.sqrt((neighbor.position[0] - current_node_goal.position[0]) ** 2 + (
                    neighbor.position[1] - current_node_goal.position[1]) ** 2)
            tentative_g = current_node_goal.g + move_cost
            heuristic_cost = min(
                heuristic(neighbor, node) for node in closed_list_start) if closed_list_start else heuristic(neighbor,
                                                                                                             start_node)
            neighbor.g = tentative_g
            neighbor.h = heuristic_cost
            neighbor.f = neighbor.g + neighbor.h
            neighbor.parent = current_node_goal
            search_path_goal.append(neighbor.position)
            heapq.heappush(open_list_goal, neighbor)

    return None, None, search_path_start, search_path_goal


def plot_grid(grid, search_start, search_goal, path1, path2, start, goal):
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
            ax.text(j + 0.5, i + 0.5, f'({i},{j})', ha='center', va='center', fontsize=8)

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

    for p in search_start:
        rect = plt.Rectangle((p[1], p[0]), 1, 1, facecolor='grey', edgecolor='black', alpha=1)
        rect.set_zorder(1)
        ax.add_patch(rect)

    for p in search_goal:
        rect = plt.Rectangle((p[1], p[0]), 1, 1, facecolor='orange', edgecolor='black', alpha=1)
        rect.set_zorder(1)
        ax.add_patch(rect)

    for p in path1:
        rect = plt.Rectangle((p[1], p[0]), 1, 1, facecolor='green', edgecolor='black')
        ax.add_patch(rect)

    path_x = [p[1] + 0.5 for p in path1]
    path_y = [p[0] + 0.5 for p in path1]
    ax.plot(path_x, path_y, color='yellow', linewidth=2, linestyle='-')
    for p in path2:
        rect = plt.Rectangle((p[1], p[0]), 1, 1, facecolor='blue', edgecolor='black')
        ax.add_patch(rect)

    path_x = [p[1] + 0.5 for p in path2]
    path_y = [p[0] + 0.5 for p in path2]
    ax.plot(path_x, path_y, color='yellow', linewidth=2, linestyle='-')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.show()


def optimize_path(path, grid, goal):
    if not path:
        return []

    optimized_path = [path[0]]  # start
    current = path[0]
    i = 1
    while current != goal:
        if i >= len(path):
            break

        next_point = path[i]

        if not is_line_clear(current, next_point, grid):
            optimized_path.append(path[i - 1])  # If there is an obstacle, use the previous point as the waypoint
            current = path[i - 1]  # Updates the current point
            i += 1
        else:
            i += 1

    if optimized_path[-1] != path[-1]:
        optimized_path.append(path[-1])  # Make sure that the end point is added to the route

    return optimized_path


def get_line_points(start, end):
    """All points on the line are calculated based on two points and deduplicated"""
    x0, y0 = start
    x1, y1 = end

    points = set()

    dx = x1 - x0
    dy = y1 - y0

    # Calculate the step size to make sure the step size is not zero
    steps = max(abs(dx), abs(dy)) * 10
    if steps == 0:
        points.add((x0, y0))
        return list(points)

    for i in range(steps + 1):
        t = i / steps
        x = x0 + t * dx
        y = y0 + t * dy
        points.add((int(round(x+0.01)), int(round(y-0.01))))

    return list(points)


def is_line_clear(start, end, grid):
    """
    Determine if there are any obstacles between two points in the grid map

    Parameter:
    start -- Start coordinates (x0, y0)
    end -- End coordinates (x1, y1)
    grid -- Grid map, 1 for obstacles and 0 for passability

    Return value:
    True -- If there are no obstacles
    False -- If there are obstacles
    """
    line_points = get_line_points(start, end)

    for (x, y) in line_points:
        if grid[x][y] == 1:  # Note grid[y][x] here, not grid[x][y]
            return False

    return True


def read_data_from_excel(file_path):
    df = pd.read_excel(file_path, header=None, engine='openpyxl')
    grid = df.iloc[0:30, :].values.tolist()
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


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# Only run if this script is executed directly (not when imported)
if __name__ == "__main__":
    # main function
    file_path = 'GridMap1.xlsx'
    grid, start, goal = read_data_from_excel(file_path)

    # Record the start time
    start_time = time.time()

    path1, path2, search_path_start, search_path_goal = a_star_bidirectional(start, goal, grid)

    # Record the end time
    end_time = time.time()

    # Calculate the time difference
    execution_time = end_time - start_time

    # Calculate the path length
    path = path2 + path1
    path = optimize_path(path, grid, goal)
    if path:
        path_length = sum(euclidean_distance(path[i], path[i - 1]) for i in range(1, len(path)))
    else:
        path_length = 0

    # Remove duplicate nodes from search_path
    unique_search_path_start = []
    visited_start = set()
    for node in search_path_start:
        if node not in visited_start:
            unique_search_path_start.append(node)
            visited_start.add(node)

    unique_search_path_goal = []
    visited_goal = set()
    for node in search_path_goal:
        if node not in visited_goal:
            unique_search_path_goal.append(node)
            visited_goal.add(node)

    # output
    print(f"Number of Extended Nodes (Start Direction): {len(unique_search_path_start)}")
    print(f"Number of Extension Nodes (End Direction): {len(unique_search_path_goal)}")
    print(f"Path Length: {path_length:.6f}")
    print(f"Number of Path Points: {len(path1) if path1 else 0}")
    print(f"Time (s): {execution_time:.6f}")

    # Draw a grid diagram
    plot_grid(grid, unique_search_path_start, unique_search_path_goal, path, [], start, goal)
