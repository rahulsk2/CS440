# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,greedy,astar)

import queue as Q
import sys


class Node:
    """A node value that encapsulates the node data and its heuristic value

        Attributes:
            position: a (row -> int, column -> int) tuple
            heuristic_value: an int that captures the heuristic value of the Node
    """

    def __init__(self, position, heuristic_value=0.0):
        """Return a Node Object"""
        self.position = (position[0], position[1])
        self.heuristic_value = heuristic_value

    def __str__(self):
        return "Position: " + str(self.position) + " Heuristic Value: " + str(self.heuristic_value)

    def __hash__(self):
        return hash(str(self.position))

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.heuristic_value < other.heuristic_value


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "dfs": dfs,
        "greedy": greedy,
        "astar": astar,
    }.get(searchMethod)(maze)



"""
This is used to pre-compute and store the graph weights for Dijkstra style heuristic
"""
distanceMap = {}

###########################
#   SEARCH FUNCTIONS      #
###########################

"""
We separate it into a search() function and a search_executor() function
The search() is the driver and the executor bit runs it for every food pellet to food pellet instance
All searches run on 1 or more objectives
"""


# Breadth First Search


def bfs_executor(maze, start_state, objectives):
    """
    This is the executor for BFS.
    The idea is to use a FIFO queue to maintain the frontier
    We use the parent_map to store the parent state of every expanded node
    We then use a utility backtrace function to get the complete path from start to end
    :param maze:
    :param start_state:
    :param objectives:
    :return: path, num_states_explored --> for a sub-goal to sub-goal
    """
    frontier = Q.Queue()
    start_node = Node(start_state)
    frontier.put(start_node)
    visited = set()
    parent_map = {}
    num_states_explored = 0
    while frontier:
        current_node = frontier.get()
        current = current_node.position
        visited.add(current)
        num_states_explored += 1
        if current in objectives:
            current_goal = current
            break
        neighbour_nodes = maze.getNeighbors(current[0], current[1])
        for each_neighbour in neighbour_nodes:
            if maze.isValidMove(each_neighbour[0], each_neighbour[1]) and each_neighbour not in visited:
                parent_map[each_neighbour] = current
                frontier.put(Node(each_neighbour))

    return backtrace(parent_map, start_state, current_goal), num_states_explored


def bfs(maze):
    """
    This acts like the driver for BFS and combines the various sub paths returned by bfs_executor
    :param maze:
    :return: full_path, total_states_explored
    """
    start_state = maze.getStart()
    objectives = maze.getObjectives()
    full_path = []
    total_states_explored = 0
    while objectives:
        path, states_explored = bfs_executor(maze, start_state, objectives)
        full_path.extend(path)
        total_states_explored += states_explored
        objectives.remove(path[-1])
        start_state = path[-1]
    return full_path, total_states_explored


# Depth First Search


def dfs_executor(maze, start_state, objectives):
    """
    The idea here is to use a stack as a LIFO frontier.
    In python we can just use the list and its pop() method to get LIFO behaviour
    Since the state space does not become extremely large for DFS,
    we have chosen to maintain the entire source to destination paths in the frontier
    :param maze:
    :param start_state:
    :param objectives:
    :return: path, num_states_explored
    """
    frontier = [[start_state]]
    visited = set()
    path = []
    num_states_explored = 0
    while frontier:
        path = frontier.pop()
        current_node = path[-1]
        visited.add(current_node)
        num_states_explored += 1
        if (current_node[0], current_node[1]) in objectives:
            break
        neighbour_nodes = maze.getNeighbors(current_node[0], current_node[1])
        for each_neighbour in neighbour_nodes:
            if maze.isValidMove(each_neighbour[0], each_neighbour[1]) and each_neighbour not in visited:
                new_path = list(path)
                new_path.append(each_neighbour)
                frontier.append(new_path)

    return path, num_states_explored


def dfs(maze):
    """
    The driver for DFS
    :param maze:
    :return: full_path, total_states_explored
    """
    start_state = maze.getStart()
    objectives = maze.getObjectives()
    full_path = []
    total_states_explored = 0
    while objectives:
        path, states_explored = dfs_executor(maze, start_state, objectives)
        full_path.extend(path)
        total_states_explored += states_explored
        objectives.remove(path[-1])
        start_state = path[-1]
    return full_path, total_states_explored

# GREEDY SEARCH


def getGreedyHeuristic(current_state, objectives):
    """
    This is the heuristic for a naive greedy search.
    For single goal, we return the manhattan distance from current to end
    For multiple goal, we return the minimum of the manhattan distance from current to each goal state

    With this, we use a Priority Queue to maintain the frontier
    We defined __lt__ on Node class to work with how Priority Queue works in python
    The Priority Queue get() always returns the Node from frontier with minimum heuristic value

    :param current_state:
    :param objectives:
    :return: heuristic h(x)
    """
    min_heuristic = sys.maxsize
    for each_objective in objectives:
        manhattan_objective = abs(current_state[0] - each_objective[0]) + abs(current_state[1] - each_objective[1])
        if manhattan_objective < min_heuristic:
            min_heuristic = manhattan_objective
    return min_heuristic


def greedy_executor(maze, start_state, objectives):
    """
    The executor for greedy search.
    It uses the heuristic getGreedyHeuristic as h(x) to guide the search
    :param maze:
    :param start_state:
    :param objectives:
    :return:
    """
    frontier = Q.PriorityQueue()
    start_node = Node(start_state, getGreedyHeuristic(start_state, objectives))
    frontier.put(start_node)
    visited = set()
    parent_map = {}
    num_states_explored = 0
    while frontier:
        current_node = frontier.get()
        current_position = current_node.position
        visited.add(current_position)
        num_states_explored += 1

        if current_position in objectives:
            current_goal = current_position
            break
        neighbour_nodes = maze.getNeighbors(current_position[0], current_position[1])
        for each_neighbour in neighbour_nodes:
            if maze.isValidMove(each_neighbour[0], each_neighbour[1]) and each_neighbour not in visited:
                parent_map[each_neighbour] = current_position
                frontier.put(Node(each_neighbour, getGreedyHeuristic(each_neighbour, objectives)))
    return backtrace(parent_map, start_state, current_goal), num_states_explored


def greedy(maze):
    """
    The driver for Greedy Search
    :param maze:
    :return: full_path, total_states_explored
    """
    start_state = maze.getStart()
    objectives = maze.getObjectives()
    full_path = []
    total_states_explored = 0
    while objectives:
        path, states_explored = greedy_executor(maze, start_state, objectives)
        full_path.extend(path)
        total_states_explored += states_explored
        objectives.remove(path[-1])
        start_state = path[-1]
    return full_path, total_states_explored


# ASTAR SEARCH

# SEARCH

def astar_executor(maze, start_state, objectives):
    """
    The astar search works just like Greedy, other than the fact that the hueristic is the sum of a heuristic function
    and the cost to reach the current state
    We tried out different heuristics, most of which are admissible
    We got the best results from a weighted minimum manhattan distance to the nearest goal and have used that as the default one
    Each Heuristic has been described in the corresponding sections
    :param maze:
    :param start_state:
    :param objectives:
    :return:
    """
    frontier = Q.PriorityQueue()
    start_node = Node(start_state, getWeightedAstarHeuristicMinDistanceToAnyObjective(start_state, objectives) + 0)
    frontier.put(start_node)
    visited = set()
    parent_map = {}
    num_states_explored = 0
    while frontier:
        current_node = frontier.get()
        current_position = current_node.position
        visited.add(current_position)
        num_states_explored += 1

        if current_position in objectives:
            current_goal = current_position
            break
        neighbour_nodes = maze.getNeighbors(current_position[0], current_position[1])
        for each_neighbour in neighbour_nodes:
            if maze.isValidMove(each_neighbour[0], each_neighbour[1]) and each_neighbour not in visited:
                parent_map[each_neighbour] = current_position
                frontier.put(Node(each_neighbour, getWeightedAstarHeuristicMinDistanceToAnyObjective(each_neighbour, objectives) +
                                                  len(backtrace(parent_map, start_state, current_position))))
    return backtrace(parent_map, start_state, current_goal), num_states_explored


def astar(maze):
    """
    The driver for Astar
    :param maze:
    :return: full_path, total_states_explored
    """
    start_state = maze.getStart()
    objectives = maze.getObjectives()
    #calculateAllDistances(maze, start_state, objectives)
    #print("DistanceMap Built!")
    full_path = []
    total_states_explored = 0
    while objectives:
        path, states_explored = astar_executor(maze, start_state, objectives)
        full_path.extend(path)
        total_states_explored += states_explored
        objectives.remove(path[-1])
        start_state = path[-1]
    return full_path, total_states_explored



# HEURISTICS


def getAstarHeuristicMinDistanceToAnyObjective(current_state, objectives):
    """
    This is a simple Manhattan Distance heuristic
    We return the minimum manhattan distance to the nearest goal
    :param current_state:
    :param objectives:
    :return: heuritic value
    """
    min_heuristic = sys.maxsize
    for each_objective in objectives:
        manhattan_objective = abs(current_state[0] - each_objective[0]) + abs(current_state[1] - each_objective[1])
        if manhattan_objective < min_heuristic:
            min_heuristic = manhattan_objective
    return min_heuristic


def getWeightedAstarHeuristicMinDistanceToAnyObjective(current_state, objectives):
    """
       This is a wighted Manhattan Distance heuristic
       We return the minimum manhattan distance to the nearest goal and multiply it by a constant factor
       Another implementation could have been where we vary the constant weight to see what gives the best result
       :param current_state:
       :param objectives:
       :return: heuritic value
       """
    min_heuristic = sys.maxsize
    for each_objective in objectives:
        manhattan_objective = abs(current_state[0] - each_objective[0]) + abs(current_state[1] - each_objective[1])
        if manhattan_objective < min_heuristic:
            min_heuristic = manhattan_objective
    if len(objectives) > 2:
        return 2 * min_heuristic
    else:
        return min_heuristic


def getAstarHeuristicSumOfAllGoals(current_state, objectives):
    """
       This is the sum of all manhattan distance from current to all goal state
       This would not be admissible as it overestimates the cost to traverse all goals
       :param current_state:
       :param objectives:
       :return: heuritic value
       """
    sum_heuristic = 0
    for each_objective in objectives:
        sum_heuristic += abs(current_state[0] - each_objective[0]) + abs(current_state[1] - each_objective[1])
    return sum_heuristic


def getAstarHeuristicSumOfMinimumConnectedGoals(current_state, objectives):
    """
    Here we find the manhattan distance to the nearest goal, and then we sum it recursively with
    the path costs from that goal to the next nearest goal
    :param current_state:
    :param objectives:
    :return: value
    """
    if not objectives:
        return 0
    min_heuristic = sys.maxsize
    for each_objective in objectives:
        manhattan_objective = abs(current_state[0] - each_objective[0]) + abs(current_state[1] - each_objective[1])
        if manhattan_objective < min_heuristic:
            min_heuristic = manhattan_objective
            min_objective = each_objective
    new_objectives = objectives.remove(min_objective)
    return min_heuristic + getAstarHeuristicSumOfMinimumConnectedGoals(min_objective, new_objectives)


def getAstarHeuristicSumOfMinimumConnectedGoalsPreComputed(current_state, objectives):
    """
        This is used in association with the next method (aggregatedCostFromObjectiveToAllOthers)
        The logic is that we precompute the manhattan distance from a each objective to every other objective
        Then we calculate the heuristic value as
        h(x) = cost from current to nearest goal, dijkstra cost from that to all other goals
        This was our most promising effort, but sadly we are getting a suboptimal result with this heuristic
        :param current_state:
        :param objectives:
        :return: heuristic value
        """
    min_heuristic = sys.maxsize
    for each_objective in objectives:
        manhattan_objective = abs(current_state[0] - each_objective[0]) + abs(current_state[1] - each_objective[1])
        if manhattan_objective < min_heuristic:
            min_heuristic = manhattan_objective
            min_objective = each_objective
    return min_heuristic + aggregatedCostFromObjectiveToAllOthers(min_objective, objectives)


def aggregatedCostFromObjectiveToAllOthers(current, objectives):
    """
    Calculates the Dijkstra single source all paths distance between each goal state
    We make use of the pre computed distance matrix
    :param current:
    :param objectives:
    :return:
    """
    dijkstraMap = {}
    for each_objective in objectives:
        dijkstraMap[each_objective] = sys.maxsize
    dijkstraMap[current] = 0
    new_objectives = objectives[:]
    while new_objectives:
        current_vertex = min(new_objectives, key=lambda objective: dijkstraMap[objective])
        if dijkstraMap[current_vertex] == sys.maxsize:
            break
        for neighbour, cost in dijkstraMap.items():
            if neighbour == current_vertex:
                continue
            alternative_route = dijkstraMap[current_vertex] + distanceMap[current_vertex, neighbour]
            if alternative_route < dijkstraMap[neighbour]:
                dijkstraMap[neighbour] = alternative_route
        new_objectives.remove(current_vertex)
    sum_distances = 0
    for distances in dijkstraMap.values():
        sum_distances += distances
    return sum_distances



###########################
#   UTILITY FUNCTIONS     #
###########################


def backtrace(parent_map, start, end):
    """
    This method is a utility function.
    It helps us trace the solution path from start to goal.
    :param parent_map:
    :param start:
    :param end:
    :return: list of path from start to end
    """
    path = [end]
    while path[-1] != start:
        path.append(parent_map[path[-1]])
    path.reverse()
    return path


def calculateAllDistances(maze, start, objectives):
    """
    This is a utility function that populates distanceMap
    It finds the manhattan distance from the current goal to all other goals
    :param maze:
    :param start:
    :param objectives:
    :return:
    """
    for objective in objectives:
        manhattan_distance = abs(start[0] - objective[0]) + abs(start[1] - objective[1])
        distanceMap[(start, objective)] = manhattan_distance
    for objective1 in objectives:
        for objective2 in objectives:
            if objective1 == objective2:
                continue
            if (objective2, objective1) in distanceMap.keys():
                distanceMap[(objective1, objective2)] = distanceMap[(objective2, objective1)]
            else:
                manhattan_distance = abs(objective1[0] - objective2[0]) + abs(objective1[1] - objective2[1])
                distanceMap[objective1,objective2] = manhattan_distance
