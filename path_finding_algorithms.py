"""
Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions.
"""

import sys
from collections import deque

from utils import *

import matplotlib.pyplot as plt


class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


def depth_first_graph_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontier = [(Node(problem.initial))]  # Stack

    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node, len(explored)
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return len(explored)


def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child, len(explored)
                frontier.append(child)
    return len(explored)


def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node, len(explored)
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)
    return len(explored)

# ______________________________________________________________________________
# Informed (Heuristic) Search


greedy_best_first_graph_search = best_first_graph_search


# Greedy best-first search is accomplished by specifying f(n) = h(n).


def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)


# ______________________________________________________________________________
# A* heuristics 

# ______________________________________________________________________________



# ______________________________________________________________________________
# Graphs and Graph Problems


class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)


def load_graph_from_file(filename):

    # Set up data structures
    nodes = {}
    edges = {}
    origin = None
    destinations = []
    graph_map = Graph()

    # Open and read the text file
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Divide the lines from the text file into sections and name each section by its component
    section = None
    for line in lines:
        line = line.strip()
        if line == "Nodes:":
            section = "nodes"
            continue
        elif line == "Edges:":
            section = "edges"
            continue
        elif line == "Origin:":
            section = "origin"
            continue
        elif line == "Destinations:":
            section = "destinations"
            continue

        if not line:
            continue

        # Assign each section's lines with each element in the corresponding data structure
        if section == "nodes":
            # e.g. {1:(2,3)}
            parts = line.split(":")
            node = int(parts[0])
            coords = list(map(int, parts[1].strip(" ()").split(',')))
            nodes[node] = coords
        elif section == "edges":
            # e.g. {(1,2):3}
            parts = line.split(":")
            n1, n2 = map(int, parts[0].strip(" ()").split(','))
            cost = int(parts[1])          
            edges.setdefault((n1, n2), cost)
        elif section == "origin":
            origin = int(line)
        elif section == "destinations":
            destinations = list(map(int, line.split(';')))

    # Create the graph from these sections
    for node in nodes:
        for n, cost in edges.items():
            if n[0] == node:
                graph_map.connect1(n[0], n[1], cost) # One way connection
    graph_map.locations = nodes       

    return graph_map, origin, destinations

def draw_solution(graph_map, origin, destinations, solution_path, title):

    # Set up Figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set up Grid
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.grid(True)
    ax.set_aspect('equal')
    
    # Draw vertices
    vertices = graph_map.locations
    for name, (x, y) in vertices.items():
        if name == origin:
            color = 'limegreen'
        elif name in destinations:
            color = 'orange'
        else:
            color = 'skyblue'

        ax.plot(x, y, 'o', markersize=10, color=color)
        label = str(name)
        if name == origin:
            label += " (Origin)"
        elif name in destinations:
            label += " (Dest)"
        ax.text(x + 0.1, y + 0.1, label, fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
               zorder=4)

    # Draw edges
    edges = {}
    for node1 in graph_map.graph_dict:
        for node2, cost in graph_map.graph_dict[node1].items():
            edges[(node1, node2)] = cost
    edge_set = set(edges.keys())
    drawn = set() # Track whether the edge has been drawn 
    for (src, dst) in edges.keys():
        x0, y0 = vertices[src]
        x1, y1 = vertices[dst]
        
        if (dst, src) in edge_set and (dst, src) not in drawn:
            # Undirected
            ax.plot([x0, x1], [y0, y1], 'k-', lw=1.5)
            ax.plot(x0, y0, 'o', markersize=5, color='black')
            ax.plot(x1, y1, 'o', markersize=5, color='black')

        elif (dst, src) not in edge_set:
            # Directed
            ax.annotate("",
                       xy=(x1, y1), xycoords='data',
                       xytext=(x0, y0), textcoords='data',
                       arrowprops=dict(arrowstyle="->", color='black', lw=1.5, mutation_scale=20))

        drawn.add((src, dst))

    # Draw Solution Path
    if solution_path and len(solution_path) > 1:
        for i in range(len(solution_path) - 1):
            src = solution_path[i]
            dst = solution_path[i + 1]
            
            x0, y0 = vertices[src]
            x1, y1 = vertices[dst]
            
            if (src, dst) in edges:
                ax.annotate("",
                            xy=(x1, y1), xycoords='data',
                            xytext=(x0, y0), textcoords='data',
                            arrowprops=dict(arrowstyle="->", color='red', lw=3, mutation_scale=20))
                    
    plt.title(title)
    plt.show()

class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""

    def __init__(self, initial, goal, graph):
        super().__init__(initial, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if isinstance(self.goal, list):
                if type(node) is str:
                    return min(int(distance(locs[node], locs[goal])) for goal in self.goal)
                else:
                    return min(int(distance(locs[node.state], locs[goal])) for goal in self.goal)
            else:
                if type(node) is str:
                    return int(distance(locs[node], locs[self.goal]))

                return int(distance(locs[node.state], locs[self.goal]))
         
        else:
            return np.inf


def runGraphSeacrh():
    # Extract test file and method from CLI arguments
    filename = sys.argv[1]
    method = sys.argv[2]

    # Load graph and extract info
    graph_map, origin, destinations = load_graph_from_file(filename)

    # Loop over each destination
    for goal in destinations:
        prob = GraphProblem(origin, goal, graph_map)

        result = None
        path = []
        no_nodes_explored = 0

        # Run the selected search method
        if method == "DFS":
            result, no_nodes_explored = depth_first_graph_search(prob)
        elif method == "BFS":
            result, no_nodes_explored = breadth_first_graph_search(prob)
        elif method == "GBFS":
            result, no_nodes_explored = best_first_graph_search(prob, lambda n: prob.h(n), display=True)
        elif method == "AS":
            result, no_nodes_explored = astar_search(prob, lambda n: prob.h(n), display=True)
        else:
            print(f"Unknown method: {method}")
            continue

        # Handle result
        if result:
            path = [node.state for node in result.path()]

            print(f"{filename} {method}")
            print(f"{result.state} {no_nodes_explored}")
            print(path)

            draw_solution(graph_map, origin, [goal], path, f"{method} Path to {goal}")
        else:
            print(f"No path found to {goal}")


# Run it
runGraphSeacrh()

