# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""

import heapq as hq
import os
import pickle
import math
import itertools


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        while self.queue:
            priority,count,task = hq.heappop(self.queue)
            # if task is not self.REMOVED:
            #     return (priority,task)
            return (priority,task)
        raise KeyError('pop from an empty priority queue')

        # TODO: finish this function!
        #raise NotImplementedError

    def remove(self, node):
        """
        Remove a node from the queue.

        Hint: You might require this in ucs. However, you may
        choose not to use it or to define your own method.

        Args:
            node (tuple): The node to remove from the queue.
        """

        entry = self.entry_finder.pop(node)
        self.queue.remove(entry)
        #entry[-1] = self.REMOVED
        #raise NotImplementedError

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        count = next(self.counter)
        entry = [node[0],count,node[1]]
        self.entry_finder[node[1]] = entry
        hq.heappush(self.queue, entry)
        # TODO: finish this function!

    def find_weight(self,node):
        """
        Returns the weight of the node from entry finder
        """
        result = self.entry_finder[node]
        return result[0]
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    #print("srart bfs")
    def construct_path(path_list,child,key):
        child = list(child)
        for node_list in path_list:
            if node_list[-1] == key:
                temp_list = node_list
                #path_list.remove(node_list)#doubt
                path_list.append(temp_list+child)
                #print("construct path list",path_list)
                return path_list

    # TODO: finish this function!
    if start == goal:
        return []
    def return_path(path_list,key):
        final_list=[]
        for paths in path_list:
            if paths[-1] == key:
                final_list.append(paths)
        if len(final_list)==1:
            #print("1pathfinal",final_list[0])
            return final_list[0]
        elif len(final_list)>1:
            final_return_list = (min(final_list,key=len))
            #print("lenghr mire than two",final_return_list)
            return final_return_list
    count=0
    # node=dict()
    # node[start]=0
    solution=[]
    #print("1",start,goal)
    path = [[start]]
    frontier = PriorityQueue()
    frontier.append((0,start))
    explored=[]#next set use
    while True:
        if frontier.size()==0:
            #print("sdjus")
            return []#check later
        #print("inside while")
        temp_node = frontier.pop()

        # if temp_node[1] == goal:
        #
        #     return return_path(path,goal)
        explored.append(temp_node[1])
        # print("explored",explored)

        graph_sorted = sorted(graph[temp_node[1]])
        count = count+1
        #print(count,temp_node[1])

        for i in graph_sorted:
            # print("child",i)
            # print("frontier",i not in frontier)
            # print("explored",i not in explored)
            if (i not in frontier) and (i not in explored):
                # print("frontier",frontier)
                # print(i, explored)
                if i==goal:
                    # print("*************************************************")
                    # print("*************************************************")
                    # print("*************************************************")
                    # print("*************************************************")
                    # print(return_path(path,temp_node[1]),"goal")
                    #print("path",path)
                    return return_path(path,temp_node[1])+[goal]
                path = construct_path(path, i, temp_node[1])
                frontier.append((0,i))
        tempdel_path = return_path(path,key=temp_node[1])
        path.remove(tempdel_path)


    #raise NotImplementedError


def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """
    #print("###########################################################################")
    # TODO: finish this function!
    def calc_distance(ongoing_path):
        if len(ongoing_path)==1:
            return 0
        total_weight_from_start =0
        for each_node_index in range(len(ongoing_path) - 1):
            edge_weight = graph.get_edge_weight(ongoing_path[each_node_index], ongoing_path[each_node_index + 1])
            total_weight_from_start += edge_weight
        return total_weight_from_start
    #Constructing different paths my function is going and storing each of them in path variable
    def construct_path(path_list, child, key):
        child = list(child)
        for node_list in path_list:
            if node_list[-1] == key:
                temp_list = node_list
                # path_list.remove(node_list)#doubt
                path_list.append(temp_list + child)
        #print("construct path list",path_list)
        return path_list

        # TODO: finish this function!

    def return_path(path_list, key):
        final_list = []
        for paths in path_list:
            if paths[-1] == key:
                final_list.append(paths)
        if len(final_list) == 1:
            #print("1pathfinal",final_list[0])
            return final_list[0]
        elif len(final_list) > 1:
            final_return_list = sorted(final_list,key=calc_distance)
            #print("lenghr mire than two",final_return_list)
            return final_return_list[-1]

    def return_goal(path_list,key):
        final_list = []
        for paths in path_list:
            if paths[-1] == key:
                final_list.append(paths)
        if len(final_list) == 1:
            return final_list[0]
        elif len(final_list)>1:
            final_return_list = sorted(final_list, key=calc_distance)
            #print("answer is",final_return_list[0])
            return final_return_list[0]

    if start == goal:
        return []
    path = [[start]]
    frontier = PriorityQueue()
    frontier.append((0, start))
    explored = []  # next set use
    while True:
        if frontier.size() == 0:
            # print("sdjus")
            return []  # check later
        # print("inside while")
        temp_node = frontier.pop()

        if temp_node[1] == goal:
            return return_goal(path,goal)
        explored.append(temp_node[1])
        #print("explored", explored)
        children = graph[temp_node[1]]
        graph_sorted = children
        # count = count + 1
        # print(count,temp_node[1])

        for child in graph_sorted:
            # print("child",child)
            # print("frontier",child not in frontier)
            # print("explored",child not in explored)
            if (child not in frontier) and (child not in explored):
                # print("frontier",frontier)
                # print(child, explored)
                path = construct_path(path, child, temp_node[1])
                #print("path",path)
                ongoing_path = return_goal(path, key=child)
                total_weight_from_start = calc_distance(ongoing_path)
                # for each_node_index in range(len(ongoing_path)-1):
                #     edge_weight = graph.get_edge_weight(ongoing_path[each_node_index],ongoing_path[each_node_index+1])
                #     total_weight_from_start += edge_weight
                #edge_weight = graph.get_edge_weight(temp_node[1],child)
                frontier.append((total_weight_from_start, child))
                #print("frontier after",frontier)
            elif child in frontier:
                #print("child in elif",child)
                child_previous_weight=frontier.find_weight(child)
                #edge_weight = graph.get_edge_weight(temp_node[1],child)
                path = construct_path(path, child, temp_node[1])
                #print("elif path",path)
                ongoing_path = return_goal(path, key=child)
                total_weight_from_start = calc_distance(ongoing_path)
                # for each_node_index in range(len(ongoing_path) - 1):
                #     edge_weight = graph.get_edge_weight(ongoing_path[each_node_index],
                #                                         ongoing_path[each_node_index + 1])
                #     total_weight_from_start += edge_weight
                if total_weight_from_start<child_previous_weight:
                    frontier.remove(child)
                    frontier.append((total_weight_from_start,child))
                    #print("FRONTIER in elif",frontier)
        tempdel_path = return_path(path, key=temp_node[1])
        #print("removing path",tempdel_path)
        path.remove(tempdel_path)


    # raise NotImplementedError


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    raise NotImplementedError


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def bidirectional_ucs(graph, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic, landmarks=None):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.
        landmarks: Iterable containing landmarks pre-computed in compute_landmarks()
            Default: None

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Surya Teja Adluri"
    raise NotImplementedError


def compute_landmarks(graph):
    """
    Feel free to implement this method for computing landmarks. We will call
    tridirectional_upgraded() with the object returned from this function.

    Args:
        graph (ExplorableGraph): Undirected graph to search.

    Returns:
    List with not more than 4 computed landmarks. 
    """
    return None


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """
    pass


# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError


def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.nodes[v]["pos"][0]), math.radians(graph.nodes[v]["pos"][1]))
    goalLatLong = (math.radians(graph.nodes[goal]["pos"][0]), math.radians(graph.nodes[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
