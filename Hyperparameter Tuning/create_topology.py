# First networkx library is imported 
# along with matplotlib
import networkx as nx
import matplotlib.pyplot as plt
   
  
# Defining a Class
class GraphVisualization:
   
    def __init__(self):
          
        # visual is a list which stores all 
        # the set of edges that constitutes a
        # graph
        self.visual = []
          
    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)
          
    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list
    # nx.draw_networkx(G) - plots the graph
    # plt.show() - displays the graph
    def visualize(self):
        G = nx.Graph()
        G.add_edges_from(self.visual)
        nx.draw_networkx(G, node_size=200, node_color="skyblue", pos=nx.kamada_kawai_layout(G))
        plt.show()


# Create Node Class
class TreeNode(object):
    # Function for adding children and saving node's value
    def __init__(self, data):
        self.data = data
        self.distance = []
        self.children = []

    def add_child(self, obj, distance):
        self.children.append(obj)
        self.distance.append(distance)


# Create Tree Class
class Tree:
    # Function for checking if a node is a leaf node
    def __init__(self):
        self.root = TreeNode('B1')
        
    def isleaf(self, node):
        if node is self.root:
            return False
        elif not node.children:
            return True
        else:
            return False


# Function for finding any leaf node from a start node        
def find_leaves(arr, node):
    
    if len(node.children) == 0:
        arr.append(node)
    else:
        for n in node.children:
            find_leaves(arr, n)


def give_leaves(node):
    arr = []
    find_leaves(arr, node)
    return arr 


# Function for finding a path from root to any given node      
def has_path(root, arr, node):

    if not root:
        return False
    else:
        arr.append(root)    
        
        if root.data == node.data:
            return True
        
        elif any(has_path(n, arr, node) for n in root.children): 
            return True
        else:
            arr.pop(-1)
            return False


def give_path(root, node):
    arr = []
    path_distance = [0.0]

    if has_path(root, arr, node):
        for idx in range(len(arr)-1):
            for child in range(len(arr[idx].children)):
                if arr[idx].children[child] == arr[idx+1]:
                    path_distance.append(path_distance[-1] + arr[idx].distance[child])
        return arr, path_distance
    else:
        return [], []


# Initialize grid with our given topology of our problem            
def create_grid():   

    # grid_length = [[Feeder it belongs to, Length in meters of branch] for each branch]
    grid_length = [[1, 45.], [1, 20.], [1, 15.], [1, 30.], [2, 40.], [2, 20.], [2, 45.], [3, 32.5], [3, 40.]]
    
    # Create Topology
    nodes = []
    tr = Tree()
    nodes.append(tr.root)
    for i in range(1, 33):
        name = "B"+str(i+1)
        nodes.append(TreeNode(name))
        
    nodes[0].add_child(nodes[1], 2.5)
    nodes[0].add_child(nodes[2], 2.5)
    nodes[0].add_child(nodes[3], 2.5)
    nodes[1].add_child(nodes[4], 5.0)
    nodes[2].add_child(nodes[5], 5.0)
    nodes[2].add_child(nodes[6], 2.5)
    nodes[3].add_child(nodes[7], 12.5)
    nodes[4].add_child(nodes[8], 7.5)
    nodes[4].add_child(nodes[9], 12.5)
    nodes[4].add_child(nodes[10], 5.0)
    nodes[5].add_child(nodes[11], 5.0)
    nodes[6].add_child(nodes[12], 12.5)
    nodes[7].add_child(nodes[13], 5.0)
    nodes[7].add_child(nodes[14], 2.5)
    nodes[8].add_child(nodes[15], 5.0)
    nodes[10].add_child(nodes[16], 2.5)
    nodes[10].add_child(nodes[17], 7.5)
    nodes[11].add_child(nodes[18], 5.0)
    nodes[12].add_child(nodes[19], 2.5)
    nodes[12].add_child(nodes[20], 7.5)
    nodes[13].add_child(nodes[21], 12.5)
    nodes[14].add_child(nodes[22], 5.0)
    nodes[15].add_child(nodes[23], 12.5)
    nodes[17].add_child(nodes[24], 10.0)
    nodes[18].add_child(nodes[25], 10.0)
    nodes[20].add_child(nodes[26], 5.0)
    nodes[22].add_child(nodes[27], 5.0)
    nodes[23].add_child(nodes[28], 12.5)
    nodes[25].add_child(nodes[29], 12.5)
    nodes[26].add_child(nodes[30], 5.0)
    nodes[27].add_child(nodes[31], 12.5)
    nodes[30].add_child(nodes[32], 10.0)
    
    # Driver code
    G = GraphVisualization()
    for node in nodes:
        parent = int(node.data[1:])
        children = node.children
        for idx in children:
            child = int(idx.data[1:])
            G.addEdge(parent, child)
    # G.visualize()
      
    return tr, nodes, grid_length
