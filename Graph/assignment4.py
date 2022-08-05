# Mohammad Bayat
#Partners: Sadia Ishak, Tanvir Khan

import collections
import heapq

import random
import sys
import time
import pandas as pd
from datetime import datetime
import timeit as tp
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


def read_graph(file_name):
    with open(file_name, 'r') as file:
        graph = []
        lines = file.readlines()
        for line in lines:
            costs = line.split(' ')
            row = []
            for cost in costs:
                row.append(int(cost))
            graph.append(row)
        return graph








def desc_graph(graph):
    number_of_vertices = len(graph)
    message = ''
    message += 'Number of vertices= ' + str(number_of_vertices) + '\n'
    non_zero = 0
    for i in range(number_of_vertices):
        for j in range(number_of_vertices):
            if graph[i][j] > 0:
                non_zero += 1
    num_edges = (non_zero / 2)
    message += 'Number of edges= ' + str(num_edges) + '\n'
    return message


def is_symmetric(graph):
    number_of_vertices = len(graph)
    for i in range(number_of_vertices):
        for j in range(number_of_vertices):
            if graph[i][j] != graph[j][i]:
                return False
    return True


def print_graph(graph, sep=' '):  # separator
    str_graph = ''
    for row in range(len(graph)):
        str_graph += sep.join([str(c) for c in graph[row]]) + '\n'  # convert the number to string and combine them
    return str_graph


def analyze_graph(file_name):
    graph = read_graph(file_name)
    output_file_name = file_name[0:-4 + len(file_name)] + '_report.txt'
    with open(output_file_name, 'w') as output_file:
        output_file.writelines('Analysis of graph ' + file_name + '\n')
        str_graph = print_graph(graph)
        symmetry=is_symmetric(graph)
        output_file.write(str_graph + '\n')
        output_file.write(str_graph)
        graph_description = desc_graph(graph)
        output_file.write(graph_description)
        output_file.write(graph_description)
        output_file.write('Symmetry= '+ str(symmetry) + '\n')
        dfs_traversal = dfs(graph)
        bfs_traversal = bfs(graph)
        prim_traversal, weight = prim_algorithm(graph)
        kruskal_graph = make_tuple(graph)
        kruskal_traversal, kruskal_min_cost = KruskalMST(kruskal_graph)
        floyd_distance,floyd_parent = floydWarshall(graph)
        dijkstra_distance = []
        dijkstra_parent = []
        for i in range(len(graph)):
            dijkstra_p, dijkstra_dis = dijkstra(graph, i)
            dijkstra_distance.append(dijkstra_dis)
            dijkstra_parent.append(dijkstra_p)
        str_graph_parent = print_graph(dijkstra_parent)
        str_graph_distance = print_graph(dijkstra_distance)
        str_graph_floyd_distance = print_graph(floyd_distance)
        str_floyd_parent=print_graph(floyd_parent)
       # file output
        output_file.write('DFS Traversal: ' + str(dfs_traversal) + '\n')
        output_file.write('BfS Traversal: ' + str(bfs_traversal) + '\n')
        output_file.write('Prim Traversal: ' + str(prim_traversal) + ',' + 'total cost: ' + str(weight) + '\n')
        output_file.write('Kruskal Traversal: ' + str(kruskal_traversal) + ',' + 'total cost: ' + str(kruskal_min_cost) + '\n')
        output_file.write('\n' + 'Dijkstra 2D Distance Matrix Representation: ' + '\n' + str(str_graph_distance) + '\n')
        output_file.write('\n' + 'Dijkstra Pred 2D Matrix Representation: ' + '\n' + str(str_graph_parent))
        output_file.write('\n' + 'Floyd 2D Distance Matrix Representation:' + '\n' + str(str_graph_floyd_distance))
        output_file.write('\n' + 'Floyd Pred 2D Matrix Representation: ' + '\n' + str(str_floyd_parent))

        # #console output
        print('Analysis of graph ' + file_name + '\n')
        print(str_graph + '\n')
        print(str_graph)
        print(graph_description)
        print('Symmetry= ' + str(symmetry) + '\n')
        print('DFS Traversal: ' + str(dfs_traversal))
        print('BfS Traversal: ' + str(bfs_traversal))
        print('Prim Traversal: ' + str(prim_traversal) + ',' + 'total cost: ' + str(weight))
        print('Kruskal Traversal: ' + str(kruskal_traversal) +  ',' + 'total cost: ' + str(kruskal_min_cost) + '\n')
        print('Dijkstra 2D Distance Matrix Representation: ' + '\n' + str(str_graph_distance) + '\n')
        print('Dijkstra Pred 2D Matrix Representation: ' + '\n' + str_graph_parent)
        print('Floyd 2D Distance Matrix Representation:' + '\n' + str(str_graph_floyd_distance))
        print('Floyd Pred 2D Matrix Representation: ' + '\n' + str(str_floyd_parent))


# https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/
def dfs_util(graph, v, visited):
    if v<len(graph):
        visited.append(v)
    for col in range(len(graph[v])):
        if graph[v][col] > 0 and col not in visited:
            dfs_util(graph, col, visited)


# https://www.geeksforgeeks.org/depth-first-search-or-dfs-for-a-graph/
def dfs(graph):
    visited = []
    dfs_util(graph, 0, visited)
    return visited


queue = []  # Initialize a queue


# https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
def bfs_util(visited, graph, node):
    visited.append(node)
    queue.append(node)
    while queue:
        s = queue.pop(0)
        for neighbour in range(len(graph)):
            if neighbour not in visited:
                visited.append(neighbour)
                queue.append(neighbour)


# https://www.geeksforgeeks.org/breadth-first-search-or-bfs-for-a-graph/
def bfs(graph):
    visited = []
    bfs_util(visited, graph, 0)
    return visited


# part of prim
def minKey(graph, key, mstSet):
    # Initialize min value
    min = sys.maxsize
    for v in range(len(graph)):
        if key[v] < min and mstSet[v] == False:
            min = key[v]
            min_index = v
    return min_index


# https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
def prim(graph, parent):
    st_tree = []
    total_cost = 0
    for i in range(1, len(graph)):
        tuple = (parent[i], i)
        total_cost += graph[i][parent[i]]
        st_tree.append(tuple)
    return [st_tree, total_cost]





# https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
def prim_algorithm(graph):
    key = [sys.maxsize] * len(graph)
    parent = [None] * len(graph)
    key[0] = 0
    mstSet = [False] * len(graph)
    parent[0] = -1
    for cout in range(len(graph)):
        u = minKey(graph, key, mstSet)
        mstSet[u] = True
        for v in range(len(graph)):
            if graph[u][v] > 0 and mstSet[v] == False and key[v] > graph[u][v]:
                key[v] = graph[u][v]
                parent[v] = u
    tree, w = prim(graph, parent)
    return [tree, w]


# part of prim
def prim_parent(graph):
    key = [sys.maxsize] * len(graph)
    parent = [None] * len(graph)
    key[0] = 0
    mstSet = [False] * len(graph)
    parent[0] = -1
    for cout in range(len(graph)):
        u = minKey(graph, key, mstSet)
        mstSet[u] = True
        for v in range(len(graph)):
            if graph[u][v] > 0 and mstSet[v] == False and key[v] > graph[u][v]:
                key[v] = graph[u][v]
                parent[v] = u
    return parent


# Part of prim
def make_tuple(graph):
    collect = []
    collect2 = []
    V = len(graph)
    for i in range(V):
        for j in range(V):
            weight = graph[i][j]
            if weight != 0:
                tuple = (i, j, weight)
                collect.append(tuple)
    collect.sort(key=lambda x: x[2])  # (0,1,4) ........
    for tuple in collect:
        List = list(tuple)  # (0,1,4) => [0,1,4]=Individual set
        collect2.append(List)
    return collect


# part of kruskal
def find(parent, i):
    if parent[i] == i:
        return i
    return find(parent, parent[i])


# part of kruskal
def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)
    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1


# https://www.geeksforgeeks.org/kruskals-algorithm-simple-implementation-for-adjacency-matrix/
def KruskalMST(graph):
    tree_mst = []
    V = len(graph)
    result = []
    i = 0
    e = 0
    parent = []
    rank = []
    for node in range(V):
        parent.append(node)
        rank.append(0)
    while e < V - 1 and i < V:
        u, v, w = graph[i]
        i = i + 1
        x = find(parent, u)
        y = find(parent, v)
        if x != y:
            e = e + 1
            result.append([u, v, w])
            union(parent, rank, x, y)
    minimumCost = 0
    for u, v, weight in result:
        minimumCost += weight
        t = (u, v)
        tree_mst.append(t)
    return [tree_mst, minimumCost]



##https://www.geeksforgeeks.org/python-program-for-dijkstras-shortest-path-algorithm-greedy-algo-7/
pred_array = []
def dijkstra(graph, src):
    row = len(graph)
    col = len(graph[0])
    dist = [float("Inf")] * row
    pred_array = [0] * row  # pred array
    dist[src] = 0
    queue = []
    for i in range(row):
        queue.append(i)
    while queue:
        u = minDistance_util(graph, dist, queue)
        queue.remove(u)
        for i in range(col):
            if graph[u][i] and i in queue:
                if dist[u] + graph[u][i] < dist[i]:
                    dist[i] = dist[u] + graph[u][i]
                    pred_array[i] = u
    return [pred_array, dist]


# part of djikastra
def minDistance_util(graph, dist, queue):
    minimum = float("Inf")
    min_index = -1
    for i in range(len(dist)):
        if dist[i] < minimum and i in queue:
            minimum = dist[i]
            min_index = i
    return min_index





container = []
# https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/

def floydWarshall(graph):
    w=len(graph)
    h=len(graph)
    dist = [[0 for x in range(w)] for y in range(h)]
    parent = [[0 for x in range(w)] for y in range(h)]
    for i in range(len(graph)):
        for j in range(len(graph)):
            dist[i][j] = graph[i][j]
            parent[i][j] = i
            if dist[i][j] == 0:
                dist[i][j] = sys.maxsize
        dist[i][i] = 0
        parent[i][i] = -1
    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                if dist[i][j] > min(dist[i][j], dist[i][k] + dist[k][j]):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
                    parent[i][j] = parent[k][j]
    return [dist,parent]




def main():
    mypath = "C:\\Users\\USER\\PycharmProjects\\Assignment4"  # might need to change this if not work
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for file in files:
        if file[0:5] == 'graph' and file.find('_report') < 0:
            analyze_graph(file)


if __name__ == '__main__':
    main()
