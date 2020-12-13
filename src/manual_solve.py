#!/usr/bin/python

from collections import defaultdict
import os.path
import os, sys
import json
import numpy as np
import re
from scipy.spatial import distance

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.


def get_background(x):
    ### The assumption is made that the most prevalent colour is the background
    x_palette, x_palette_counts = np.unique(x, return_counts=True)
    x_background = x_palette[np.argmax(x_palette_counts)]
    # print(x_clusters.values())
    return x_background

def identify_clusters(x, b_colour):
    # a cluster is a set of non background coloured points that each have another non background colourd pointed within sqrt(2)

    def dfs(adj_list, visited, vertex, result, key):
        visited.add(vertex)
        result[key].append(vertex)
        for neighbor in adj_list[vertex]:
            if neighbor not in visited:
                dfs(adj_list, visited, neighbor, result, key)

    #Scans the image column by column left to right, if it encounters a non background colour,
    #The first one it encounters it it add the coordinates to a dict under the key (cluster_id) 0
    #From then on it checks to see if any nonbackground coloured point encountered's euclidian distance is < sqrt of 2
    #If it is, it adds appends the coordinates to the last under the cluster_id, if not it creates a new cluster
    coord_dict = {}
    cluster_no = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] != b_colour:
                if len(coord_dict) == 0:
                    coord_dict[cluster_no] = list()
                    coord_dict[cluster_no].append(tuple([i, j]))
                    cluster_no += 1
                else:
                    in_a_cluster = False
                    for cd in coord_dict.keys():
                        if len([distance.euclidean(coord, (i, j)) for coord in coord_dict[cd] if
                                distance.euclidean(coord, (i, j)) <= np.sqrt(2)]) > 0:
                            coord_dict[cd].append(tuple([i, j]))
                            in_a_cluster = True
                            break
                    if not in_a_cluster:
                        coord_dict[cluster_no] = list()
                        coord_dict[cluster_no].append(tuple([i, j]))
                        cluster_no += 1
    # As the sytems scan row by row or column by column, it's guaranteed to cluster U shapres or similar seperately
    # There it it's necessary to find overlaps between the first iteration of clusters.
    refine_cluster = {}
    for i in range(len(coord_dict.values()) - 1):
        for j in range(i + 1, len(coord_dict.values())):
            distance_list = [len([distance.euclidean(coord, coord_) for coord_ in (list(coord_dict.values())[j]) if
                                  distance.euclidean(coord, coord_) <= np.sqrt(2)]) for coord in
                             (list(coord_dict.values())[i])]
            distance_list = [d for d in distance_list if d > 0]
            if len(distance_list) > 0:
                refine_cluster[(i, j)] = len(distance_list)

    #The previous step finds overlaps between found clusters
    #This step groups together clusters that all share overlaps
    edges = list(refine_cluster.keys())
    adj_list = defaultdict(list)
    for x, y in edges:
        adj_list[x].append(y)
        adj_list[y].append(x)
    result = defaultdict(list)
    visited = set()
    for vertex in adj_list:
        if vertex not in visited:
            dfs(adj_list, visited, vertex, result, vertex)
    x = list(coord_dict.keys())
    y = list(result.values())
    #The clusters that had no errors are now added to the new clusters that were created in the previous step
    #and subsequently returned.
    for v in x:
        included = False
        for xt in y:
            if v in xt:
                included = True
        if not included:
            y.append([v])
    final_clusters = {}
    cluster_index = 0
    for sl in y:
        temp_list = list()
        for l in sl:
            temp_list = temp_list + (coord_dict[l])
        temp_list = sorted(temp_list)
        final_clusters[cluster_index] = temp_list
        cluster_index += 1

    return final_clusters


def extract_frame(cluster, x):
    #Specific to 6b9890af it decides which cluster is the shape and which is the red frame
    cluster_ = np.array(cluster)
    i_min, i_max = np.min(cluster_[:, 0]), np.max(cluster_[:, 0])
    j_min, j_max = np.min(cluster_[:, 1]), np.max(cluster_[:, 1])
    corners = [(i_min, j_min), (i_min, j_max), (i_max, j_min), (i_max, j_max)]
    non_edge_points = 0
    four_corners = all([True if c in cluster else False for c in corners])
    for p in cluster_:
        if not (i_min in p or i_max in p or j_min in p or j_max in p):
            non_edge_points += 1

    if non_edge_points == 0 and four_corners:
        #"Frame detected"
        return (True, [(i - i_min, j - j_min) for i, j in corners], (cluster - np.array(corners[0])), cluster)
    else:
        #"No Frame Detected")
        return (False, [(i - i_min, j - j_min) for i, j in corners], (cluster - np.array(corners[0])), cluster)


def get_scale(frame, shape):
    #This see what the scaling is required for changing from the input coordinate system to the output coordinate system
    frame = np.array(frame)
    frame_min_max = []
    for i in range(len(frame)):
        if i % 3 == 0:
            frame[i] = frame[i] + 1 - (2 * (i % 2))
            frame_min_max.append(frame[i])

    frame_min_max = frame_min_max - frame_min_max[0]
    frame_min_max = np.array(frame_min_max)
    shape = np.array(shape)
    return (frame_min_max[-1] + 1) / (shape[-1] + 1), tuple(frame_min_max[-1] + 1)


def matrix_scaler(scale, shape,shape_cluster, original_coordinates, background, X):
    #Up to now, the shape is just viewed as a non-background colour where as it has a colour in the original mapping
    #This mataches the orignal cluster coordinates there colours, it then creates a shape matrix in the new coordinate system
    #With the correct colors. In 6b9890af there's only a single colur in a cluster, however the function was written in this manner
    #So it could be recycled for scaling multicolour shapes if need,
    colour_map = {}
    for trans, co in zip(shape_cluster, original_coordinates):
        colour_map[tuple(trans)] = X[co[0]][co[1]]
    y = np.ones(shape=(shape[-1][0] + 1, shape[-1][1] + 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if (i, j) in colour_map.keys():
                y[i][j] = y[i][j] * colour_map[i, j]
            else:
                y[i][j] = y[i][j] * background

    #This scales the shape up to the transformed size
    y_ = np.ones(shape=(int(scale[-1][0]), int(scale[-1][1])))
    for i in range(y_.shape[0]):
        for j in range(y_.shape[1]):
            y_[i][j] = y_[i][j] * y[(int(i / scale[0][0]))][(int(j / scale[0][1]))]

    # Adding the frame
    n, m = y_.shape
    X0 = np.ones((n, 1))
    y_ = np.hstack((X0, y_, X0))
    n, m = y_.shape
    X1 = np.ones((1, m))
    y_ = np.vstack((X1, y_, X1))
    return y_


def colour_frame(frame_cluster, original_coordinates, Y, X):
    #This simpy colours in the frame, once again the frame is all red but this was written to be a general solution
    colour_map = {}
    for trans, co in zip(frame_cluster, original_coordinates):
        # print(trans,co)
        colour_map[tuple(trans)] = X[co[0]][co[1]]
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if (i, j) in colour_map.keys():
                Y[i][j] = Y[i][j] * colour_map[i, j]

    return Y


def get_subsections(cluster, x):
    sub_cluster = {}
    for v in cluster:
        if x[v[0]][v[1]] in sub_cluster.keys():
            sub_cluster[x[v[0]][v[1]]].append(v)
        else:
            sub_cluster[x[v[0]][v[1]]] = list()
            sub_cluster[x[v[0]][v[1]]].append(v)
    main_shape = None
    for k, v in sub_cluster.items():
        if len(v) > 1:
            main_shape = k
            break
    min_distance = 100
    pivot_point = None
    for k, v in sub_cluster.items():
        if k != main_shape:
            min_distance = 100
            for v_ in sub_cluster[main_shape]:
                if distance.euclidean(v, v_) < min_distance:
                    min_distance = distance.euclidean(v, v_)
                    pivot_point = v_

    rel = {(1, 0): -2, (-1, 0): 2, (0, 1): -1, (0, -1): 1}
    rel_corners = {(1, 1): 3, (-1, 1): 1, (1, -1): 2, (-1, -1): 0}
    if_no_foil = {(-1): 0, (1): 1, (-2): 0, (2): 2, (-2, 1): 2, (2, 1): 0, (-2, -1): 1, (2, 1): 3}

    relative_pos = []
    pivot_has_foil = False
    for k, v in sub_cluster.items():
        if k != main_shape:

            rel_pos = tuple(np.array(pivot_point) - np.array(v[0]))
            if rel_pos in rel_corners.keys():

                sub_cluster[main_shape] = (sub_cluster[main_shape], pivot_point, rel_corners[rel_pos])
                rel_pos_ = tuple(np.array(v[0]) - np.array(pivot_point))
                sub_cluster[k] = (v, rel_corners[rel_pos_])

                pivot_has_foil = True
            else:
                relative_pos.append((k, rel[rel_pos]))
    if not pivot_has_foil:
        sub_cluster[main_shape] = (
        sub_cluster[main_shape], pivot_point, if_no_foil[tuple([x[1] for x in relative_pos])])

    for k, v in relative_pos:
        sub_cluster[k] = (sub_cluster[k], sub_cluster[main_shape][-1] + v)

    # Subcluster is a dictionary, it's keys are the colour, it's values are a tuple the first entry is the points for the subcluter,
    # The second is the subclusters location relative to the others 0: topleft, 1: topright, 2: bottomleft, 3: bottomright

    return (sub_cluster, main_shape)


def map_to_colour(clusters,x):
    colour_as_key = {}
    for k,v in clusters.items():
        colour_as_key[x[v[0][0]][v[0][1]]] = v
    return colour_as_key


def transfrom_input_clusters_to_colour_map(clusters, x_edges, y_edges):
    coordinate_to_compare = {}
    single_lines = {}
    for k, v in clusters.items():
        line = []
        for v_ in v:
            print(v_)
            if v_[1] in x_edges:
                #ON X Edge
                coordinate_to_compare[k] = [v_[0], 0]
                # fix Y and generate points
                for i in range(x_edges[1] + 1):
                    line.append(np.array([v_[0], i]))
            elif v_[0] in y_edges:
                #On Y Edge
                coordinate_to_compare[k] = [0, v_[1]]
                # fix X and generate points
                for i in range(y_edges[1] + 1):
                    line.append(np.array([i, v_[1]]))
        single_lines[k] = line

    distance_from_origin = 9999
    closest_to_origin = None
    for k, v in coordinate_to_compare.items():
        dist = np.sum(np.array(v))
        if dist < distance_from_origin:
            closest_to_origin = k
            distance_from_origin = dist

    space_between_lines = None
    for k, v in coordinate_to_compare.items():
        if k != closest_to_origin:
            space_between_lines = np.array(v) - np.array(coordinate_to_compare[closest_to_origin])

    final_lines = {}
    for k, v in single_lines.items():
        points = []
        for i in range(5):
            temp = v + space_between_lines * 2 * i
            [points.append(tuple(t)) for t in temp]
            final_lines[k] = points

    colour_map = {}
    for k, v in final_lines.items():
        for v_ in v:
            colour_map[v_] = k

    return colour_map








def solve_b775ac94(x):
    background = get_background(x)
    clu = identify_clusters(x, background)
    for k, v in clu.items():
        clu[k] = get_subsections(v, x)

    colour_map = {}
    background_colour = 0
    X_ = x
    axis_translating = {2: np.array([1, 0]), 4: np.array([1, 0]), 1: np.array([0, 1]), 5: np.array([0, 1]),
                        3: np.array([1, 1])}
    # Data structure for values on cluster
    # First is which cluster
    # Second is Tuple, index 0 is Dict of subclusters, index 1 is the key for the main sub cluster to be reflected
    # Is the key for the subclusters (key is their colour)
    # value index 0 is a list of the subcluster points, index -1 it's position relative to origin, for main sub cluster index 1 = the pivot points
    for k, v in clu.items():
        translated_subcluster = {}
        for k_, v_ in v[0].items():
            if k_ != v[1]:
                tran_const = axis_translating[v_[-1] + v[0][v[1]][-1]]
                # Get the midpoint between the too pivot points
                # get the distance between the points and the midpoint
                # divide them by 0.5 which is the step size, this gives the translation
                midpoint = (np.array(v[0][v[1]][1]) + v_[0][0]) / 2
                # print(midpoint,tran_const)
                translated_coordinates = []
                for v_c in v[0][v[1]][0]:
                    steps = ((midpoint - np.array(v_c)) / 0.5) * tran_const
                    translated_coordinates.append(tuple((np.array(v_c) + steps).astype(int)))
                translated_subcluster[k_] = translated_coordinates
            else:
                translated_subcluster[k_] = v[0][v[1]][0]
                pass

        for k, v in translated_subcluster.items():
            for v_ in v:
                colour_map[v_] = k
        # print(colour_map)
    y = np.ones(shape=X_.shape)

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if (i, j) in colour_map.keys():
                y[i][j] = y[i][j] * colour_map[(i, j)]
            else:
                y[i][j] = background_colour


    return y.astype(int)

def solve_6b9890af(x):
    background = get_background(x)
    clu = identify_clusters(x, background)
    frame, shape = None, None
    container_shape = []
    for k, v in clu.items():
        container_shape.append(extract_frame(v, x))
    for k, v, c_, og in container_shape:
        if k == True:
            frame = v
            frame_cluster = c_
            frame_og_co = og
        else:
            shape = v
            original_coordinates = og
            shape_cluster = c_
    ##This returns the shape with a frame, the final step is to colour in the frame
    Y = matrix_scaler(get_scale(frame, shape), shape, shape_cluster, original_coordinates, 0, x)
    x = colour_frame(frame_cluster, frame_og_co, Y, x)
    return x.astype(int)

def solve_0a938d79(x):
    far_edge = np.array(x.shape) - 1
    y_edges = (0, far_edge[0])
    x_edges = (0, far_edge[1])

    clu = identify_clusters(x, 0)
    clu = map_to_colour(clu, x)
    colour_map = transfrom_input_clusters_to_colour_map(clu, x_edges, y_edges)

    y = np.ones(shape=x.shape)

    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if (i, j) in colour_map.keys():
                y[i][j] = y[i][j] * colour_map[(i, j)]
            else:
                y[i][j] = 0

    return y.astype(int)


def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()

