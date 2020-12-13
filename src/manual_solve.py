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

"""
Student Name: Colm Brandon 
Student Number: 20236454
Github Repo: https://github.com/cbrandon20/ARC
"""

"""
Summary Reflection:
Numpy and Scipy were the only non-standard libraries used:
Scipy, the distance module was used for calculating the euclidian distance between coloured points in the matrix, to identify clusters
Numpy was used for matrix and vector manipulation

The approach that was taken to the problem, was viewing this a coordinate system from (0,0) to (grid_height, grid_width) and then to iterate
over column by column (this was somewhat counter-intuitive as the coordinates took the format of (y,x) rather than (x,y)),
it also lead to alot of nested loops which isn't the perfect ideal way to work with numpy, but it yielded adequate results.

First step in each of the solutions was to identify the background colour, in order to do this the assumption was made that the most frequently occuring colour was the background.
This assumption held true for each of the problems attempted here, but there would definitely be exceptions in some ARC problems, so a more sophisticated algorithm would be needed for a general solution.
Once the background was found the second common step was to find the clusters of coloured points. A cluster in this case was a set of non background coloured points that each have atleast one other 
non point in the set within a euclidean distance of sqrt(2). Each of the problems required some sort of "image" translation where it was reflecting across and x, scaling, etc.. 
and in the main they required mapping/translating between various cartesian coordinate systems. Finally when it came to generating the output matrix the same paradigm was used,
this was creating a "colour_map" a dictionary with key->[tuple(i,j)] value->[colour_number] pairs, a matrix with the correct output shape was generated, then iterated through, if the (i,j) coordinate was a keyin 
the colour map the value in the matrix was set to that keys value, if it wasn't it was set to zero.

6b9890af's unique challenge was distinguishing between the "shape" and the "frame", then seeting the frame to the global coordinate system and scaling the shape up 
(through a simple form of interpolation) to the correct proportions of the frame such that is its edges touch the interior of the frame. b775ac94's was breaking the clusters into sub clusters, 
finding the point on the full shape that was closest to each of the single colour points, the finding where the points were relative to eachoter 
This mapping was used-> (0: topleft, 1: topright, 2: bottomleft, 3: bottomright), this then guided which translation needed to be done on the orignal shape. 
Finally with 0a938d79, the unique challenge was finding which edges(top|bottom, left|right), the points were, then finding there orientation relative to eachother, which enable the generation of the new lines
with the correct displacement.

One observation was noticed during the undertaking of this assignment, when sandboxing solutions I began writing a function that compared the input matrixes to the output matrixes,
to give binary answers to questions such as does the background stay the same, does the matrix shape change, etc.. that could then guide the solve methods flow, and which of the transformation functions were used.
However it quickly became how many tests were required and in order to abstract some input->output feature mappings to a binary question were increasily complex depending on the task, and would sometimes require
several binary questions. This was for a set of 3 of the problems in ARC and Hand coding these tests was somewhat infeasible in the time frame and for this problem there was little to gain so was abandonded, although it was valuable because through the process I gained 
some insight the difficulty of this problem. Based on this anecdotal evidence coming up with a general AI solution for all the puzzles in ARC would seem incredible, without any human insight about the problem domains built into the system, 
I can't see it learning all the relationships with the limited sets of training given. Which would indicate that we're a long way from truly Abstract Reasoning through AI at this point.

"""

# get_background  and identify_clusters used by all solve_* functions

def get_background(x):
    ### The assumption is made that the most prevalent colour is the background
    #This gets all the unique colours and in the matrix and their counts
    x_palette, x_palette_counts = np.unique(x, return_counts=True)
    x_background = x_palette[np.argmax(x_palette_counts)]
    #returns the colour with the largest count value
    return x_background

def identify_clusters(x, b_colour):

    #Iterates recursively trough the the graph of overlapping clusters, generate is single tuple for for each set of clusters that need to be combined
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
    x,y = list(coord_dict.keys()),list(result.values())

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

    #This creates the final set of clusters, by combining the overlaping clusters and also adding the clusters that were correct from the start to the dict final_clusters
    for sl in y:
        temp_list = list()
        for l in sl:
            temp_list = temp_list + (coord_dict[l])
        temp_list = sorted(temp_list)
        final_clusters[cluster_index] = temp_list
        cluster_index += 1

    return final_clusters


"""
b775ac94 required transformations
Identify the clusters of non background colours
Each cluster contains a single colour which contains the full shape?? to be translated
There is then either 2 or 3 other single colured points joined to the full shape
Depending on the singles points position relative to the full shape it indicates a reflection 
of the full shape across the x or y axis or both, in the colour of the single point
Get single Colour Shape -> Copy -> Reflect Across Axis (based on adjoining coloured dots) -> Repeat for all coloured dots

This implemenation solves all the training and testing data correctly

"""

def solve_b775ac94(x):
    background = get_background(x)
    clu = identify_clusters(x, background)
    for k, v in clu.items():
        clu[k] = get_subsections(v, x)

    colour_map = {}

    #This dictionary is used in conjunction with the (0: topleft, 1: topright, 2: bottomleft, 3: bottomright) to apply the reflections across the appropriate axis
    #for instances if the translation is from 2: bottomleft -> 3: bottomright, k+k = 5 -> [0,1] a translation across the y axis
    axis_translating = {2: np.array([1, 0]), 4: np.array([1, 0]), 1: np.array([0, 1]), 5: np.array([0, 1]),
                        3: np.array([1, 1])}

    # Data structure for values in clu
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
                translated_coordinates = []
                #Iterates the through each point and calculates the number of steps to the origin
                #Trans_const sets the irrevelant steps to 0, i.e if translating across the x axis, it sets the y steps to zero
                for v_c in v[0][v[1]][0]:
                    steps = ((midpoint - np.array(v_c)) / 0.5) * tran_const
                    translated_coordinates.append(tuple((np.array(v_c) + steps).astype(int)))
                translated_subcluster[k_] = translated_coordinates
            else:
                #This is the orignal shape that does not need to be translated, is copied directly
                translated_subcluster[k_] = v[0][v[1]][0]
                pass

        #Flat maps the dictionary, with the keys being the coordinates and the value the colour at that coordinate
        for k, v in translated_subcluster.items():
            for v_ in v:
                colour_map[v_] = k



    #Creates the output matrix, with the correct shape but filled with ones
    #then fills in the colours correctly using the colour_map dictionary
    y = np.ones(shape=x.shape)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if (i, j) in colour_map.keys():
                y[i][j] = y[i][j] * colour_map[(i, j)]
            else:
                y[i][j] = background


    return y.astype(int)


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

    #These are translation mappings used for or in conjunction with the (0: topleft, 1: topright, 2: bottomleft, 3: bottomright)
    rel = {(1, 0): -2, (-1, 0): 2, (0, 1): -1, (0, -1): 1}
    rel_corners = {(1, 1): 3, (-1, 1): 1, (1, -1): 2, (-1, -1): 0}
    # Needed if there is no coloured point on the diagnol to the main shape
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
        sub_cluster[main_shape] = ([main_shape], pivot_point, if_no_foil[tuple([x[1] for x in relative_pos])])

    #Needed if there is no coloured point on the diagnol to the main shape
    for k, v in relative_pos:
        sub_cluster[k] = (sub_cluster[k], sub_cluster[main_shape][-1] + v)

    # Subcluster is a dictionary, it's keys are the colour, it's values are a tuple the first entry is the points for the subcluter,
    # The second is the subclusters location relative to the others 0: topleft, 1: topright, 2: bottomleft, 3: bottomright

    return (sub_cluster, main_shape)

"""
There is a red frame and a shape in each problem.
The red frame becomes the global coordinates and the shape needs to be interpolated to fully occupy the interior of the red frame
the goal is essentially zooming in on the shape, with the magnifcation being relative to the differece in size between the shape and the frame
Get Frame -> Get Shape -> Set output to Frame -> Scale Shape to fit Frame -> Center Shape in Frame

This implemenation solves all the training and testing data correctly
"""

def solve_6b9890af(x):
    background = get_background(x)
    clu = identify_clusters(x, background)
    frame, shape = None, None
    container_shape = []
    #Iterates over the clusters identifies which is the shape and which is the frame
    for k, v in clu.items():
        container_shape.append(extract_frame(v, x))
    #
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
    #This colours in the frame, as it was added as 2 rows, 2 columns of 1's using np's hstack and vstack
    x = colour_frame(frame_cluster, frame_og_co, Y, x)
    return x.astype(int)

def extract_frame(cluster, x):
    #Specific to 6b9890af it decides which cluster is the shape and which is the red frame
    cluster_ = np.array(cluster)
    #Gets the extrema's in the cluster
    i_min, i_max = np.min(cluster_[:, 0]), np.max(cluster_[:, 0])
    j_min, j_max = np.min(cluster_[:, 1]), np.max(cluster_[:, 1])

    #This is set of corners for square/rectangular space that contain the cluster
    corners = [(i_min, j_min), (i_min, j_max), (i_max, j_min), (i_max, j_max)]
    non_edge_points = 0

    ##Checks to see if the cluster contains al four corners of the space it contains
    four_corners = all([True if c in cluster else False for c in corners])
    #Checks to see if there are any points that aren't on the edge
    for p in cluster_:
        if not (i_min in p or i_max in p or j_min in p or j_max in p):
            non_edge_points += 1

    #It finds either a space or a frame and returns a tuple of format (FLAG, the_translated corner points, the translated cluster_points, original cluster points)
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
    #Gets to top left and bottom right corners of the frame
    for i in range(len(frame)):
        if i % 3 == 0:
            frame[i] = frame[i] + 1 - (2 * (i % 2))
            frame_min_max.append(frame[i])
    #Translates them such the the top left corner of the frame is now the origin
    frame_min_max = frame_min_max - frame_min_max[0]
    frame_min_max = np.array(frame_min_max)
    shape = np.array(shape)
    #return a tuple of the scale up required for each axis and the dimensions of the frame (+1 to adjust for zero indexing)
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

"""
There are two single dots of different colours attached to either the x edges or y edges, 
the goal is to draw a line from the single dots across to the opposite edge
Then repeat the lines pattern until the end of the grid is met
Points -> Lines -> Copy Lines -> Translate & Repeat 

This implemenation solves all the training and testing data correctly
"""

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


def map_to_colour(clusters,x):
    colour_as_key = {}
    for k,v in clusters.items():
        colour_as_key[x[v[0][0]][v[0][1]]] = v
    return colour_as_key


def transfrom_input_clusters_to_colour_map(clusters, x_edges, y_edges):
    #Some loops etc,, where they're not strictly necessary as there's only every two points, however it was written in this manner
    #so that it could easily be adapted to m points

    #Takes in the clusters and the x,y edges
    coordinate_to_compare = {}
    single_lines = {}
    numberoflines = []
    #finds if the points are or y edges, depending on which edge the point is on
    for k, v in clusters.items():
        line = []
        for v_ in v:
            print(v_)
            if v_[1] in x_edges:
                #ON X Edge
                numberoflines.append(y_edges[-1])
                coordinate_to_compare[k] = [v_[0], 0]
                # fix Y and generates points required to make line to opposite edge
                for i in range(x_edges[1] + 1):
                    line.append(np.array([v_[0], i]))
            elif v_[0] in y_edges:
                #On Y
                numberoflines.append(x_edges[-1])
                coordinate_to_compare[k] = [0, v_[1]]
                # fix X and generates points required to make line to opposite edge
                for i in range(y_edges[1] + 1):
                    line.append(np.array([i, v_[1]]))
        single_lines[k] = line

    #Stores the lines in the dict single_lines their key is their colour

    #Finds which point is closest to the origin
    distance_from_origin = 9999
    closest_to_origin = None
    for k, v in coordinate_to_compare.items():
        dist = np.sum(np.array(v))
        if dist < distance_from_origin:
            closest_to_origin = k
            distance_from_origin = dist

    #Finds the distance between the two lines in the (y,x)
    space_between_lines = None
    for k, v in coordinate_to_compare.items():
        if k != closest_to_origin:
            space_between_lines = np.array(v) - np.array(coordinate_to_compare[closest_to_origin])

    #Geneates the repeating patterns of lines
    final_lines = {}
    counter = 0
    for k, v in single_lines.items():
        points = []
        its = int((numberoflines[counter] - np.sum(v[0])) / (np.sum(space_between_lines)))
        for i in range(its):
            temp = v + space_between_lines * 2 * i
            [points.append(tuple(t)) for t in temp]
            final_lines[k] = points

    #Creates a dictionary of key (coordinate) value (colour) pairs used to create the ouput matrix
    colour_map = {}
    for k, v in final_lines.items():
        for v_ in v:
            colour_map[v_] = k

    return colour_map




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

