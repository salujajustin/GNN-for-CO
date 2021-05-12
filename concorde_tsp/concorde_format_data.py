import pickle, sys, os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_tsp(ax1, instance, solution, dist, heatmap=None, mask=None, markersize=5, title="TSP"):
    """Plots the tsp route on matplotlib axis ax1.

    Args:
        ax1: A matplotlib Axes object
        instance: A list of lists that contains the number of nodes and coordinates of each node
        solution: A list of indices indicating the ordered sequence of indices to be traveled from the instance 

    Returns:
        total_dist: The calculated total distance traveled from the TSP closed tour
    """
    
    #  Locations of all nodes in the instance
    loc = np.array(instance)

    #  Find the route by ordering loc by the tour
    tour = np.array(solution)
    
    #  Want the ordered seqence of the tour
    routes = [loc[tour[r]].tolist() for r in range(len(loc))]
    print(routes)
    routes = np.array(routes)

    xs, ys = routes.transpose()


    #  Add plot arrows
    qv = ax1.quiver(
        xs[:-1],
        ys[:-1],
        xs[1:] - xs[:-1],
        ys[1:] - ys[:-1],
        scale_units='xy',
        angles='xy',
        scale=1,
    )

    ax1.plot(xs, ys, 'o')
    ax1.set_title('{}, {} nodes, total distance {:.3f}'.format(title, len(instance), dist))
    plt.show()

    return ax1

def calculate_edges(G):
  ''' generates edges between all unique pairs of nodes weighted by euclidean distance
    Input:
      G - (nx2 array) 2D spatial distribution of n nodes
    Output:
      weights - (nxn array) euclidean distance matrix

  '''
  n = len(G)
  weights = np.zeros((n, n))
  for i in range(len(G)):
    for j in range(len(G)):
      if i > j:
        eucl_dist = (np.linalg.norm(G[i]-G[j]))
        weights[i][j] = eucl_dist
        weights[j][i] = eucl_dist
  return weights

def read_concorde_sol(file, dist):
  ''' reads a .sol file outputted by concorde, and returns the optimal tour and its distance
    Input:
      file - string .sol file
      dist - 2D distance matrix of the graph
    Output:
      tour - (n array) optimal tour
      total_dist - (float) optimal tour length
  '''
  with open(file, "r") as f:
    data = f.read().splitlines()
  tour = data[1:]
  tour = np.array([x.split(" ")[:-1] for x in tour]).flatten().astype(int)
  total_dist = 0
  for i in range(len(tour)-1):
    total_dist += dist[tour[i]][tour[i+1]]
  total_dist += dist[tour[-1]][tour[0]]
  return tour, total_dist

def write_files(data):
  ''' write .tsp Concorde format files for all graphs
    Input:
      data - array of data directly from loaded .pkl file
    Output:
      files are written to concorde_data/*.tsp
  '''
  for i in tqdm(range(len(data))):
    write_file(data[i], "concorde_data/tsp100_test_seed1234_"+str(i)+".tsp")

def write_file(G, file):
  ''' write a .tsp Concorde format files
    Input:
      G - 2D array of graph
      file - (str) filename
    Output:
      file is written to specified input file name
  '''
  n = len(G)
  with open(file, "w") as f:
    f.write("NAME: " + file[:-4] + "\n")
    f.write("TYPE: TSP\n")
    f.write("COMMENT: " + str(n) + " locations\n")
    f.write("DIMENSION: " + str(n) + "\n")
    f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
    f.write("NODE_COORD_SECTION\n")
    for i, node in enumerate(G):
      f.write(str(i+1) + " " + str(node[0]) + " " + str(node[1]) + "\n")
  f.close()

def save_tours(val_data):
  ''' iterate over Concorde tours and save the optimal tours and distances
    Input: 
      val_data - dataset loaded from validation .pkl file
    Output:
      saves tours in concorde_tours.npy, saves tour distances in concorde_dists.npy
      prints the mean optimal distance over the validation data
  '''
  all_tours = list()
  all_dists = list()
  for i in tqdm(range(10000)):
    dist = calculate_edges(val_data[i])
    try:
      tour, dist = read_concorde_sol("concorde_data/tsp100_test_seed1234_" +str(i) + ".sol", dist)
      all_tours.append(tour)
      all_dists.append(dist/1000)
    except:
      print(np.mean(all_dists))
      print("missing file " + "concorde_data/tsp100_test_seed1234_" +str(i) + ".sol")
      sys.exit(1)
  all_tours = np.array(all_tours)
  all_dists = np.array(all_dists)
  print(np.mean(all_dists))
  np.save("concorde_tours.npy", all_tours, allow_pickle=True)
  np.save("concorde_dists.npy", all_dists, allow_pickle=True)

def main():
  val_data = np.array(pickle.load(open("tsp100_validation_seed4321.pkl", "rb")))*1000
  test_data = np.array(pickle.load(open("tsp100_test_seed1234.pkl", "rb")))*1000
  # write Concorde formatted .tsp files with graph information
  write_files(test_data)
  # run Concorde
  os.system("bash benchmark_concorde.sh")
  # iterate through saved tours, and save the distances
  save_tours(test_data)
  all_dists = np.load("concorde_dists.npy", allow_pickle=True)
  all_tours = np.load("concorde_tours.npy", allow_pickle=True)
  print(all_tours)
  # print mean distance
  print(np.mean(all_dists))
  # creat plot of instance 5
  fig, ax = plt.subplots()
  instance5 = val_data[5]
  dist = calculate_edges(instance5)
  tour, total_dist = read_concorde_sol("concorde_data/tsp100_validation_seed4321_" +str(5) + ".sol", 
                                        dist)
  print(tour)
  print(total_dist)
  plot_tsp(ax, instance5/1000, tour, total_dist/1000, heatmap=None, mask=None, markersize=5, title="TSP")
    
if __name__ == "__main__":
  main()