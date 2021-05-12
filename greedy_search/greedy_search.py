import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

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

def search(G):
  '''
    Input:
      G - (nx2 array) 2D spatial distribution of n nodes
    Output:
      tour - greedy euclidean cycle that tours all nodes without repeating any nodes or edges
      dist - 
  '''
  n = len(G)
  nodes = [i for i in range(n)]
  dists = calculate_edges(G)
  #  initialization
  init = 0
  curr = init
  tour = [curr]
  nodes.remove(curr)
  dist = 0
  while (len(nodes) > 0):
    # choose node with lowest distance
    neighbor_dists = np.array([dists[curr][j] for j in nodes])
    best_index = np.argmin(neighbor_dists)
    # update node, distance
    dist += neighbor_dists[best_index]
    curr = nodes[best_index]
    nodes.remove(curr)
    tour.append(curr)
  # update distance by forming cycle
  dist += dists[tour[-1]][init]
  return tour, dist

def run_benchmark():
  val_data = np.array(pickle.load(open("tsp100_test_seed1234.pkl", "rb")))
  greedy_tours = list()
  greedy_dists = list()
  times = list()
  for graph in tqdm(val_data):
    start = time.time()
    tour, dist = search(graph)
    end = time.time()
    greedy_tours.append(tour)
    greedy_dists.append(dist)
    times.append(end-start)
  greedy_tours = np.array(greedy_tours)
  greedy_dists = np.array(greedy_dists)
  times = np.array(times)
  np.save("greedy_tours.npy", greedy_tours, allow_pickle=True)
  np.save("greedy_dists.npy", greedy_dists, allow_pickle=True)
  print(np.mean(greedy_dists))
  print(np.mean(times))

def generate_plot():
  # plot instance
  val_data = np.array(pickle.load(open("tsp100_validation_seed4321.pkl", "rb")))
  instance5 = val_data[5]
  solution, dist = search(instance5)
  fig, ax = plt.subplots()
  plot_tsp(ax, instance5, solution, dist, heatmap=None, mask=None, markersize=5, title="TSP")


def main():
  generate_plot()
  run_benchmark()

if __name__ == "__main__":
  main()
