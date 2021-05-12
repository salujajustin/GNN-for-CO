import numpy as np
import pdb
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


def plot_tsp(ax1, instance, solution, heatmap, mask, markersize=5, title="TSP"):
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
    
    #  Want the ordered sequence of the tour
    routes = [loc[tour[r]].tolist() for r in range(len(loc))]
    routes = np.array(routes)
    xs, ys = routes.transpose()
    start_coord = routes[0]

    #  Add heatmap edges to plot
    if mask is not None:
        frm, to = np.triu(mask).nonzero()  # Return non-zero indices of the upper triangle of an array. 
        edges_coords = np.stack((loc[frm], loc[to]), -2)
        
        weights = heatmap[frm, to]
        edge_colors = np.concatenate((np.tile([1, 0, 0], (len(weights), 1)), weights[:, None]), -1)

        lc_edges = LineCollection(edges_coords, colors=edge_colors, linewidths=1)
        ax1.add_collection(lc_edges)

    #  Calculate the round trip Euclidean distance between pairs of points on route
    total_dist = 0.0
    x_prev, y_prev = start_coord   
    for (x, y) in routes:
        total_dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)      
        x_prev, y_prev = x, y

    #  Complete round trip by adding back to the starting position
    x0, y0 = start_coord
    total_dist += np.sqrt((x0 - x_prev) ** 2 + (y0 - y_prev) ** 2)

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
    ax1.set_title('{}, {} nodes, total distance {:.3f}'.format(title, len(instance), total_dist))

    return total_dist
