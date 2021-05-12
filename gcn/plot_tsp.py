import numpy as np
import torch
import os
import pickle5 as pickle
import torch.nn.functional as F
import torch.nn as nn
import sys 
def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def load_heatmaps(filename, symmetric=True):
    if filename is None:
        return None
    heatmaps, *_ = load_dataset(filename)
    if (heatmaps >= 0).all():
        print("Warning: heatmaps where not stored in logaritmic space, conversion may be lossy!")
        heatmaps = np.log(heatmaps)
    return heatmaps if not symmetric else np.maximum(heatmaps, np.transpose(heatmaps, (0, 2, 1)))

class Beamsearch(object):
    def __init__(self, beam_size, batch_size, num_nodes,
                 dtypeFloat=torch.FloatTensor, dtypeLong=torch.LongTensor, 
                 probs_type='raw', random_start=False):
        # Beamsearch parameters
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.num_nodes = num_nodes
        self.probs_type = probs_type
        # Set data types
        self.dtypeFloat = dtypeFloat
        self.dtypeLong = dtypeLong
        # Set beamsearch starting nodes
        self.start_nodes = torch.zeros(batch_size, beam_size).type(self.dtypeLong)
        if random_start == True:
            # Random starting nodes
            self.start_nodes = torch.randint(0, num_nodes, (batch_size, beam_size)).type(self.dtypeLong)
        # Mask for constructing valid hypothesis
        self.mask = torch.ones(batch_size, beam_size, num_nodes).type(self.dtypeFloat)
        self.update_mask(self.start_nodes)  # Mask the starting node of the beam search
        # Score for each translation on the beam
        self.scores = torch.zeros(batch_size, beam_size).type(self.dtypeFloat)
        self.all_scores = []
        # Backpointers at each time-step
        self.prev_Ks = []
        # Outputs at each time-step
        self.next_nodes = [self.start_nodes]

    def get_current_state(self):
        current_state = (self.next_nodes[-1].unsqueeze(2)
                         .expand(self.batch_size, self.beam_size, self.num_nodes))
        return current_state

    def get_current_origin(self):
        return self.prev_Ks[-1]

    def advance(self, trans_probs):
        if len(self.prev_Ks) > 0:
            if self.probs_type == 'raw':
                beam_lk = trans_probs * self.scores.unsqueeze(2).expand_as(trans_probs)
            elif self.probs_type == 'logits':
                beam_lk = trans_probs + self.scores.unsqueeze(2).expand_as(trans_probs)
        else:
            beam_lk = trans_probs
            # Only use the starting nodes from the beam
            if self.probs_type == 'raw':
                beam_lk[:, 1:] = torch.zeros(beam_lk[:, 1:].size()).type(self.dtypeFloat)
            elif self.probs_type == 'logits':
                beam_lk[:, 1:] = -1e20 * torch.ones(beam_lk[:, 1:].size()).type(self.dtypeFloat)
        # Multiply by mask
        beam_lk = beam_lk * self.mask
        beam_lk = beam_lk.view(self.batch_size, -1)  # (batch_size, beam_size * num_nodes)
        # Get top k scores and indexes (k = beam_size)
        bestScores, bestScoresId = beam_lk.topk(self.beam_size, 1, True, True)
        # Update scores
        self.scores = bestScores
        # Update backpointers
        prev_k = bestScoresId / self.num_nodes
        self.prev_Ks.append(prev_k)
        # Update outputs
        new_nodes = bestScoresId - prev_k * self.num_nodes
        self.next_nodes.append(new_nodes)
        # Re-index mask
        perm_mask = prev_k.unsqueeze(2).expand_as(self.mask)  # (batch_size, beam_size, num_nodes)
        self.mask = self.mask.gather(1, perm_mask)
        # Mask newly added nodes
        self.update_mask(new_nodes)

    def update_mask(self, new_nodes):
        arr = (torch.arange(0, self.num_nodes).unsqueeze(0).unsqueeze(1)
               .expand_as(self.mask).type(self.dtypeLong))
        new_nodes = new_nodes.unsqueeze(2).expand_as(self.mask)
        update_mask = 1 - torch.eq(arr, new_nodes).type(self.dtypeFloat)
        self.mask = self.mask * update_mask
        if self.probs_type == 'logits':
            # Convert 0s in mask to inf
            self.mask[self.mask == 0] = 1e20

    def sort_best(self):
        return torch.sort(self.scores, 0, True)

    def get_best(self):
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    def get_hypothesis(self, k):
        assert self.num_nodes == len(self.prev_Ks) + 1

        hyp = -1 * torch.ones(self.batch_size, self.num_nodes).type(self.dtypeLong)
        for j in range(len(self.prev_Ks) - 1, -2, -1):
            hyp[:, j + 1] = self.next_nodes[j + 1].gather(1, k).view(1, self.batch_size)
            k = self.prev_Ks[j].gather(1, k)
        return hyp

def calculate_edges(nodes):
  num_nodes = nodes.shape[0]
  src, dst = list(), list()
  weights = np.zeros((num_nodes,num_nodes))
  for i in range(weights.shape[0]):
    for j in range(weights.shape[0]):
      weights[i,j] = np.linalg.norm(nodes[i]-nodes[j])
  neighbors = np.ones((num_nodes,num_nodes))
  np.fill_diagonal(neighbors, 2)
  node_id = np.arange(num_nodes)
  for i in range(len(nodes)-1):
    for j in range(i+1, len(nodes)):
      src.append(i)
      dst.append(j)
  return np.array(src), np.array(dst), weights, neighbors, node_id

def tour_nodes_to_tour_len(nodes, W_values):
    tour_len = 0
    for idx in range(len(nodes) - 1):
        i = nodes[idx]
        j = nodes[idx + 1]
        tour_len += W_values[i][j]
    tour_len += W_values[j][nodes[0]]
    return tour_len

def is_valid_tour(nodes, num_nodes):
    return sorted(nodes) == [i for i in range(num_nodes)]

def beamsearch_tour_nodes_shortest(y_pred_edges, x_edges_values, beam_size, batch_size, num_nodes,
                                   dtypeFloat, dtypeLong, probs_type='raw', random_start=False):
    if probs_type == 'raw':
        y = y_pred_edges#[:, :, :, 1]  # B x V x V
    elif probs_type == 'logits':
        y = y_pred_edges#[:, :, :, 1]  # B x V x V
        y[y == 0] = -1e-20  # Set 0s (i.e. log(1)s) to very small negative number
    # Perform beamsearch
    beamsearch = Beamsearch(beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type, random_start)
    trans_probs = y.gather(1, beamsearch.get_current_state())
    for step in range(num_nodes - 1):
        beamsearch.advance(trans_probs)
        trans_probs = y.gather(1, beamsearch.get_current_state())
    ends = torch.zeros(batch_size, 1).type(dtypeLong)
    shortest_tours = beamsearch.get_hypothesis(ends)
    shortest_lens = [1e6] * len(shortest_tours)
    for idx in range(len(shortest_tours)):
        shortest_lens[idx] = tour_nodes_to_tour_len(shortest_tours[idx].cpu().numpy(),
                                                    x_edges_values[idx].cpu().numpy())
    for pos in range(1, beam_size):
        ends = pos * torch.ones(batch_size, 1).type(dtypeLong)  # New positions
        hyp_tours = beamsearch.get_hypothesis(ends)
        for idx in range(len(hyp_tours)):
            hyp_nodes = hyp_tours[idx].cpu().numpy()
            hyp_len = tour_nodes_to_tour_len(hyp_nodes, x_edges_values[idx].cpu().numpy())
            if hyp_len < shortest_lens[idx] and is_valid_tour(hyp_nodes, num_nodes):
                shortest_tours[idx] = hyp_tours[idx]
                shortest_lens[idx] = hyp_len
    return shortest_tours


def calc_dist(instance, solution, heatmap, mask, markersize=5, title="TSP"):
    loc = np.array(instance)
    tour = np.array(solution)
    routes = [loc[tour[r]].tolist() for r in range(len(loc))]
    routes = np.array(routes)
    xs, ys = routes.transpose()
    start_coord = routes[0]
    if mask is not None:
        frm, to = np.triu(mask).nonzero()  # Return non-zero indices of the upper triangle of an array. 
        edges_coords = np.stack((loc[frm], loc[to]), -2)
    total_dist = 0.0
    x_prev, y_prev = start_coord   
    for (x, y) in routes:
        total_dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)      
        x_prev, y_prev = x, y
    x0, y0 = start_coord
    total_dist += np.sqrt((x0 - x_prev) ** 2 + (y0 - y_prev) ** 2)
    return total_dist

cut = int(sys.argv[1])#10000
map_name = str(sys.argv[2])
tsp_name = str(sys.argv[3])
tsp_size = int(sys.argv[4])
beam_size = int(sys.argv[5])
nodes = np.array(load_dataset(tsp_name))
maps = np.array(load_heatmaps(map_name))
num_nodes = tsp_size
#beam_size = 1280
#beam_size = 10000
batch = 1#cut
dtypeFloat = torch.cuda.FloatTensor
dtypeLong = torch.cuda.LongTensor
weight_mat = np.empty((0,100,100))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time
t0= time.time()
bs_holder = np.empty((0,100))
dist=0
for i in range(cut):
    print(i)
    _,_,temp,_,_ = calculate_edges(nodes[i])
#    weight_mat = np.concatenate((weight_mat,temp.reshape([1,100,100])),axis =0)
    maps_inp = torch.Tensor(maps[i,...]).unsqueeze(0).to(device)
    weight_mat = torch.Tensor(temp).unsqueeze(0).to(device)
    t_ini = time.time()
    bs_nodes = beamsearch_tour_nodes_shortest(maps_inp, weight_mat, beam_size,
                                              batch, num_nodes, dtypeFloat, dtypeLong, 
                                              probs_type='logits')
    print("time:",time.time()- t_ini)
    
#    bs_holder = np.concatenate((bs_holder,bs_nodes.cpu().numpy()),axis = 0)
    instance = nodes[i,...]
#    solution = np.array(bs_holder[i,...].reshape([-1]),dtype = np.int)
    solution = np.array(bs_nodes.reshape(-1),dtype =np.int)
#    print(solution,solution.shape)
    heatmap = maps[i,...]
    maping = np.exp(heatmap)
    adj = maping > 1e-5#args.heatmap_threshold
    adj[:, 0] = 0
    adj[0, :] = 0
    dist_temp = calc_dist(instance, solution, heatmap, adj, markersize=5, title="TSP")
    dist += dist_temp
    print("tour length:",dist/(i+1))
t1 = time.time()
#bs_nodes = beamsearch_tour_nodes_shortest(maps_inp, weight_mat_inp, beam_size,
#                                          batch, num_nodes, dtypeFloat, dtypeLong, 
#                                          probs_type='logits')
print("taken all", (t1-t0)/cut)
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


def plot_tsp(ax1, instance, solution, heatmap, mask, markersize=5, title="TSP"):
    loc = np.array(instance)
    tour = np.array(solution)
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
    plt.savefig('tsp_plot',dpi=600)
    return total_dist



def calc_dist(instance, solution, heatmap, mask, markersize=5, title="TSP"):
    loc = np.array(instance)
    tour = np.array(solution)
    routes = [loc[tour[r]].tolist() for r in range(len(loc))]
    routes = np.array(routes)
    xs, ys = routes.transpose()
    start_coord = routes[0]
    if mask is not None:
        frm, to = np.triu(mask).nonzero()  # Return non-zero indices of the upper triangle of an array. 
        edges_coords = np.stack((loc[frm], loc[to]), -2)
    total_dist = 0.0
    x_prev, y_prev = start_coord   
    for (x, y) in routes:
        total_dist += np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)      
        x_prev, y_prev = x, y
    x0, y0 = start_coord
    total_dist += np.sqrt((x0 - x_prev) ** 2 + (y0 - y_prev) ** 2)
    return total_dist
fig,ax1 = plt.subplots(1)

idx_plot = 5
_,_,temp,_,_ = calculate_edges(nodes[idx_plot])
#    weight_mat = np.concatenate((weight_mat,temp.reshape([1,100,100])),axis =0)
maps_inp = torch.Tensor(maps[idx_plot,...]).unsqueeze(0).to(device)
weight_mat = torch.Tensor(temp).unsqueeze(0).to(device)
t_ini = time.time()
bs_nodes = beamsearch_tour_nodes_shortest(maps_inp, weight_mat, beam_size,
                                          batch, num_nodes, dtypeFloat, dtypeLong, 
                                          probs_type='logits')
instance = nodes[idx_plot,...]
solution = bs_nodes[idx_plot,...]
maping = np.exp(maps[idx_plot,...])
adj = maping>1e-5
adj[:,0] = 0
adj[0,:] = 0
plot_tsp(ax1,instance, solution, maping, adj, markersize=5, title="TSP")
