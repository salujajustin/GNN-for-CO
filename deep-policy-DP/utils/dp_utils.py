import os 
import torch
import numpy as np
from .dp import BatchGraph, StreamingTopK, SimpleBatchTopK, run_dp

def evaluate_dp(is_vrp, batch, heatmaps, beam_size, collapse, score_function, heatmap_threshold, knn, verbose):

    coords = torch.cat((batch['depot'][:, None], batch['loc']), 1) if is_vrp else batch
    demands = batch['demand'] if is_vrp else None
    vehicle_capacities = batch['capacity'] if is_vrp else None
    graph = BatchGraph.get_graph(
        coords, score_function=score_function, heatmap=heatmaps, heatmap_threshold=heatmap_threshold, knn=knn, quantize_cost_dtype=torch.int32,
        demand=demands, vehicle_capacity=vehicle_capacities,
        start_node=0, node_score_weight=1.0, node_score_dist_to_start_weight=0.1
    )
    assert graph.batch_size == len(coords)
    add_potentials = graph.edge_weight is not None
    assert add_potentials == ("potential" in score_function.split("_"))

    if False:
        # This implementation is simpler but slower
        candidate_queue = SimpleBatchTopK(beam_size)
    else:
        candidate_queue = StreamingTopK(
            beam_size,
            dtype=graph.score.dtype if graph.score is not None else graph.cost.dtype,
            verbose=verbose,
            payload_dtypes=(torch.int32, torch.int16),  # parent = max 1e9, action = max 2e3 (for VRP with 1000 nodes)
            device=coords.device,
            alloc_size_factor=10. if beam_size * graph.batch_size <= int(1e6) else 2.,  # up to 1M we can easily allocate 10x so 10MB
            kthvalue_method='sort',  # Other methods may increase performance but are experimental / buggy
            batch_size=graph.batch_size
        )

    mincost_dp_qt, solution = run_dp(
        graph, candidate_queue, return_solution=True, collapse=collapse,
        beam_device=coords.device, bound_first=True, # Always bound first #is_vrp or beam_size >= int(1e7),
        sort_beam_by='group_idx', trace_device='cpu',
        verbose=verbose, add_potentials=add_potentials
    )
    assert len(mincost_dp_qt) == graph.batch_size
    assert len(solution) == graph.batch_size
    solutions_np = [sol.cpu().numpy() if sol is not None else None for sol in solution]
    cost = graph.dequantize_cost(mincost_dp_qt)
    return solutions_np, cost, graph.batch_size

