# import torch
# from sklearn.metrics import average_precision_score, roc_auc_score
from utils.load_configs import get_link_prediction_args

# def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
#     """
#     get metrics for the link prediction task
#     :param predicts: Tensor, shape (num_samples, )
#     :param labels: Tensor, shape (num_samples, )
#     :return:
#         dictionary of metrics {'metric_name_1': metric_1, ...}
#     """
#     predicts = predicts.cpu().detach().numpy()
#     labels = labels.cpu().numpy()

#     average_precision = average_precision_score(y_true=labels, y_score=predicts)
#     roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

#     return {'average_precision': average_precision, 'roc_auc': roc_auc}


# def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
#     """
#     get metrics for the node classification task
#     :param predicts: Tensor, shape (num_samples, )
#     :param labels: Tensor, shape (num_samples, )
#     :return:
#         dictionary of metrics {'metric_name_1': metric_1, ...}
#     """
#     predicts = predicts.cpu().detach().numpy()
#     labels = labels.cpu().numpy()

#     roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

#     return {'roc_auc': roc_auc}


import numpy as np


def analyze_target_historical_event_time_diff(
    neighbor_times_list: list,
    node_interact_times: np.ndarray,
    num_neighbors: int,
    sample_neighbor_strategy: str = "recent",
):
    """Analyze the average, median, and maximum time differences between the target edge
    interaction times in node_interact_times and the historical edge interaction event times in
    neighbor_times_list.

    Also measure the number of temporal neighbors for each node. Note that we only consider the
    most recent num_neighbors neighbors for each node.
    :param neighbor_times_list: list of ndarrays of neighbor interaction times for each node
    :param node_interact_times: ndarray, node interaction times for each node in the current batch
    :param num_neighbors: int, number of temporal neighbors to consider for each node
    :param sample_neighbor_strategy: str, strategy to sample neighbors to analyze time differences,
        either "recent" or "uniform"
    :return avg_time_diff: ndarray, shape (batch_size,), average time differences between the
        current interaction time and the historical interaction times
    :return median_time_diff: ndarray, shape (batch_size,), median time differences between the
        current interaction times and the historical interaction times
    :return max_time_diff: ndarray, shape (batch_size,), maximum time differences between the
        current interaction times and the historical interaction times
    :return num_temporal_neighbors: ndarray, shape (batch_size,), number of temporal neighbors for
        each node
    """
    # Compute the time differences between the target edge interaction times and the historical edge interaction times
    # Initialize a ndarray of shape (batch_size, num_neighbors) to np.nan
    print('here0')
    time_diffs = np.full((len(node_interact_times), num_neighbors), np.nan)
    print('here0')
    num_temporal_neighbors = np.full(len(node_interact_times), np.nan)
    print('here1')
    for i, neighbor_times in enumerate(neighbor_times_list):
        # Only consider the most recent num_neighbors neighbors
        if sample_neighbor_strategy == "recent":
            neighbor_times = neighbor_times[-num_neighbors:]
        elif sample_neighbor_strategy == "uniform":
            if len(neighbor_times) > 0:
                sampled_indices = np.random.choice(a=len(neighbor_times), size=num_neighbors)
                neighbor_times = neighbor_times[sampled_indices]
                neighbor_times = np.sort(neighbor_times)

        num_temporal_neighbors[i] = len(neighbor_times)
        if len(neighbor_times) > 0:
            time_diffs[i, -len(neighbor_times) :] = node_interact_times[i] - neighbor_times
    print('here2')
    # Compute the average, median, and maximum time differences
    avg_time_diffs = np.nanmean(time_diffs, axis=1)
    median_time_diffs = np.nanmedian(time_diffs, axis=1)
    max_time_diffs = np.nanmax(time_diffs, axis=1)

    return avg_time_diffs, median_time_diffs, max_time_diffs, num_temporal_neighbors

import numpy as np

# def analyze_target_historical_event_time_diff(
#     node_interact_times,
#     neighbor_times_list,
#     num_neighbors=10,
#     sample_neighbor_strategy="recent",
#     sample_ratio=0.05,  # fraction of nodes to process
# ):
#     print("here0")

#     # Sample a subset of indices to reduce computation
#     num_samples = int(len(node_interact_times) * sample_ratio)
#     sample_indices = np.random.choice(len(node_interact_times), num_samples, replace=False)

#     # Prepare output arrays (1D, lightweight)
#     avg_time_diffs = np.full(num_samples, np.nan)
#     median_time_diffs = np.full(num_samples, np.nan)
#     max_time_diffs = np.full(num_samples, np.nan)
#     num_temporal_neighbors = np.full(num_samples, np.nan)

#     print("here1")

#     # Process each node in a streaming manner
#     for j, i in enumerate(sample_indices):
#         neighbor_times = neighbor_times_list[i]

#         # Apply sampling strategy
#         if sample_neighbor_strategy == "recent":
#             neighbor_times = neighbor_times[-num_neighbors:]
#         elif sample_neighbor_strategy == "uniform":
#             if len(neighbor_times) > 0:
#                 sampled_indices = np.random.choice(a=len(neighbor_times), size=min(num_neighbors, len(neighbor_times)), replace=False)
#                 neighbor_times = np.sort(neighbor_times[sampled_indices])

#         num_temporal_neighbors[j] = len(neighbor_times)

#         # Compute time difference stats on-the-fly
#         if len(neighbor_times) > 0:
#             diffs = node_interact_times[i] - neighbor_times
#             avg_time_diffs[j] = np.mean(diffs)
#             median_time_diffs[j] = np.median(diffs)
#             max_time_diffs[j] = np.max(diffs)

#         # Optional: print progress periodically for large data
#         if j % 10000 == 0 and j > 0:
#             print(f"Processed {j}/{num_samples} samples")

#     print("here2")

#     # Same output format as original
#     return avg_time_diffs, median_time_diffs, max_time_diffs, num_temporal_neighbors



# import torch

# def analyze_target_historical_event_time_diff(
#     neighbor_times_list: list,
#     node_interact_times: np.ndarray,
#     num_neighbors: int,
#     sample_neighbor_strategy: str = "recent",
# ):
#     """
#     GPU-accelerated version of analyze_target_historical_event_time_diff.
#     Keeps identical input/output structure as the original function.
#     """

#     # Select GPU if available
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)

#     # Move data to GPU
#     node_interact_times = torch.tensor(node_interact_times, dtype=torch.float32, device=device)
    
#     print("gpu1")

#     # Initialize tensors on GPU
#     batch_size = len(node_interact_times)
#     time_diffs = torch.full((batch_size, num_neighbors), float("nan"), device=device)
#     print("gpu2")
#     num_temporal_neighbors = torch.full((batch_size,), float("nan"), device=device)
#     print("gpu3")

#     # Compute time differences for each node
#     # for i, neighbor_times in enumerate(neighbor_times_list):
#     #     if len(neighbor_times) == 0:
#     #         continue

#     #     neighbor_times_tensor = torch.tensor(neighbor_times, dtype=torch.float32, device=device)

#     #     # Sampling strategy
#     #     if sample_neighbor_strategy == "recent":
#     #         neighbor_times_tensor = neighbor_times_tensor[-num_neighbors:]
#     #     elif sample_neighbor_strategy == "uniform":
#     #         if len(neighbor_times_tensor) > num_neighbors:
#     #             sampled_indices = torch.randint(
#     #                 low=0, high=len(neighbor_times_tensor), size=(num_neighbors,), device=device
#     #             )
#     #             neighbor_times_tensor = torch.sort(neighbor_times_tensor[sampled_indices])[0]

#     #     num_temporal_neighbors[i] = len(neighbor_times_tensor)

#     #     if len(neighbor_times_tensor) > 0:
#     #         time_diffs[i, -len(neighbor_times_tensor):] = (
#     #             node_interact_times[i] - neighbor_times_tensor
#     #         )
    
#     # batch_size = 1024  # tune depending on your GPU memory

#     # for start in range(0, len(node_interact_times), batch_size):
#     #     end = min(start + batch_size, len(node_interact_times))
#     #     batch_neighbors = neighbor_times_list[start:end]
#     #     batch_times = node_interact_times[start:end]

#     #     for j, neighbor_times in enumerate(batch_neighbors):
#     #         i = start + j
#     #         if len(neighbor_times) == 0:
#     #             continue

#     #         neighbor_times_tensor = torch.tensor(neighbor_times, dtype=torch.float32, device=device)

#     #         # Sampling
#     #         if sample_neighbor_strategy == "recent":
#     #             neighbor_times_tensor = neighbor_times_tensor[-num_neighbors:]
#     #         elif sample_neighbor_strategy == "uniform":
#     #             if len(neighbor_times_tensor) > num_neighbors:
#     #                 sampled_indices = torch.randint(
#     #                     low=0, high=len(neighbor_times_tensor), size=(num_neighbors,), device=device
#     #                 )
#     #                 neighbor_times_tensor = torch.sort(neighbor_times_tensor[sampled_indices])[0]

#     #         num_temporal_neighbors[i] = len(neighbor_times_tensor)

#     #         if len(neighbor_times_tensor) > 0:
#     #             time_diffs[i, -len(neighbor_times_tensor):] = (
#     #                 node_interact_times[i] - neighbor_times_tensor
#     #             )

#     #         del neighbor_times_tensor
#     #     torch.cuda.empty_cache()  # clear after each mini-batch
    
#     # import torch

#     device_count = torch.cuda.device_count()
#     print(f"Using {device_count} GPUs")  # ✅ Added line to display number of GPUs

#     if device_count < 2:
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         batch_size = args.splitbatch
#         for start in range(0, len(node_interact_times), batch_size):
#             end = min(start + batch_size, len(node_interact_times))
#             batch_neighbors = neighbor_times_list[start:end]

#             for j, neighbor_times in enumerate(batch_neighbors):
#                 i = start + j
#                 if len(neighbor_times) == 0:
#                     continue

#                 neighbor_times_tensor = torch.tensor(neighbor_times, dtype=torch.float32, device=device)

#                 # Sampling
#                 if sample_neighbor_strategy == "recent":
#                     neighbor_times_tensor = neighbor_times_tensor[-num_neighbors:]
#                 elif sample_neighbor_strategy == "uniform":
#                     if len(neighbor_times_tensor) > num_neighbors:
#                         sampled_indices = torch.randint(
#                             low=0, high=len(neighbor_times_tensor),
#                             size=(num_neighbors,), device=device
#                         )
#                         neighbor_times_tensor = torch.sort(neighbor_times_tensor[sampled_indices])[0]

#                 num_temporal_neighbors[i] = len(neighbor_times_tensor)

#                 if len(neighbor_times_tensor) > 0:
#                     time_diffs[i, -len(neighbor_times_tensor):] = (
#                         node_interact_times[i] - neighbor_times_tensor
#                     )

#                 del neighbor_times_tensor
#             torch.cuda.empty_cache()

#     else:
#     # Multi-GPU parallel processing
#         batch_size = 1024
#         devices = [torch.device(f"cuda:{i}") for i in range(device_count)]
#         streams = [torch.cuda.Stream(device=d) for d in devices]

#         for start in range(0, len(node_interact_times), batch_size * device_count):
#             # Divide the work among GPUs
#             for gpu_idx, device in enumerate(devices):
#                 gpu_start = start + gpu_idx * batch_size
#                 gpu_end = min(gpu_start + batch_size, len(node_interact_times))
#                 if gpu_start >= len(node_interact_times):
#                     continue

#                 with torch.cuda.stream(streams[gpu_idx]):
#                     batch_neighbors = neighbor_times_list[gpu_start:gpu_end]
                    
#                     # Move only this batch's node_interact_times to the GPU
#                     batch_times = torch.tensor(
#                         node_interact_times[gpu_start:gpu_end],
#                         dtype=torch.float32,
#                         device=device
#                     )

#                     for j, neighbor_times in enumerate(batch_neighbors):
#                         i = gpu_start + j
#                         if len(neighbor_times) == 0:
#                             continue

#                         neighbor_times_tensor = torch.tensor(
#                             neighbor_times, dtype=torch.float32, device=device
#                         )

#                         # Sampling
#                         if sample_neighbor_strategy == "recent":
#                             neighbor_times_tensor = neighbor_times_tensor[-num_neighbors:]
#                         elif sample_neighbor_strategy == "uniform":
#                             if len(neighbor_times_tensor) > num_neighbors:
#                                 sampled_indices = torch.randint(
#                                     low=0, high=len(neighbor_times_tensor),
#                                     size=(num_neighbors,), device=device
#                                 )
#                                 neighbor_times_tensor = torch.sort(
#                                     neighbor_times_tensor[sampled_indices]
#                                 )[0]

#                         num_temporal_neighbors[i] = len(neighbor_times_tensor)

#                         if len(neighbor_times_tensor) > 0:
#                             time_diffs[i, -len(neighbor_times_tensor):] = (
#                                 batch_times[j] - neighbor_times_tensor
#                             ).detach().cpu()  # Move result back to CPU

#                         del neighbor_times_tensor
#                     del batch_times
#                     torch.cuda.empty_cache()

#         # Synchronize all GPUs
#         for stream in streams:
#             stream.synchronize()



    
#     print("gpu4")
#     # NaN-safe reduction operations
#     avg_time_diffs = torch.nanmean(time_diffs, dim=1)
#     median_time_diffs = torch.nanmedian(time_diffs, dim=1).values
#     max_time_diffs = torch.nanmax(time_diffs, dim=1).values
    
#     print('gpu5')

#     # Move back to CPU before returning (keep same type as original)
#     avg_time_diffs = avg_time_diffs.cpu().numpy()
#     median_time_diffs = median_time_diffs.cpu().numpy()
#     max_time_diffs = max_time_diffs.cpu().numpy()
#     num_temporal_neighbors = num_temporal_neighbors.cpu().numpy()
    

#     # ✅ Return identical to the original signature
#     return avg_time_diffs, median_time_diffs, max_time_diffs, num_temporal_neighbors



def analyze_inter_event_time(
    neighbor_times_list: list,
    node_interact_times: np.ndarray,
):
    """Compute the average inter-event time between two consecutive interactions for a target
    node's history and then average across nodes."""
    # avg_inter_event_time = 0
    # median_inter_event_time = 0
    # total_num = 0
    avg_inter_event_time_list = []
    median_inter_event_time_list = []
    for i, neighbor_times in enumerate(neighbor_times_list):
        neighbor_times = np.append(neighbor_times, node_interact_times[i])
        # calculate the inter-event time (difference between adjacent elements)
        inter_event_times = np.diff(neighbor_times)
        # assert inter event times are non-negative
        assert np.all(inter_event_times >= 0)
        node_i_avg_inter_event_time = np.mean(inter_event_times)
        node_i_median_inter_event_time = np.median(inter_event_times)
        if not np.isnan(
            node_i_avg_inter_event_time
        ):  # will be nan if there were no temporal neighbors (historical interactions)
            # avg_inter_event_time += node_i_avg_inter_event_time
            avg_inter_event_time_list.append(node_i_avg_inter_event_time)
            # median_inter_event_time += node_i_median_inter_event_time
            # total_num += 1
        if not np.isnan(node_i_median_inter_event_time):
            median_inter_event_time_list.append(node_i_median_inter_event_time)

    # avg_inter_event_time /= total_num
    avg_inter_event_time = np.mean(avg_inter_event_time_list)
    std_inter_event_time = np.std(avg_inter_event_time_list)
    # median_inter_event_time = np.median(median_inter_event_time_list)
    median_inter_event_time = np.mean(median_inter_event_time_list)
    # median_inter_event_time /= total_num
    return avg_inter_event_time, median_inter_event_time, std_inter_event_time
