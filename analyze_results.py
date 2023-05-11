"""
    plot the curve of success rate and query counts
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def get_success_query_curve(counts, max_count, n_im=1000):
    """given a list of count numbers, return queries vs success
    Args:
        counts (list): 
        max_count (int): the cut-off number of counts
    """
    # cut off at max_count
    counts = np.array(sorted(counts))
    queries = [] # x axis
    success_vs_query = [] # y axis
    for i in range(max_count):
        queries.append(i+1)
        success_vs_query.append( (counts <= i+1).sum()/n_im*100 )
    return queries, success_vs_query


def print_count_stats(counts, max_count, n_im=1000):
    counts = np.array(counts)
    counts = counts[counts<=max_count]
    mean = f"{counts.mean():.0f}" if counts.mean() > 10 else f"{counts.mean():.1f}"
    std = f"{counts.std():.0f}" if counts.std() > 10 else f"{counts.std():.1f}"
    # return f"Avg: {counts.mean():.0f} \pm {counts.std():.0f} ({len(counts)/n_im*100:.1f}\%)"
    init_rate = (counts==counts.min()).sum()/n_im*100
    return f"Avg: {mean} \pm {std} ({len(counts)/n_im*100:.1f}\%), init_rate: {init_rate:.1f}\%"


def get_success_rates(dict_k_valid_id_v_success_list, all_model_names, max_count):
    success_list_stack = []
    for valid_id in dict_k_valid_id_v_success_list:
        success_list = np.array(dict_k_valid_id_v_success_list[valid_id])[:max_count]
        success_list = success_list.sum(axis=0).astype(bool).astype(int).tolist()
        success_list_stack.append(success_list)
    success_list_stack = np.array(success_list_stack).sum(axis=0)
    success_rates = [success_cnt/len(dict_k_valid_id_v_success_list) for success_cnt in success_list_stack]
    for idx, success_rate in enumerate(success_rates):
        print(f"success rate of {all_model_names[idx]}: {success_rate*100:.1f}")
    return success_rates

counts_all = defaultdict(list)
root = Path("results_voc") # this folder is for table 1,2 (single obj)
dataset = 'voc'
n_wb = 2
surrogate = 'Faster R-CNN'
n_iters = 20
x_alpha = 3
lr_w = 1e-2
iterw = 10
single = True
no_balancing = False
eps = 10

victim_names = ['RetinaNet', 'Libra R-CNN', 'FoveaBox', 'FreeAnchor', 'DETR']
max_count = 6 # max 5 queries, max cut-off number of counts is 6

success_rates_list = []
count_dict = defaultdict(list)
valid_dict = defaultdict(list)
for victim_name in victim_names:
    exp_name = f'BB_{n_wb}wb_linf_{eps}_iters{n_iters}_alphax{x_alpha}_victim_{victim_name}_lr{lr_w}_iterw{iterw}'
    if dataset != 'voc':
        exp_name += f'_{dataset}'
    if n_wb == 1:
        exp_name += f'_{surrogate}'
    if single:
        exp_name = exp_name + "_single"
    if no_balancing:
        exp_name += '_noBalancing'

    folder = root / exp_name
    models_all = ['Faster R-CNN', 'YOLOv3', 'FCOS', 'Grid R-CNN', 'SSD']
    all_model_names = models_all[:n_wb] + [victim_name]
    dict_k_valid_id_v_success_list = np.load(folder / "dict_k_valid_id_v_success_list.npy", allow_pickle=True).item()
    dict_query_list = np.load(folder / "dict_k_sucess_id_v_query.npy", allow_pickle=True).item()

    n_valid = len(dict_k_valid_id_v_success_list)
    n_success = len(dict_query_list)
    counts = [i for i in dict_query_list.values()]
    print(f"n_valid: {n_valid}, n_success: {n_success}")

    count_dict[victim_name] = counts
    valid_dict[victim_name] = n_valid

    success_rates = get_success_rates(dict_k_valid_id_v_success_list, all_model_names, max_count)
    success_rates_list.append(success_rates)
    print(f"{victim_name}, {print_count_stats(counts, max_count, n_im=n_valid)}")
    print()

# average for surrogates
success_rates_list = np.array(success_rates_list)
for item in success_rates_list.mean(axis=0):
    print(f"avg: {item*100:.1f}")


# show plots
plt.figure(figsize=(5,4))
# for idx in range(1,6):
    # label =  f"linf10_x{idx}s"
    # counts = counts_all[label]
    # print(f"{label}, {print_count_stats(counts, max_count, n_im=n_valid)}")
    # x, y = get_success_query_curve(counts, max_count, n_im=n_valid)
    # x = [i - 1 for i in x]
    # plt.plot(x, y, label=label, linewidth=3)

for victim_name in victim_names:
    counts = count_dict[victim_name]
    n_valid = valid_dict[victim_name]
    print(f"{victim_name}, {print_count_stats(counts, max_count, n_im=n_valid)}")
    x, y = get_success_query_curve(counts, max_count, n_im=n_valid)
    x = [i - 1 for i in x]
    plt.plot(x, y, label=victim_name, linewidth=3)

# plt.ylim(0,80)
plt.xlabel("Queries", fontsize=12)
plt.ylabel("Fooling Rate (%)", fontsize=12)
plt.legend(loc='lower right', ncol=2)
plt.tight_layout()
# plt.savefig(f'{label}.png')
plt.savefig(f'{n_wb}wb_linf{eps}_{dataset}.pdf')
plt.close()
