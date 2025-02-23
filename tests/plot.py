# python -m my_project.tests.plot
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

# import seaborn as sns
# sns.set_palette("Purples")

FONT_NAME = 'sans-serif'
TICK_FONTSIZE = 58 
TITLE_SIZE = 62 


def read_data(dir, is_overhead=False, is_avg_sz=False):
    DATA = {}
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                name, bs, pl, gl, dataset = lines[1].split(" ")
                if is_overhead:
                    name += f"|({pl}, {gl})"
                if is_avg_sz:
                    name += f"|{dataset}"
                DATA[name] = {}
                DATA[name]["BS"] = int(bs)
                DATA[name]["PL"] = int(pl)
                DATA[name]["GL"] = int(gl)
                DATA[name]["dataset"] = dataset

                lines = lines[2:]
                num_dataset_runs = 0
                for i, line in enumerate(lines):
                    j = i+2+num_dataset_runs
                    to_add = None 
                    if "total_timings" in line:
                        to_add = "total_lats"
                        j = j - num_dataset_runs
                    elif "forward_timings" in line:
                        to_add = "fwd_lats"
                    elif "reqs_per_s" in line:
                        to_add = "reqs_per_s"
                        num_dataset_runs = np.fromstring(lines[i+1], sep=' ').size
                    elif "GTs" in line:
                        to_add = "GTs"
                    elif "req_timings" in line:
                        to_add = "req_lats"
                    elif "req_fwd_timings" in line:
                        to_add = "req_fwd_lats"
                    elif "secs_per_token" in line:
                        to_add = "secs_per_token"
                    elif "batch_size" in line:
                        to_add = "batch_sizes"
                        j = j - num_dataset_runs
                    elif "token_nums" in line:
                        to_add = "token_nums"
                        j = j - num_dataset_runs
                    if to_add is not None:
                        DATA[name][to_add] = []
                        for row in lines[i+1:j]:
                            row = np.fromstring(row, sep=' ')
                            if row.size:
                                DATA[name][to_add].append(row)
    return DATA


def reorder_data(DATA, order):
    ordered_data = OrderedDict()
    for key in order:
        ordered_data[key] = DATA[key]
    return ordered_data

def overhead_bar_plot(DATA):
    data_total = []
    data_fwd = {} 
    data_overhead = {} 
    labels = [[], []]
    i = 0
    key_to_ind = {}
    for name in DATA.keys():
        if "HF" in name:
            continue
        else:
            pretty_name = name.split("|")[0]
            if pretty_name == "CLI":
                pretty_name = "Engine+\nServer+\nCLI"
            elif pretty_name == "Server":
                pretty_name = "Engine+\nServer"
            elif pretty_name == "Engine+Nopad":
                pretty_name = "Engine+\nNopad"
            pl, gl = DATA[name]["PL"], DATA[name]["GL"]
            key = (pl, gl)
            total_lats = DATA[name]["total_lats"][0]
            fwd_lats = DATA[name]["fwd_lats"][0] / 1000 # secs

            if key not in data_overhead:
                data_overhead[key] = [] 
                data_fwd[key] = []
                key_to_ind[key] = i
                key_ind = i
                i += 1
            else:
                key_ind = key_to_ind[key]
            labels[key_ind].append(pretty_name)
            data_overhead[key].append(np.median(total_lats - fwd_lats))
            data_fwd[key].append(np.median(fwd_lats))
            print(f"MAX OVERHEAD LATENCY {key} {pretty_name}: {np.max(total_lats-fwd_lats)}")

    # Number of (pl, gl) combinations
    num_pl_gl_combinations = len(data_overhead)

    # Create subplots with a 1-row layout, with as many columns as there are (pl, gl) combinations
    fig, axs = plt.subplots(1, num_pl_gl_combinations, figsize=(25 * num_pl_gl_combinations, 25))

    # If there's only one (pl, gl) combination, wrap axs in a list for consistency
    if num_pl_gl_combinations == 1:
        axs = [axs]

    # Plot each (pl, gl) combination in a subplot
    for i, ((pl, gl), fwd_lats) in enumerate(data_fwd.items()):
        overheads = data_overhead[(pl, gl)]
        bar_width = 0.2 / len(labels[i])
        x = np.arange(len(labels[i])) * 0.1
        ax = axs[i]  # Get the axis for the current plot
        barcontainer1 = ax.bar(x, fwd_lats, bar_width, label="Forwarding Latency")
        barcontainer2 = ax.bar(x, overheads, bottom=fwd_lats, width=bar_width, label="Overhead Latency")

        for bar in barcontainer1:
            height = bar.get_height()
            offset = 0.001 if i == 1 else 0.008
            ax.text(
                bar.get_x() + offset,  # X position: bar's right edge + offset
                bar.get_y() + height / 2,            # Y position: middle of the bar height
                f'{height:.2f}',                         # Text: value of the bar
                va='center',                         # Vertical alignment: center
                ha='left',                           # Horizontal alignment: left
                fontsize=TICK_FONTSIZE,                          # Font size
                color='black'                        # Font color
            )
        for bar, overhead in zip(barcontainer2, overheads):
            offset = 0.04 if i == 0 else 0.5
            height = bar.get_height()
            ax.text(
                bar.get_x() + 0.008,  # X position: bar's right edge + offset
                bar.get_y() + offset,            # Y position: middle of the bar height
                f'{overhead:.2f}',                         # Text: value of the bar
                va='center',                         # Vertical alignment: center
                ha='left',                           # Horizontal alignment: left
                fontsize=TICK_FONTSIZE,                          # Font size
                color='black'                        # Font color
            )

        ax.set_title(f'PL={pl}, GL={gl}', fontname=FONT_NAME, fontsize=TICK_FONTSIZE)
        ax.set_facecolor('lightgray')
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticks(x)  # Set tick positions explicitly
        ax.set_xticklabels(labels[i], fontname=FONT_NAME, fontsize=TICK_FONTSIZE)

        yticks = ax.get_yticks()
        ax.set_yticks(yticks)  # Set tick positions explicitly
        formatted_yticks = [f'{y:.2f}' for y in yticks]
        ax.set_yticklabels(formatted_yticks, fontname=FONT_NAME, fontsize=TICK_FONTSIZE)
        ax.yaxis.grid(color='white', linestyle='-')
        ax.set_axisbelow(True)

    axs[0].set_ylabel('Latency [s]', fontname=FONT_NAME, fontsize=TICK_FONTSIZE)
    plt.legend(
        fontsize=40,
        # markerscale=1.5,
        # loc='best',  # Automatically chooses the best location
        # bbox_to_anchor=(1, 1)  # Moves the legend outside the plot
    )
    fig.suptitle('Median Overhead and Forwarding Latencies by (PL, GL) Combinations', fontname=FONT_NAME, fontsize=TITLE_SIZE)
    plots_dir = "my_project/tests/plots/"
    plt.savefig(os.path.join(plots_dir, "overhead_bar_plot.pdf"))

def overhead_plot(DATA):
    data_overhead = {} 
    labels = [[], []]
    i = 0
    key_to_ind = {}
    for name in DATA.keys():
        if "HF" in name:
            continue
        else:
            pretty_name = name.split("|")[0]
            if pretty_name == "CLI":
                pretty_name = "Engine+\nServer+\nCLI"
            elif pretty_name == "Server":
                pretty_name = "Engine+\nServer"
            elif pretty_name == "Engine+Nopad":
                pretty_name = "Engine+\nNopad"
            pl, gl = DATA[name]["PL"], DATA[name]["GL"]
            key = (pl, gl)
            total_lats = DATA[name]["total_lats"][0]
            fwd_lats = DATA[name]["fwd_lats"][0] / 1000 # secs

            if key not in data_overhead:
                data_overhead[key] = [] 
                key_to_ind[key] = i
                key_ind = i
                i += 1
            else:
                key_ind = key_to_ind[key]
            labels[key_ind].append(pretty_name)
            data_overhead[key].append(total_lats - fwd_lats)
            # data_overhead_HF.append(total_lats - hf_data)

    # Number of (pl, gl) combinations
    num_pl_gl_combinations = len(data_overhead)
    # labels = list(labels)

    # Create subplots with a 1-row layout, with as many columns as there are (pl, gl) combinations
    fig, axs = plt.subplots(1, num_pl_gl_combinations, figsize=(25 * num_pl_gl_combinations, 25))

    # If there's only one (pl, gl) combination, wrap axs in a list for consistency
    if num_pl_gl_combinations == 1:
        axs = [axs]

    # Plot each (pl, gl) combination in a subplot
    for i, ((pl, gl), overheads) in enumerate(data_overhead.items()):
        ax = axs[i]  # Get the axis for the current plot
        # Create boxplot for the current (pl, gl) combination
        ax.boxplot(
            overheads,
            medianprops=dict(color="black", linewidth=4),   # Make the median line thicker
            boxprops=dict(linewidth=4),                     # Make the box borders thicker
            whiskerprops=dict(linewidth=4),                 # Make the whiskers thicker
            capprops=dict(linewidth=4),                     # Make the caps thicker
            flierprops=dict(marker='o', color='red', markersize=24, markeredgewidth=4),
            showmeans=False,
        )


        for j, values in enumerate(overheads):
                median = np.median(values)
                # ax.text(j + 1-0.27, median, f'{median:.3f}', horizontalalignment='right', verticalalignment='center', fontsize=TICK_FONTSIZE, color='black')
                ax.text(j + 1+0.25, median, f'{median:.3f}', horizontalalignment='left', verticalalignment='center', fontsize=TICK_FONTSIZE, color='black')

        ax.set_title(f'PL={pl}, GL={gl}', fontname=FONT_NAME, fontsize=TICK_FONTSIZE)
        ax.set_facecolor('lightgray')
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        axs[i].set_xticklabels(labels[i], fontname=FONT_NAME, fontsize=TICK_FONTSIZE)
        ax.yaxis.grid(color='white', linestyle='-')
        ax.set_axisbelow(True)

        yticks = axs[i].get_yticks()
        axs[i].set_yticks(yticks)  # Set tick positions explicitly
        formatted_yticks = [f'{y:.3f}' for y in yticks]
        axs[i].set_yticklabels(formatted_yticks, fontname=FONT_NAME, fontsize=TICK_FONTSIZE)

    axs[0].set_ylabel('Overhead [s]', fontname=FONT_NAME, fontsize=TICK_FONTSIZE)
    axs[1].set_ylim(top=1.0)
    fig.suptitle('Overhead by (PL, GL) Combinations', fontname=FONT_NAME, fontsize=TITLE_SIZE)
    plots_dir = "my_project/tests/plots/"
    plt.savefig(os.path.join(plots_dir, "overhead_plot.pdf"))

def tp_plot(DATA, dataset, scale):
    labels = list(DATA.keys())
    fig, ax = plt.subplots()
    for name in DATA.keys():
        data = []
        x = DATA[name]["reqs_per_s"][0]
        for secs_per_token in DATA[name]["secs_per_token"]:
            secs_per_token = secs_per_token[secs_per_token != np.inf]
            avg_tp = np.mean(secs_per_token)
            data.append(avg_tp)
        plt.plot(x, data, label=name)
        plt.scatter(x, data)

    scale_string = "Small Scale" if scale == "small" else "Large Scale"
    plt.title(f'Normalized Latency on {dataset} Dataset ({scale_string})', fontname=FONT_NAME)
    plt.xlabel('Request Rate [reqs/s]', fontname=FONT_NAME)
    plt.ylabel('Normalized Latency [s/token]', fontname=FONT_NAME)
    plt.legend()
    plt.yticks(fontname=FONT_NAME)
    plt.gca().set_facecolor('lightgray')
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(color='white', linestyle='-')
    ax.set_axisbelow(True)

    plots_dir = "my_project/tests/plots/"
    plt.savefig(os.path.join(plots_dir, f"{dataset}_tp_plot_{scale}_scale.pdf"))

def avg_sz_plot(DATA):
    avgs = []
    dataset_avgs = {}
    models = set()
    for name in DATA.keys():
        pretty_name = name.split('|')[0]
        models.add(pretty_name)
        dataset = DATA[name]["dataset"]
        include = len(DATA[name]["batch_sizes"][0]) // 4
        token_nums = DATA[name]["batch_sizes"][0]#[:3*include]
        # avgs.append(np.mean(token_nums))
        avg = np.mean(token_nums)
        if dataset not in dataset_avgs:
            dataset_avgs[dataset] = {} 
        dataset_avgs[dataset][pretty_name] = avg

    datasets = list(dataset_avgs.keys())  # Unique dataset names
    models = list(models)  # Unique model names
    num_datasets = len(datasets)
    num_models = len(models)
    x = np.arange(len(datasets)) * 0.012
    bar_width = 0.01 / num_models  # Width of bars for each model


    fig, ax = plt.subplots() # figsize=(20,8)
    # Plot bars for each model within each dataset
    for i, model in enumerate(models):
        model_avgs = [dataset_avgs[dataset].get(model, 0) for dataset in datasets]  # Averages per dataset for the current model
        bar_positions = x + (i * bar_width)  # Offset positions for the bars of the current model
        bars = ax.bar(bar_positions, model_avgs, bar_width, label=model)

        # Annotate bars with the mean values
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center', fontsize=8, color='black')

    ax.set_title('Average Batch Size', fontname=FONT_NAME)
    ax.set_ylabel('Number of Batched Requests', fontname=FONT_NAME)
    ax.set_xticks(x + (bar_width * (num_models - 1) / 2))  # Center x-ticks
    ax.set_xticklabels(datasets, fontname=FONT_NAME)
    ax.yaxis.grid(color='white', linestyle='-')
    ax.set_axisbelow(True)
    # ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.legend(loc="best")
    plt.yticks(fontname=FONT_NAME)
    # plt.grid(color='white', linestyle='-')
    plt.gca().set_facecolor('lightgray')
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()

    plots_dir = "my_project/tests/plots/"
    plt.savefig(os.path.join(plots_dir, f"avg_sz_plot.pdf"))

overhead_dir = "my_project/tests/overhead_results/"
shareGPT_large_dir = "my_project/tests/dataset_results/large_scale/ShareGPT"
alpaca_large_dir = "my_project/tests/dataset_results/large_scale/Alpaca"
shareGPT_small_dir = "my_project/tests/dataset_results/small_scale/ShareGPT"
alpaca_small_dir = "my_project/tests/dataset_results/small_scale/Alpaca"
avg_sz_dir = "my_project/tests/avg_sz_results/"

order_overhead = [
    "Engine|(256, 128)",
    "Engine|(1024, 1024)",
    "Server|(256, 128)",
    "Server|(1024, 1024)",
    "CLI|(256, 128)",
    "CLI|(1024, 1024)",
    "Engine+Nopad|(256, 128)",
    "Engine+Nopad|(1024, 1024)",
]
oder_lat = ["SimpleReqManager", "Nopad"]
DATA_OVERHEAD = reorder_data(read_data(overhead_dir, is_overhead=True), order_overhead)
DATA_shareGPT_large = reorder_data(read_data(shareGPT_large_dir), oder_lat)
DATA_alpaca_large = reorder_data(read_data(alpaca_large_dir), oder_lat)
DATA_shareGPT_large = reorder_data(read_data(shareGPT_large_dir), oder_lat)
DATA_alpaca_large = reorder_data(read_data(alpaca_large_dir), oder_lat)
DATA_shareGPT_small = reorder_data(read_data(shareGPT_small_dir), oder_lat)
DATA_alpaca_small = reorder_data(read_data(alpaca_small_dir), oder_lat)
DATA_avg_sz = read_data(avg_sz_dir, is_avg_sz=True)

overhead_plot(DATA_OVERHEAD)
overhead_bar_plot(DATA_OVERHEAD)
tp_plot(DATA_shareGPT_large, dataset="ShareGPT", scale="large")
tp_plot(DATA_alpaca_large, dataset="Alpaca", scale="large")
tp_plot(DATA_shareGPT_small, dataset="ShareGPT", scale="small")
tp_plot(DATA_alpaca_small, dataset="Alpaca", scale="small")
avg_sz_plot(DATA_avg_sz)