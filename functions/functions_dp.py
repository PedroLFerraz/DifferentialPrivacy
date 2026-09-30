"""
Useful functions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import opendp.prelude as dp
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
from tqdm import tqdm
dp.enable_features("contrib")


def gen_laplacian_noise(epsilon, sensitivity, size=1):
    """
    Generate laplacian noise based on differential privacy params
    """

    # Calculate the scale parameter
    b = sensitivity / epsilon

    # Generate Laplacian noise
    noise = np.random.laplace(loc=0, scale=b, size=size)
    return noise
np.random.seed(seed=0)

def calculate_grouped_heat_density(df, epsilon=None, sensitivity=None, chain=None, group_size=None, n_groups=None):
    """
    Group the elements of the DataFrame and calculate the new heat densities.
    
    Parameters:
    - df: The input DataFrame.
    - epsilon: The privacy parameter for Laplacian noise (required if chain is not provided).
    - sensitivity: The sensitivity parameter for Laplacian noise (required if chain is not provided).
    - chain: The OpenDP function for generating noise (required if epsilon and sensitivity are not provided).
    - group_size: The size of each group (required if n_groups is not provided).
    - n_groups: The number of groups (required if group_size is not provided).
    
    Returns:
    - new_df: DataFrame with original (non-noisy) heat densities.
    - new_noisy_df: DataFrame with noisy heat densities.
    """
    
    if n_groups is not None:
        group_size = len(df) // n_groups
        remainder = len(df) % n_groups
    elif group_size is None:
        raise ValueError("Either group_size or n_groups must be provided.")
    
    grouped_gas_consumption = []
    grouped_noisy_gas_consumption = []
    grouped_distance = []

    start_index = 0
    laplacian_noise = None
    
    if epsilon and sensitivity:
        laplacian_noise = gen_laplacian_noise(epsilon, sensitivity, size=len(df) - (group_size if n_groups is None else 0) + 1)

    for i in range(n_groups if n_groups is not None else len(df) - (group_size-1)):
        gas_sum = 0
        distance_sum = 0
        current_group_size = group_size + (1 if n_groups is not None and i < remainder else 0)

        for j in range(current_group_size):
            gas_sum += df.iloc[start_index + j]['gas_consumption']
            distance_sum += df.iloc[start_index + j]['distance']

        if chain:
            gas_sum_noisy = max(0, chain(df.iloc[start_index:start_index + current_group_size]['gas_consumption'].to_list()))
        elif laplacian_noise is not None:
            gas_sum_noisy = max(0, gas_sum + laplacian_noise[i])
        else:
            raise ValueError("Either OpenDP chain or (epsilon and sensitivity) must be provided for noise generation.")

        grouped_gas_consumption.append(gas_sum)
        grouped_distance.append(distance_sum)
        grouped_noisy_gas_consumption.append(gas_sum_noisy)

        start_index += current_group_size if n_groups is not None else 1

    # Create new DataFrames from the grouped values
    new_df = pd.DataFrame({
        'gas_consumption': grouped_gas_consumption,
        'distance': grouped_distance
    })

    new_noisy_df = pd.DataFrame({
        'gas_consumption': grouped_noisy_gas_consumption,
        'distance': grouped_distance
    })

    # Calculate the new heat_density
    new_df['heat_density'] = new_df['gas_consumption'] / new_df['distance']
    new_noisy_df['heat_density'] = new_noisy_df['gas_consumption'] / new_noisy_df['distance']

    # Calculate cumulative distances
    new_df['cumulative_distances'] = np.cumsum(new_df['distance']) - new_df['distance'] / 2
    new_noisy_df['cumulative_distances'] = new_df['cumulative_distances']

    return new_df, new_noisy_df


def calculate_grouped_mean_gas_consumption(df, epsilon=None, sensitivity=None, chain=None, group_size=None, n_groups=None):
    """
    Group the elements of the DataFrame and calculate the mean gas consumption.
    
    Parameters:
    - df: The input DataFrame with only the 'gas_consumption' column.
    - epsilon: The privacy parameter for Laplacian noise (required if chain is not provided).
    - sensitivity: The sensitivity parameter for Laplacian noise (required if chain is not provided).
    - chain: The OpenDP function for generating noise (required if epsilon and sensitivity are not provided).
    - group_size: The size of each group (required if n_groups is not provided).
    - n_groups: The number of groups (required if group_size is not provided).
    
    Returns:
    - new_df: DataFrame with the original (non-noisy) mean gas consumption.
    - new_noisy_df: DataFrame with the noisy mean gas consumption.
    """
    
    if n_groups is not None:
        group_size = len(df) // n_groups
        remainder = len(df) % n_groups
    elif group_size is None:
        raise ValueError("Either group_size or n_groups must be provided.")
    
    grouped_gas_consumption = []
    grouped_noisy_gas_consumption = []

    start_index = 0
    laplacian_noise = None
    
    if epsilon and sensitivity:
        laplacian_noise = gen_laplacian_noise(epsilon, sensitivity, size=len(df) - (group_size if n_groups is None else 0) + 1)

    for i in range(n_groups if n_groups is not None else len(df) - (group_size-1)):
        gas_sum = 0
        current_group_size = group_size + (1 if n_groups is not None and i < remainder else 0)

        for j in range(current_group_size):
            gas_sum += df.iloc[start_index + j]['gas_consumption']

        mean_gas_consumption = gas_sum / current_group_size

        if chain:
            noisy_gas_sum = max(0, chain(df.iloc[start_index:start_index + current_group_size]['gas_consumption'].to_list()))
            mean_noisy_gas_consumption = noisy_gas_sum / current_group_size
        elif laplacian_noise is not None:
            noisy_gas_sum = max(0, gas_sum + laplacian_noise[i])
            mean_noisy_gas_consumption = noisy_gas_sum / current_group_size
        else:
            raise ValueError("Either OpenDP chain or (epsilon and sensitivity) must be provided for noise generation.")

        grouped_gas_consumption.append(mean_gas_consumption)
        grouped_noisy_gas_consumption.append(mean_noisy_gas_consumption)

        start_index += current_group_size if n_groups is not None else 1

    # Create new DataFrames from the grouped values
    new_df = pd.DataFrame({
        'gas_consumption': grouped_gas_consumption,
    })

    new_noisy_df = pd.DataFrame({
        'gas_consumption': grouped_noisy_gas_consumption,
    })

    return new_df, new_noisy_df

def find_longest_consecutive_sequence(indices):
    """
    Find where the heat grid should be built
    """
    if len(indices) == 0:
        return []
    sorted_indices = sorted(indices)
    longest_sequence = []
    current_sequence = [sorted_indices[0]]

    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == sorted_indices[i-1] + 1:
            current_sequence.append(sorted_indices[i])
        else:
            if len(current_sequence) > len(longest_sequence):
                longest_sequence = current_sequence
            current_sequence = [sorted_indices[i]]

    if len(current_sequence) > len(longest_sequence):
        longest_sequence = current_sequence

    return longest_sequence

def calculate_accuracy(list1, list2):
    """
    Calculate the accuracy based on how many points of list1 are in list2.

    Parameters:
    list1 (list): The list of points to check.
    list2 (list): The list of points to compare against.

    Returns:
    float: The accuracy as a percentage.
    """
    count = 0
    for point in list1:
        if point in list2:
            count += 1

    accuracy = count / len(list1) if len(list1) > 0 else 0
    return accuracy * 100  # Return accuracy as a percentage

def jaccard_similarity(list1, list2):
    """
    Calculates the Jaccard similarity between two lists.
    """
    set1, set2 = set(list1), set(list2)
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0:
        return 0
    return (len(intersection) / len(union))*100

def are_lists_mutually_inclusive(list1, list2):
    """
    Check if all elements of list1 are in list2 or all elements of list2 are in list1.

    Parameters:
    list1 (list): The first list of elements.
    list2 (list): The second list of elements.

    Returns:
    bool: True if either all elements of list1 are in list2 or all elements of list2 are in list1.
    """
    set1 = set(list1)
    set2 = set(list2)

    is_list1_in_list2 = set1.issubset(set2)
    is_list2_in_list1 = set2.issubset(set1)

    return is_list1_in_list2 or is_list2_in_list1

def equal_lists(list1, list2):
    """
    Check if list1 is equal to list2.

    Parameters:
    list1 (list): The first list of elements.
    list2 (list): The second list of elements.

    Returns:
    bool: True if either all elements of list1 are in list2 or all elements of list2 are in list1.
    """

    set1 = set(list1)
    set2 = set(list2)

    is_list1_in_list2 = set1.issubset(set2)
    is_list2_in_list1 = set2.issubset(set1)

    return is_list1_in_list2 and is_list2_in_list1

def calculate_confusion_metrics(df, noisy_df, heat_density_limit):
    # Filter the dataframes based on the heat density limit
    df_high_density = df[df['heat_density'] > heat_density_limit]
    noisy_df_high_density = noisy_df[noisy_df['heat_density'] > heat_density_limit]

    # Get the total set of all possible indexes
    all_indexes = set(df.index)

    # Get the indexes of high-density areas in both original and noisy datasets
    high_density_indexes = set(df_high_density.index)
    noisy_high_density_indexes = set(noisy_df_high_density.index)

    # True Positives (TP)
    TP = len(high_density_indexes.intersection(noisy_high_density_indexes))

    # False Positives (FP)
    FP = len(noisy_high_density_indexes.difference(high_density_indexes))

    # False Negatives (FN)
    FN = len(high_density_indexes.difference(noisy_high_density_indexes))

    # True Negatives (TN)
    TN = len(all_indexes.difference(high_density_indexes.union(noisy_high_density_indexes)))

    # Calculate metrics

    # Accuracy
    accuracy = ((TP + TN) / (TP + TN + FP + FN))*100

    # Precision
    precision = (TP / (TP + FP))*100 if (TP + FP) != 0 else 0

    # Recall
    recall = (TP / (TP + FN))*100 if (TP + FN) != 0 else 0

    # F1-Score
    f1_score = (2 * (precision * recall) / (precision + recall)) if (precision + recall) != 0 else 0

    # Return the metrics as a list
    return [accuracy, precision, recall, f1_score]

def create_dp_sum(sensitivity, epsilon, bounded_sum, clamp):
    base_lap = dp.m.make_laplace(
        dp.atom_domain(T=float),
        dp.absolute_distance(T=float),
        scale=sensitivity / epsilon
    )
    return clamp >> bounded_sum >> base_lap

def run_single_evaluation(df_generated, size, epsilon, heat_density_limit, sensitivity, bounded_sum, clamp, n_runs, calculation_fn):
    accuracies = []
    f1_scores = []
    rmse_list = []
    rmse_perc_list = []
    mape_list = []

    dp_sum = create_dp_sum(sensitivity, epsilon, bounded_sum, clamp)

    for _ in range(n_runs):
        df, noisy_df = calculation_fn(df_generated, n_groups=size, chain=dp_sum)

        if 'heat_density' in df.columns:
            df_high_density = df[df['heat_density'] > heat_density_limit]
            noisy_df_high_density = noisy_df[noisy_df['heat_density'] > heat_density_limit]

            accuracy = calculate_confusion_metrics(df, noisy_df, heat_density_limit)[0]
            # accuracy = jaccard_similarity(df_high_density.index, noisy_df_high_density.index)
            f1_score = calculate_confusion_metrics(df, noisy_df, heat_density_limit)[3]

            rmse = np.sqrt(mean_squared_error(df['heat_density'], noisy_df['heat_density']))
            rmse_perc = np.sqrt(mean_squared_error(df['heat_density'], noisy_df['heat_density'])) / df['heat_density'].mean() * 100
            mape = np.mean(np.abs((df['heat_density'] - noisy_df['heat_density']) / df['heat_density'])) * 100
        else:
            rmse = np.sqrt(mean_squared_error(df['gas_consumption'], noisy_df['gas_consumption']))
            rmse_perc = np.sqrt(mean_squared_error(df['gas_consumption'], noisy_df['gas_consumption'])) / df['gas_consumption'].mean() * 100
            accuracy = 0
            f1_score = 0
            mape = 0
        
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        rmse_perc_list.append(rmse_perc)
        rmse_list.append(rmse)
        mape_list.append(mape)

    return {
        'epsilon': epsilon,
        'accuracy_mean': np.mean(accuracies) if accuracies else None,
        'accuracy_std': np.std(accuracies) if accuracies else None,
        'f1_score_mean': np.mean(f1_scores) if f1_scores else None,
        'f1_score_std': np.std(f1_scores) if f1_scores else None,
        'rmse_mean': np.mean(rmse_list),
        'rmse_std': np.std(rmse_list),
        'rmse_perc_mean': np.mean(rmse_perc_list),
        'rmse_perc_std': np.std(rmse_perc_list),
        'mape_mean': np.mean(mape_list),
        'mape_std': np.std(mape_list),
        'n_groups': size
    }

def evaluate_over_epsilon(df_generated, n_groups, epsilon_values, heat_density_limit, sensitivity, bounded_sum, clamp, n_runs, calculation_fn):
    all_results = []

    for size in n_groups:
        results = Parallel(n_jobs=-1, backend='threading')(delayed(run_single_evaluation)(
            df_generated, size, epsilon, heat_density_limit, sensitivity, bounded_sum, clamp, n_runs, calculation_fn
        ) for epsilon in tqdm(epsilon_values, desc=f'Number of Groups {size}'))
        all_results.extend(results)

    return all_results

def plot_index(x, noisy_x=None, param='gas_consumption'):
    """
    Plot with index
    """
    # Define the font sizes
    title_fontsize = 14
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12

    title = param.replace('_', ' ').title() # 'Heat Density' or 'Gas Consumption'
    
    if noisy_x is None:
        # Only one variable provided, plot a single plot
        plt.figure(figsize=(8, 6))
        plt.bar(np.arange(len(x)), x, label='Data')
        plt.xlabel('Index', fontsize=label_fontsize)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        plt.gca().yaxis.set_major_formatter(formatter)

        # Ensure the scientific notation (xEy) is displayed on the top
        plt.gca().ticklabel_format(style='scientific', axis='both', useOffset=True)
        plt.ylabel(f'{title} (kWh)', fontsize=label_fontsize)
        plt.title(f'{title}', fontsize=title_fontsize)

        # Set tick sizes for both axes
        plt.tick_params(axis='x', labelsize=tick_fontsize)
        plt.tick_params(axis='y', labelsize=tick_fontsize)

        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        # Two variables provided, plot subplots
        _, axs = plt.subplots(2, 1, figsize=(8, 6))

        # Plot for original data
        axs[0].bar(np.arange(len(x)), x, label='Generated Data')
        axs[0].set_xlabel('Index', fontsize=label_fontsize)
        ylimit = max(x.max(), noisy_x.max()) * 1.1
        axs[0].set_ylim(0, ylimit)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        axs[0].yaxis.set_major_formatter(formatter)

        # Set tick sizes for both axes
        axs[0].tick_params(axis='x', labelsize=tick_fontsize)
        axs[0].tick_params(axis='y', labelsize=tick_fontsize)

        axs[0].set_ylabel(f'{title} (kWh)', fontsize=label_fontsize)
        axs[0].set_title(f'Generated {title}', fontsize=title_fontsize)

        plt.gca().yaxis.set_major_formatter(formatter)

        # Ensure the scientific notation (xEy) is displayed on the top
        plt.gca().ticklabel_format(style='scientific', axis='both', useOffset=True)
        axs[0].set_ylabel(f'{title} (kWh)')
        axs[0].set_title(f'Generated {title}')

        # Plot for noisy data
        axs[1].bar(np.arange(len(x)),noisy_x, label='Noisy Data', color='orange')
        axs[1].set_xlabel('Index', fontsize=label_fontsize)
        axs[1].set_ylabel(f'{title} (kWh)', fontsize=label_fontsize)
        axs[1].set_title(f'Noisy {title}', fontsize=label_fontsize)
        axs[1].set_ylim(0, ylimit)
        axs[1].tick_params(axis='x', labelsize=tick_fontsize)
        axs[1].tick_params(axis='y', labelsize=tick_fontsize)

        # Adjust layout
        plt.tight_layout()

        # Display the plot
        plt.show()

def plot_limit(df, df_noisy, df_limit, df_noisy_limit, size=0, file_name=None, param='heat_density', close=True):
    """
    Plot with limits
    """
    # Define font sizes
    title_fontsize = 14
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12

    title = param.replace('_', ' ').title() # 'Heat Density' or 'Gas Consumption'

    _, axs = plt.subplots(2, 1, figsize=(8, 6))
    ylimit = max(df[param].max(), df_noisy[param].max()) * 1.1

    # Plot for the original data
    axs[0].bar(df.index, df[param], label='All Points')
    axs[0].scatter(df_limit.index, df_limit[param], color='red', label=f'High {title}')
    axs[0].set_ylim(0, ylimit)

    # Plot for the noisy data
    axs[1].bar(df_noisy.index, df_noisy[param], label='All Points', color='orange', alpha=1)
    axs[1].scatter(df_noisy_limit.index, df_noisy_limit[param], color='red', label=f'High {title}')
    axs[1].set_ylim(0, ylimit)

    # Apply labels and titles
    axs[0].set_xlabel('Index', fontsize=label_fontsize)
    axs[0].set_ylabel(f'{title} (kWh/m)', fontsize=label_fontsize)
    axs[0].set_title(f'High {title} Highlighted (Group size {100/(size+1)})', fontsize=title_fontsize)

    axs[1].set_xlabel('Index', fontsize=label_fontsize)
    axs[1].set_ylabel(f'{title} (kWh/m)', fontsize=label_fontsize)
    axs[1].set_title(f'Noisy High {title} Highlighted (Group size {100/(size+1)})', fontsize=title_fontsize)

    # Formatter for scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    axs[0].yaxis.set_major_formatter(formatter)
    axs[1].yaxis.set_major_formatter(formatter)

    # Apply scientific formatting for the y-axis
    axs[0].ticklabel_format(style='scientific', axis='y', useOffset=True)
    axs[1].ticklabel_format(style='scientific', axis='y', useOffset=True)

    # Set tick label font sizes
    axs[0].tick_params(axis='x', labelsize=tick_fontsize)
    axs[0].tick_params(axis='y', labelsize=tick_fontsize)
    axs[1].tick_params(axis='x', labelsize=tick_fontsize)
    axs[1].tick_params(axis='y', labelsize=tick_fontsize)

    # Adjust layout and show/save the plot
    plt.tight_layout()

    if file_name is not None: 
        plt.savefig(f'{file_name}')
    if close:
        plt.close()
    else:
        plt.show()

def plot_limit_decision(df, df_noisy, df_limit, df_noisy_limit, size=0, file_name=None, param='heat_density', close=True):
    """
    Plot with limits
    """
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))

    df['is_high'] = df.index.isin(df_limit.index)
    df_noisy['is_high'] = df_noisy.index.isin(df_noisy_limit.index)

    axs[0].scatter(df.index, df['is_high'], color='blue', alpha=0.5)
    axs[0].step(df.index, df['is_high'], where='mid', color='blue', alpha=1, label='High Values')
    
    axs[1].scatter(df_noisy.index, df_noisy['is_high'], color='orange', alpha=0.5)
    axs[1].step(df_noisy.index, df_noisy['is_high'], where='mid', color='orange', alpha=1, label='High Values')

    axs[0].set_xlabel('Index')
    axs[0].set_ylabel(f'High Heat Density (True/False)')
    axs[0].set_ylim(-0.1, 1.1)
    axs[0].set_yticks([0, 1], ['False', 'True'])
    axs[0].set_title(f'High Heat Density Highlighted (Number of Groups = {100/(size+1)})')

    axs[1].set_xlabel('Index')
    axs[1].set_ylabel(f'High Heat Density (True/False)')
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].set_yticks([0, 1], ['False', 'True'])
    axs[1].set_title(f'Noisy High Heat Density Highlighted (Number of Groups = {100/(size+1)})')

    for ax in axs:
        ax.legend()
        ax.grid(False)

    plt.tight_layout()
    if file_name is not None: 
        plt.savefig(f'{file_name}')
    if close:
        plt.close()
    else:
        plt.show()

def plot_distances(df, param):
    """
    Plots one parameter in terms of the distances
    """

    title = param.replace('_', ' ').title() # 'Heat Density' or 'Gas Consumption'

    plt.scatter(df['cumulative_distances'], df[param], color='blue', zorder=5)

    for x, y in zip(df['cumulative_distances'], df[param]):
        plt.plot([x, x], [0, y], color='gray', linestyle='--', zorder=1)

    for i in range(len(df)):
        if i == 0:
            plt.plot([0, df['distance'].iloc[i]], [df[param].iloc[i], df[param].iloc[i]], color='red', linestyle='-', linewidth=2, zorder=3)
        else:
            plt.plot([df['cumulative_distances'].iloc[i] - df['distance'].iloc[i]/2,
                    df['cumulative_distances'].iloc[i] + df['distance'].iloc[i]/2] ,
                    [df[param].iloc[i], df[param].iloc[i]],
                    color='red', linestyle='-', linewidth=2, zorder=3)

    plt.xlabel('Distances')
    plt.ylabel(f'{title}')
    plt.title(f'{title} vs Distances')
    plt.grid(True)
    plt.show()

def plot_epsilon(all_results):
    for n_groups in all_results['n_groups'].unique():
        results = all_results[all_results['n_groups'] == n_groups]
        epsilon_values = results['epsilon']
        accuracy_mean = np.array(results['accuracy_mean'])
        accuracy_std = np.array(results['accuracy_std'])                       

        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epsilon_values, accuracy_mean, color='b', label='Accuracy')
        plt.fill_between(epsilon_values, np.clip(accuracy_mean - accuracy_std, 0, 100), np.clip(accuracy_mean + accuracy_std, 0,100), color='b', alpha=0.2)
        plt.xscale('log')
        plt.xlabel('Epsilon')
        plt.ylabel('Accuracy (%)')
        plt.title(f'Accuracy per Epsilon (Groups size = {100/n_groups})')
        plt.ylim(0, 105)
        plt.grid(True)

        rmse_mean = np.array(results['rmse_mean'])
        rmse_std = np.array(results['rmse_std'])
        plt.subplot(1, 2, 2)
        plt.plot(epsilon_values, rmse_mean, color='r', label='RMSE/Mean')
        plt.fill_between(epsilon_values, np.clip(rmse_mean - rmse_std, 0, max(rmse_mean - rmse_std)), rmse_mean + rmse_std, color='r', alpha=0.2)
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.gca().ticklabel_format(style='scientific', axis='both', useOffset=True)
        plt.xscale('log')
        plt.xlabel('Epsilon')
        plt.ylabel('RMSE/Mean (%)')
        plt.title(f'RMSE/Mean per Epsilon (Group size = {100/n_groups})')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'../images/epsilon/results_number_groups_{n_groups}.png')
        plt.close()

def plot_rmse_mape(all_results):
    for n_groups in all_results['n_groups'].unique():
        results = all_results[all_results['n_groups'] == n_groups]
        epsilon_values = results['epsilon']
        rmse_mean = np.array(results['rmse_mean'])
        rmse_std = np.array(results['rmse_std'])
        f1_score_mean = np.array(results['f1_score_mean'])
        f1_score_std = np.array(results['f1_score_std'])
                                          
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epsilon_values, rmse_mean, color='g', label='RMSE')
        plt.fill_between(epsilon_values, np.clip(rmse_mean - rmse_std, 0, max(rmse_mean - rmse_std)), rmse_mean + rmse_std, color='g', alpha=0.2)
        plt.xscale('log')
        plt.xlabel('Epsilon')
        plt.ylabel('RMSE')
        plt.title(f'RMSE per Epsilon (Group size = {100/n_groups})')
        plt.grid(True)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epsilon_values, f1_score_mean/100, color='m', label='F1 Score')
        plt.fill_between(epsilon_values, np.clip((f1_score_mean - f1_score_std)/100, 0, 100), np.clip((f1_score_mean + f1_score_std)/100, 0, 100), color='m', alpha=0.2)
        plt.xscale('log')
        plt.xlabel('Epsilon')
        plt.ylabel('F1 Score (%)')
        plt.title(f'F1 per Epsilon (Group size = {100/n_groups})')
        plt.ylim(0, 105)
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'../images/epsilon/results_number_groups_{n_groups}_rmse_mape.png')
        plt.close()