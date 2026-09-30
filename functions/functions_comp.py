import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class DataAggregatorEvaluator:
    def __init__(self, df):
        self.df = df.dropna(subset=['qnutzwaerme_2020_kwh']).reset_index(drop=True)

    def calculate_distance(self, x1, y1, x2, y2):
        """
        Calculate Euclidean distance between two points.
        """
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def aggregate_data(self, method='distance', distance_threshold=85, n_groups=2):
        """
        Aggregate data using the specified method.
        """
        if method == 'random':
            return self._aggregate_random(n_groups)
        elif method == 'distance':
            return self._aggregate_distance(distance_threshold)
        else:
            raise ValueError("Unsupported aggregation method")

    def _aggregate_random(self, n_groups):
        """
        Aggregate data by creating random groups. Remaining indices that do not form a full group will be discarded.
        """
        # Determine the number of full groups that can be formed
        n_full_groups = len(self.df) // n_groups
        group_indices = np.random.choice(len(self.df), (n_full_groups, n_groups), replace=False)

        aggregated_data = []
        for group in group_indices:
            aggregated_row = {
                'id': self.df.loc[group, 'id_x'].values,
                'x_coord': self.df.loc[group, 'xcoord_x'].mean(),
                'y_coord': self.df.loc[group, 'ycoord_x'].mean(),
                '2020': self.df.loc[group, '2020'].mean(),
                'qnutzwaerme_2020_kwh': self.df.loc[group, 'qnutzwaerme_2020_kwh'].mean()
            }
            aggregated_data.append(aggregated_row)

        aggregated_df = pd.DataFrame(aggregated_data)
        aggregated_df.drop_duplicates(subset=['id'], inplace=True)

        return aggregated_df

    def _aggregate_distance(self, distance_threshold):
        """
        Aggregate data by grouping points based on distance.
        """
        groups = []
        for i in range(len(self.df)):
            group = [i]
            for j in range(len(self.df)):
                if self.calculate_distance(self.df.loc[i, 'xcoord_x'], self.df.loc[i, 'ycoord_x'],
                                           self.df.loc[j, 'xcoord_x'], self.df.loc[j, 'ycoord_x']) < distance_threshold and i != j:
                    group.append(j)
            groups.append(group)

        aggregated_data = []
        for group in groups:
            aggregated_row = {
                'id': sorted(self.df.loc[group, 'id_x'].values),
                'x_coord': self.df.loc[group, 'xcoord_x'].mean(),
                'y_coord': self.df.loc[group, 'ycoord_x'].mean(),
                '2020': self.df.loc[group, '2020'].mean(),
                'qnutzwaerme_2020_kwh': self.df.loc[group, 'qnutzwaerme_2020_kwh'].mean()
            }
            aggregated_data.append(aggregated_row)

        aggregated_df = pd.DataFrame(aggregated_data)
        aggregated_df.drop_duplicates(subset=['id'], inplace=True)

        return aggregated_df

    def aggregate_and_evaluate(self, distance_threshold):
        """
        Aggregate data based on distance and evaluate the results.
        """
        aggregated_df = self.aggregate_data(method='distance', distance_threshold=distance_threshold)

        metrics = self._evaluate(aggregated_df)
        return metrics

    def aggregate_and_evaluate_random(self, n_groups, n_experiments):
        """
        Aggregate data by random grouping and evaluate the results.
        """
        metrics_list = []

        for _ in range(n_experiments):
            self.df.reset_index(drop=True, inplace=True)
            aggregated_df = self.aggregate_data(method='random', n_groups=n_groups)
            metrics = self._evaluate(aggregated_df)
            metrics_list.append(metrics)

        mean_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}
        std_metrics = {key: np.std([m[key] for m in metrics_list]) for key in metrics_list[0]}

        return mean_metrics, std_metrics

    def _evaluate(self, aggregated_df):
        """
        Evaluate the aggregated data against the true values.
        """
        y_true = aggregated_df['2020']
        y_pred = aggregated_df['qnutzwaerme_2020_kwh']

        metrics = {
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'rmse_perc': np.sqrt(mean_squared_error(y_true, y_pred)) / y_true.mean() * 100,
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'corr': y_true.corr(y_pred)
        }

        return metrics

    def plot_metric(self, results_df, results_std=None, metric='r2', evaluation_type='distance_threshold'):
        """
        Plot the specified metric against the evaluation type.
        """
        if results_std is not None:
            std_scores = results_std[metric]
        else:
            std_scores = 0

        plt.figure(figsize=(10, 6))

        # Plot the metric against evaluation_type
        plt.plot(results_df[evaluation_type], results_df[metric], label='RMSE / Mean', color='blue')

        # Handle standard deviation fill
        if metric.lower() == 'corr':
            # Ensure values are between 0 and 1
            lower_bound = np.clip(results_df[metric] - std_scores, 0, 1)
            upper_bound = np.clip(results_df[metric] + std_scores, 0, 1)
        else:
            lower_bound = results_df[metric] - std_scores
            upper_bound = results_df[metric] + std_scores

        # Fill between the bounds
        plt.fill_between(results_df[evaluation_type], 
                        lower_bound,
                        upper_bound, 
                        color='blue', alpha=0.2, label='± 1 Standard Deviation')

        # Title and labels
        plt.title(f'RMSE / Mean vs {evaluation_type.replace("_", " ").title()}', fontsize=16)
        plt.xlabel(f'{evaluation_type.replace("_", " ").title()} (m)', fontsize=15)
        plt.ylabel('RMSE / Mean (%)', fontsize=15)

        # Set axis limits based on metric
        if metric.lower() == 'corr':
            plt.ylim(-0.05, 1.05)
        elif metric.lower() in ['rmse', 'rmse_perc']:
            plt.ylim(0, 60)

        # Configure tick label sizes
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        # Legend and grid
        plt.legend(fontsize=15)
        plt.grid(True)

        # Show plot
        plt.show()