"""Core Skills Drill — Descriptive Analytics

Compute summary statistics, plot distributions, and create a correlation
heatmap for the sample sales dataset.

Usage:
    python drill_eda.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_summary(df):
    """Compute summary statistics for all numeric columns.

    Args:
        df: pandas DataFrame with at least some numeric columns

    Returns:
        DataFrame containing count, mean, median, std, min, max
        for each numeric column. Save the result to output/summary.csv.
    """
    summary = df.describe()
    summary.loc['median'] = df.median(numeric_only=True)
    summary = summary.loc[['count', 'mean', 'median', 'std', 'min', 'max']]
    summary.to_csv("output/summary.csv")
    return summary


def plot_distributions(df, columns, output_path):
    """Create a 2x2 subplot figure with histograms for the specified columns.

    Args:
        df: pandas DataFrame
        columns: list of 4 column names to plot (use numeric columns)
        output_path: file path to save the figure (e.g., 'output/distributions.png')

    Returns:
        None — saves the figure to output_path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color='skyblue')
        axes[i].set_title(f'Distribution of {col}', fontsize=14)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correlation(df, output_path):
    """Compute Pearson correlation matrix and visualize as a heatmap.

    Args:
        df: pandas DataFrame with numeric columns
        output_path: file path to save the figure (e.g., 'output/correlation.png')

    Returns:
        None — saves the figure to output_path
    """
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Pearson Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """Load data, compute summary, and generate all plots."""
    os.makedirs("output", exist_ok=True)

    # Load the CSV from data/sample_sales.csv
    data_path = os.path.join("data", "sample_sales.csv")
    df = pd.read_csv(data_path)

    # Feature Engineering: Derive total_price and month
    df['total_price'] = df['quantity'] * df['unit_price']
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month

    # Call compute_summary and save the result
    compute_summary(df)

    # Choose 4 numeric-friendly columns and call plot_distributions
    plot_cols = ['quantity', 'unit_price', 'total_price', 'month']
    plot_distributions(df, plot_cols, "output/distributions.png")

    # Call plot_correlation
    plot_correlation(df, "output/correlation.png")
    print("EDA completed. Outputs saved to the 'output' directory.")


if __name__ == "__main__":
    main()
