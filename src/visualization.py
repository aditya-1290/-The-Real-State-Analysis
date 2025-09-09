import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_histogram(df, column):
    """
    Plot a histogram for a given column.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.show()

def plot_scatter(df, x, y):
    """
    Plot a scatter plot for two columns.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f'Scatter plot of {x} vs {y}')
    plt.show()

def plot_correlation_matrix(df):
    """
    Plot the correlation matrix.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def plot_interactive_scatter(df, x, y, color=None):
    """
    Plot an interactive scatter plot using Plotly.
    """
    fig = px.scatter(df, x=x, y=y, color=color)
    fig.show()
