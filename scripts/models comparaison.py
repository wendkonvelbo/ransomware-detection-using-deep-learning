import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# This dictionary is where your model results will be stored.
# You must run your model functions first to populate this dictionary.
# Example (uncommented and filled out):
# results = {
#     'Standalone CNN': {'Accuracy': 0.985, 'F1-Score': 0.984},
#     'Standalone LSTM': {'Accuracy': 0.980, 'F1-Score': 0.979},
#     'Standalone DNN': {'Accuracy': 0.982, 'F1-Score': 0.981},
#     'Standalone RNN': {'Accuracy': 0.975, 'F1-Score': 0.974},
#     'Hybrid CNN-LSTM': {'Accuracy': 0.991, 'F1-Score': 0.991},
#     'Hybrid RNN-DNN': {'Accuracy': 0.988, 'F1-Score': 0.988}
# }
# For your final script, the dictionary should be populated by the model functions.

def plot_comparison_chart(results):
    """
    Generates a bar chart comparing the performance of all models.

    Args:
        results (dict): A dictionary containing 'Accuracy' and 'F1-Score'
                        for each model.
    """
    if not results:
        print("Error: The 'results' dictionary is empty. Please run the model training functions first.")
        return

    df = pd.DataFrame(results).T
    
    # Create the bar chart
    df.plot(kind='bar', figsize=(15, 8), rot=45)
    
    # Set plot title and labels
    plt.title('Performance Comparison of All Models', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    
    # Set y-axis limits to highlight the differences
    plt.ylim(0.9, 1.0)
    
    # Add legend and grid
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of the bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # First, run your model functions to populate the 'results' dictionary.
    # For a complete example, your main script's 'if __name__ == "__main__":' block
    # should look like the one in our previous conversations, which calls all functions.
    
    # This is a placeholder for demonstration. Your actual code will populate this from the functions.
    results = {
        'Standalone CNN': {'Accuracy': 0.985, 'F1-Score': 0.984},
        'Standalone LSTM': {'Accuracy': 0.980, 'F1-Score': 0.979},
        'Standalone DNN': {'Accuracy': 0.982, 'F1-Score': 0.981},
        'Standalone RNN': {'Accuracy': 0.975, 'F1-Score': 0.974},
        'Hybrid CNN-LSTM': {'Accuracy': 0.991, 'F1-Score': 0.991},
        'Hybrid RNN-DNN': {'Accuracy': 0.988, 'F1-Score': 0.988}
    }
    
    plot_comparison_chart(results)