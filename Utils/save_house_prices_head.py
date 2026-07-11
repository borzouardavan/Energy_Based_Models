import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import os

def save_dataframe_head():
    # Load dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame.head()
    
    # Plotting the table
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Round data for better display
    df_rounded = df.round(3)
    
    table = ax.table(cellText=df_rounded.values,
                     colLabels=df_rounded.columns,
                     cellLoc='center',
                     loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    # Save to Utils
    save_path = os.path.join(os.path.dirname(__file__), 'house_prices_head.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Table successfully saved to {save_path}")

if __name__ == "__main__":
    save_dataframe_head()
