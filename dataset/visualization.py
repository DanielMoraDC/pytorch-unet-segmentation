import pandas as pd
import numpy as np

import skimage.io

from typing import List

import matplotlib.pyplot as plt


def display_instance(row: pd.Series,
                     ax,
                     title_column: str = 'category_name',
                     fontsize: int = 14) -> None:
    ax.imshow(skimage.io.imread(row['image_path']), aspect='auto')
    ax.set_title(row[title_column], fontsize=fontsize)
    ax.axis('off')

    
def display_instances(df: pd.DataFrame,
                      display_fn=display_instance,
                      n_cols=7,
                      **display_args):
    n = len(df)
    n_rows = int(np.ceil(n/n_cols))
    # Estimate figsize
    figsize = (3*n_cols, 3.75*n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    for i in range(n):
        row = df.iloc[i]

        if n_rows == 1:
            ax = axes[i]
        else:
            i_idx, j_idx = np.unravel_index(i, (n_rows, n_cols))
            ax = axes[i_idx][j_idx]

        display_fn(row, ax, **display_args)

    fig.tight_layout()
