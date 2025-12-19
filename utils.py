import os
import matplotlib.pyplot as plt

output_dir = '/kaggle/working/output'
figures_dir = os.path.join(output_dir, 'figures')

os.makedirs(figures_dir, exist_ok=True)

def save_figure(name, dpi=150, close=True):
    path = os.path.join(figures_dir, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close()