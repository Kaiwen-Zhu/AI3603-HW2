from glob import glob
from plot import plot_Q_table

def plot_all_Q(dir_name):
    for filepath in glob(f"{dir_name}/Q_*.pkl"):
        plot_Q_table(filepath, save=1)


if __name__ == '__main__':
    plot_all_Q("sarsa_6_1_9999_984403")