from glob import glob
from plot import plot_Q_table

def analyse(dir_name):
    for filepath in glob(f"{dir_name}/Q_*.pkl"):
        plot_Q_table(filepath, save=1)


if __name__ == '__main__':
    analyse("sarsa_6_1_9999_984403")