from glob import glob
from plot import plot_Q_table_dist

def analyse_conv(dir_name):
    for filepath in glob(f"{dir_name}/Q_*.pkl"):
        plot_Q_table_dist(filepath, f"{dir_name}/Q_999.pkl", save=1)


if __name__ == '__main__':
    analyse_conv("sarsa_9_1_9999_more_2682")