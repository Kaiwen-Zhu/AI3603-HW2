import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.colors import ListedColormap
import numpy as np
import pickle as pkl
from glob import glob



def plot_reward(file_path, save):
    with open(file_path) as f:
        vals = f.readline().split(', ')[:-1]
    vals = [float(val) for val in vals]

    plt.scatter(np.arange(1000), vals, s=3)
    plt.xlabel('episode')
    plt.ylabel('reward')
    
    mode = max(vals, key=vals.count)
    plt.axhline(mode, color='red', lw=1)

    ticks = list(np.arange(-7000, 0, 1000)) + [mode]
    plt.yticks(ticks=ticks, labels=ticks)

    if save:
        plt.savefig(file_path.replace('txt', 'png'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_eps(file_path, save):
    with open(file_path) as f:
        vals = f.readline().split(', ')[:-1]
    vals = [float(val) for val in vals]

    plt.plot(vals)
    plt.xlabel('episode')
    plt.ylabel('epsilon value')
    plt.scatter(1000, vals[-1], color='r', s=7)
    plt.annotate(text=(1000, round(vals[-1],3)), xy=(1000,vals[-1]), xytext=(800,0.08), color='r')

    if save:
        plt.savefig(file_path.replace('txt', 'png'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_path(file_path, save):
    with open(file_path) as f:
        s_a_s_ = f.readlines()
    s_a_s_ = [list(map(int, itm.strip().split(', '))) for itm in s_a_s_]
    path = [itm[0] for itm in s_a_s_] + [s_a_s_[-1][-1]]
    path_x = [p%12 + 0.5 for p in path]
    path_y = [3 - p//12 + 0.5 for p in path]

    plt.figure(figsize=(9, 3))
    plt.fill([1,1,11,11], [0,1,1,0], color='k')

    plt.xlim(0, 12)
    plt.ylim(0, 4)

    plt.xticks(ticks=np.arange(0.5, 12), labels=np.arange(1, 13))
    plt.yticks(ticks=np.arange(0.5, 4), labels=np.arange(1, 5))
    plt.gca().set_aspect(1)

    plt.hlines(np.arange(5), xmin=0, xmax=12, colors='gray', lw=1)
    plt.vlines(np.arange(13), ymin=0, ymax=4, colors='gray', lw=1)

    plt.plot(path_x, path_y)

    if save:
        plt.savefig(file_path.replace('txt', 'png'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_reward_eps_path(dirname, save=0):
    plot_reward(dirname + "/reward.txt", save)
    plot_eps(dirname + "/eps.txt", save)
    plot_path(dirname + "/path.txt", save)


def draw_colorbar(graduation, vmin, vmax):
    """draw and return the colorbar defined by graduation"""
    scale = 10  # changing 1 in Q-value corresponds to changing 10 in the color
    N = (graduation[-1][0] - graduation[0][0]) * scale + 1  # number of colors in the colorbar
    c_vals = np.ones((N, 4))
    mark = [(anchor[0]-graduation[0][0]) * scale for anchor in graduation]
    for ch in range(3):
        for i in range(len(graduation)-1):
            c_vals[mark[i] : mark[i+1], ch] = np.linspace(
                graduation[i][1][ch], graduation[i+1][1][ch], mark[i+1]-mark[i])
    my_cmap = ListedColormap(c_vals)

    # create dummy invisible image to display the colorbar
    img = plt.imshow(np.array([[0,1]]), cmap=my_cmap, vmin=vmin, vmax=vmax)
    img.set_visible(False)
    plt.colorbar(orientation="vertical")

    return my_cmap


def plot_Q_table(file_path, save=0):
    with open(file_path, 'rb') as f:
        Q_table = pkl.load(f)

    plt.figure(figsize=(12, 4))

    # set the axes
    plt.ylim(0, 4)
    plt.xlim(0, 12)

    plt.xticks(ticks=np.arange(0.5, 12), labels=np.arange(1, 13))
    plt.yticks(ticks=np.arange(0.5, 4), labels=np.arange(1, 5))
    plt.gca().set_aspect(1)

    plt.hlines(np.arange(5), xmin=0, xmax=12, colors='gray', lw=1)
    plt.vlines(np.arange(13), ymin=0, ymax=4, colors='gray', lw=1)

    plt.fill([1,1,11,11], [0,1,1,0], color='k')  # draw the cliff

    graduation = [
        (-100, (1,        0,        0       )),  # red
        (-30,  (1,        0x8c/255, 0       )),  # dark orange
        (-20,  (1,        1,        0       )),  # yellow
        (-10,  (0,        1,        0x7f/255)),  # spring green
        (-5,   (0x41/255, 0x69/255, 0xe1/255)),  # royal blue
        (0,    (0x80/255, 0,        0x80/255))   # purple
    ]
    my_cmap = draw_colorbar(graduation, -100, 0)

    for state, vals in Q_table.items():
        if state == 47:
            continue
        x, y = state%12, 3 - state//12
        regions = [
            ([x, x+0.5, x+1], [y+1, y+0.5, y+1]),  # top
            ([x+0.5, x+1, x+1], [y+0.5, y, y+1]),  # right
            ([x, x+0.5, x+1], [y, y+0.5, y]),  # bottom
            ([x, x+0.5, x], [y, y+0.5, y+1])  # left
        ]

        # mark Q-value
        t_color = lambda val: 'w' if val >= -7 else 'k'
        text_pos = [
            (x + 0.25, y + 0.8), (x + 0.55, y + 0.4),
            (x + 0.25, y + 0.1), (x       , y + 0.4)
        ]
        for i in range(4):
            plt.text(text_pos[i][0], text_pos[i][1], s=round(vals[i],1),
                fontsize=8, color=t_color(vals[i]))
        
        # fill color
        mapQ = lambda val: (val - graduation[0][0]) / (
        graduation[-1][0] - graduation[0][0])  # map Q-value to float in [0,1]
        for i in range(4):
            plt.fill(regions[i][0], regions[i][1], color=my_cmap(mapQ(vals[i])))

        # mark action
        action = max(vals, key=vals.get)
        xs, ys = regions[action][0], regions[action][1]
        plt.plot(xs + [xs[0]], ys + [ys[0]], color='r')
    
    if save:
        img_path = file_path.replace('Q_', '').replace('pkl', 'png')
        plt.savefig(img_path, bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()


def plot_all_Q(dir_name):
    for filepath in glob(f"{dir_name}/Q_*.pkl"):
        plot_Q_table(filepath, save=1)


def plot_Q_table_dist(file_path, save=0):
    """plot the rate of overestimation"""
    with open(file_path, 'rb') as f:
        Q_table = pkl.load(f)

    def r(s, a, gamma=0.9):
        x, y = s%12, 3 - s//12
        des = (11, 0)
        if a == 0:
            y = min(y+1, 3)
        elif a == 1:
            x = min(x+1, 11)
        elif a == 2:
            y = max(y-1, 0)
        else:
            x = max(x-1, 0)
        if y == 0 and x > 0:
            return -100 - (1 - gamma**13) / (1 - gamma)
        leng = abs(des[0] - x) + abs(des[1] - y) + 1
        return - (1 - gamma**leng) / (1 - gamma)

    for s in Q_table.keys():
        for a in Q_table[s].keys():
            Q_table[s][a] = round((Q_table[s][a] - r(s, a)) / r(s, a) * 100)

    plt.figure(figsize=(12, 4))

    # set the axes
    plt.ylim(0, 4)
    plt.xlim(0, 12)
    
    plt.xticks(ticks=np.arange(0.5, 12), labels=np.arange(1, 13))
    plt.yticks(ticks=np.arange(0.5, 4), labels=np.arange(1, 5))
    plt.gca().set_aspect(1)

    plt.hlines(np.arange(5), xmin=0, xmax=12, colors='gray', lw=1)
    plt.vlines(np.arange(13), ymin=0, ymax=4, colors='gray', lw=1)

    plt.fill([1,1,11,11], [0,1,1,0], color='k')  # draw the cliff

    graduation = [
        (0,   (0x80/255, 0,        0x80/255)),  # purple
        (5,   (0x41/255, 0x69/255, 0xe1/255)),  # royal blue
        (10,  (0,        1,        0x7f/255)),  # spring green
        (20,  (1,        1,        0       )),  # yellow
        (50,  (1,        0x8c/255, 0       )),  # dark orange
        (100, (1,        0,        0       ))   # red
    ]
    my_cmap = draw_colorbar(graduation, 0, 100)

    for state, vals in Q_table.items():
        if state == 47:
            continue
        x, y = state%12, 3 - state//12
        regions = [
            ([x, x+0.5, x+1], [y+1, y+0.5, y+1]),  # top
            ([x+0.5, x+1, x+1], [y+0.5, y, y+1]),  # right
            ([x, x+0.5, x+1], [y, y+0.5, y]),  # bottom
            ([x, x+0.5, x], [y, y+0.5, y+1])  # left
        ]

        # mark Q-value
        t_color = lambda val: 'w' if val <= 6 else 'k'
        text_pos = [
            (x + 0.25, y + 0.8), (x + 0.55, y + 0.4),
            (x + 0.25, y + 0.1), (x      , y + 0.4)
        ]
        for i in range(4):
            plt.text(text_pos[i][0], text_pos[i][1], s="{}%".format(round(vals[i],1)),
                fontsize=7, color=t_color(vals[i]))
        
        # fill color
        mapQ = lambda val: (val - graduation[0][0]) / (
        graduation[-1][0] - graduation[0][0])  # map Q-value to float in [0,1]
        for i in range(4):
            plt.fill(regions[i][0], regions[i][1], color=my_cmap(min(mapQ(vals[i]), 0.99)))
    
    if save:
        img_path = file_path.replace('Q', 'dist').replace('pkl', 'png')
        plt.savefig(img_path, bbox_inches='tight', dpi=200)
        plt.close()
    else:
        plt.show()


def plot_conv(dirname, save=0):
    """plot the process of convergence of Q-values"""
    for filepath in glob(f"{dirname}/*_*.txt"):
        with open(filepath) as f:
            vals = f.readline().split(', ')[:-1]
        vals = [float(val) for val in vals]

        focus = list(map(int, filepath[filepath.index('\\')+1 : filepath.index('.')].split('_')))
        path = [[24,1], [30,1]]
        ls = '--' if focus in path else '-'
        plt.plot(vals, label="Q({},{})".format(str(focus[0]), str(focus[1])), linestyle=ls)
        plt.xlabel('episode')
        plt.ylabel('Q-value')
        

    plt.legend()
    if save:
        plt.savefig(dirname + '/Q_conv.png', bbox_inches='tight')
    else:
        plt.show()
    plt.close()



if __name__ == '__main__':
    plot_reward_eps_path("sarsa_9_1_9999_86981", save=1)