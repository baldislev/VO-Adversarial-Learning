import matplotlib.pyplot as plt
import numpy as np


def results_dict_to_vec(results, keys, ylabel):
    x = []
    y = []

    for item in results:
        item_data = ''
        for key in keys:
            if key == 't_decay':
                if item['attack'] == 'pgd':
                    item_data += 'scheduled' if item[key] != 'None' else ''
            elif key == 'minibatch_size':
                if key not in item:
                    item_data += 'non stochastic'
                else:
                    item_data += 'batch_size: ' + item[key] + '\n'
            elif key == 'train_target_t_crit':
                item_data += 'patch: ' + item['target_t_factor'] + '\n'
            elif key == 'train_rot_crit':
                item_data += 'quat_prod: ' + item['rot_factor'] + '\n'
            elif key == 'train_flow_crit':
                item_data += item['flow_factor'] + '\n'
            else:
                item_data += f'{item[key]} \n'
        x.append(item_data)
        y.append(item[ylabel])

    return x, y


def plot(x, y, xlabel, ylabel, title):
    font_title = {
        'family': 'serif',
        'color': 'darkred',  # '#0343DF'
        'weight': 'normal',
        'size': 18,
    }
    font_labels = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
    }

    plt.style.use('ggplot')
    plt.figure(figsize=(15, 15))
    plt.bar(np.array(x), np.array(y), color='#8495FF')
    plt.xticks(fontname='serif', color='black', size=14)
    plt.ylabel(ylabel, labelpad=20, fontdict=font_labels)
    plt.xlabel(xlabel, labelpad=20, fontdict=font_labels)
    plt.title(title, pad=20, fontdict=font_title)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f'{title}.png')


def plot_by_ylabel(results, keys, ylabel, xlabel, title):
    results = sorted(results, key=lambda x: x[ylabel], reverse=True)
    x, y = results_dict_to_vec(results, keys, ylabel)
    plot(x, y, xlabel, ylabel, title)
