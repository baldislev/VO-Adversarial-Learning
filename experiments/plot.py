import matplotlib.pyplot as plt
import numpy as np
from res_parser import parse_results, get_run_params_from_path


def get_top10_by_param(res_list, param='adv_rms'):
    top10 = []
    results_sorts_by_param = sorted(res_list, key=lambda item: item[f'{param}'],reverse=True)
    top10_list = results_sorts_by_param[:10]
    for i, run in enumerate(top10_list):
        run_data = get_run_params_from_path(run['res_dir'])
        # print(run_data)
        run_data['loss'] = (param, run[f'{param}'])
        top10.append(run_data)
    return top10

def results_dict_to_vec(results):
    # [{
    #     'gradient': 'gradient_ascent',
    #     'data': 'whole_data',
    #     'norm': 'Linf',
    #     'trajectory': 'opt_whole_trajectory',
    #     'train_t_crit': 'none',
    #     'train_factor': '1.0',
    #     'train_rot_crit': 'quat_product',
    #     'train_flow_crit': 'mse',
    #     'train_target_t_crit': 'none',
    #     'eval_crit': 'rms',
    #     'epsilon': '1',
    #     'epochs': '4',
    #     'alpha': '0.05',
    #     'beta': '0.5',
    #     't_decay': 'None',
    #     'loss':  ('adv_rms', 0.726706153973937)
    # }]
    # returns:
    #   x: ['gd_wd_trc:qp_tfc:mse_epochs:4_alpha:0.05_beta:0.5_td:None']
    #   y: [0.726706153973937]

    x = []
    y = []

    shortcut_dict = {'gradient_ascent': 'gd', 'stochastic_gradient_ascent': 'sgd', 'whole_data': 'wd',
                     'split_data': 'sd', 'quat_product': 'qp', 'none': 'x', 'None': 'x'}

    for item in results:
        item_data = f"{shortcut_dict[item['gradient']]}_" \
                    f"{shortcut_dict[item['data']]}_" \
                    f"trc:{shortcut_dict[item['train_rot_crit']]}_" \
                    f"tfc:{item['train_flow_crit']}_" \
                    f"epochs:{item['epochs']}_" \
                    f"alpha:{item['alpha']}_" \
                    f"beta:{item['beta']}_" \
                    f"td:{item['t_decay']}" \
                    .replace('none', 'x') \
                    .replace('None', 'x')
        x.append(item_data)
        y.append(item['loss'][1])

    return x, y

def plot(x, y, title, ylabel):
    font_title = {
        'family': 'serif',
        'color': 'darkred', #'#0343DF'
        'weight': 'normal',
        'size': 18,
    }
    font_labels = {
        'family': 'serif',
        'color': 'darkred',
        'weight': 'normal',
        'size': 16,
    }

    plt.figure(figsize=(15, 15))
    plt.plot(np.array(x), np.array(y), 'o', markersize=10)
    plt.xticks(rotation=90, fontname='serif', color='black', size=14)
    plt.ylabel(ylabel, labelpad = 20, fontdict=font_labels)
    plt.title(title, pad = 20, fontdict=font_title)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(f'{title}.png')

def plot_by_ylabel(results, title, ylabel):
    top10 = get_top10_by_param(results, ylabel)
    # print(top10)
    x, y = results_dict_to_vec(top10)
    plot(x, y, title, ylabel)

def plot_top10_by_adv_rms(results):
    title = 'top10 runs by adv_rms'
    ylabel = 'adv_rms'
    plot_by_ylabel(results, title, ylabel)

def plot_top10_by_adv_mean_partial_rms(results):
    title = 'top10 runs by adv_mean_partial_rms'
    ylabel = 'adv_mean_partial_rms'
    plot_by_ylabel(results, title, ylabel)

if __name__ == '__main__':

    results = parse_results()

    plot_top10_by_adv_rms(results)
    plot_top10_by_adv_mean_partial_rms(results)
