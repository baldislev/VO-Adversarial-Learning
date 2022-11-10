import os
import csv
import numpy as np
import argparse
from plot_vlad import plot_by_ylabel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sort_crit', default='none', type=str, metavar='ATT', help='how to sort results')
    parser.add_argument('--filter_keys', default='', type=str, metavar='ATT', help='keys to filter. comma-space separated')
    parser.add_argument('--filter_vals', default='', type=str, metavar='ATT', help='value of that key to filter. comma-space separated')
    parser.add_argument('--dont_print', action='store_true', help='either to print or simply sort/filter the results')
    parser.add_argument('--x_keys', default='', type=str, metavar='ATT', help='keys to aggregate as x from result. + separated')
    parser.add_argument('--y_label', default='', type=str, metavar='ATT', help='criterion whose value will serve as y.')
    parser.add_argument('--x_label', default='', type=str, metavar='ATT', help='label name of x axis.')
    parser.add_argument('--title', default='', type=str, metavar='ATT', help='title of a plot to create. If set, a plot will be produced. ')
    args = parser.parse_args()

    return args


def construct_filter_crit(keys, vals):
    filter_crit = {}
    if keys == '' or vals == '':
        return filter_crit
    filter_keys = keys.split('+')
    filter_vals = vals.split('+')
    if len(filter_keys) == len(filter_vals):
        for key, val in zip(filter_keys, filter_vals):
            filter_crit[key] = val
    return filter_crit


def get_x_keys(key_str):
    return key_str.split('+')


def increase_dict(dict, start, delta):
    for key in dict:
        if dict[key] > start:
            dict[key] += delta
    return dict


def get_crit_str_from_path(path):
    crit_str = path.split('results_')[1]
    crit_str = crit_str.split('.')[0]
    return crit_str


def get_fieldnames_from_path(path):
    crit_str = get_crit_str_from_path(path)
    fieldnames = ['dataset_idx', 'dataset_name', 'traj_idx', 'traj_name', 'frame_idx',
                  'clean_' + crit_str, 'adv_' + crit_str, 'adv_delta_' + crit_str,
                  'adv_ratio_' + crit_str, 'adv_delta_ratio_' + crit_str, ]
    return fieldnames


def get_float_from_split(whole_str, partial_str):
    partial_len = len(partial_str)
    return str(float(whole_str) + float(partial_str) / 10 ** partial_len)


def get_attack_from_folder_str(folder_str):
    attack_dict = {}
    split = folder_str.split('_')

    if split[1] != 'const':
        attack_dict['norm'] = split[-1]
        attack_dict['attack'] = split[1]
        if attack_dict['attack'] not in ['pgd', 'apgd']:
            attack_dict['attack'] += '_pgd'
    else:
        attack_dict['attack'] = split[1]

    return attack_dict


def get_eval_crit_from_folder_str(folder_str):
    return '_'.join(folder_str.split('_')[1:])


def get_gradient_from_folder_str(folder_str):
    grad_dict = {}
    split = folder_str.split('_')
    if folder_str == 'gradient_ascent':
        grad_dict['gradient'] = folder_str
    else:
        grad_dict['gradient'] = '_'.join(split[:3])
        grad_dict['minibatch_size'] = split[5]
    return grad_dict


def get_data_params_from_folder_str(folder_str):
    data_dict = {}
    split = folder_str.split('_')
    if folder_str == 'whole_data':
        data_dict['data'] = folder_str
    else:
        data_dict['data'] = '_'.join(split[:2])
        if len(split) > 2:  # backward compatibility.
            if split[0] == 'split':
                data_dict['train_data'] = '_'.join(split[2:5])
                data_dict['eval_data'] = '_'.join(split[5:6])
                data_dict['test_data'] = '_'.join(split[6:7])
            elif split[0] == 'eval':
                data_dict['eval_data'] = split[2]
    return data_dict


def get_train_crit_dict(folder_str):
    # opt_t_crit_mean_partial_rms_factor_1_0_rot_crit_none_flow_crit_none_target_t_crit_none
    crit_dict = {}
    split_idx = {'t_crit': 3, 't_factor_1': 5, 't_factor_2': 6, 'rot_crit': 9, 'rot_factor_1': 12, 'rot_factor_2': 13,
                 'flow_crit': 12, 'flow_factor_1': 14, 'flow_factor_2': 15,
                 'target_t_crit': 16, 'target_factor_1': 18, 'target_factor_2': 19}

    split = folder_str.split('_')
    # t_crit: can be 'rms' or 'mean_partial_rms'
    if split[split_idx['t_crit']] == 'mean':
        crit_dict['train_t_crit'] = '_'.join(split[split_idx['t_crit']:split_idx['t_crit'] + 3])
        split_idx = increase_dict(split_idx, split_idx['t_crit'], 2)
    else:
        crit_dict['train_t_crit'] = split[split_idx['t_crit']]
    crit_dict['t_factor'] = get_float_from_split(split[split_idx['t_factor_1']], split[split_idx['t_factor_2']])
    # rot_crit: can be none or 'quat_product'
    if split[split_idx['rot_crit']] == 'quat':
        crit_dict['train_rot_crit'] = '_'.join(split[split_idx['rot_crit']:split_idx['rot_crit'] + 2])
        crit_dict['rot_factor'] = get_float_from_split(split[split_idx['rot_factor_1']], split[split_idx['rot_factor_2']])
        split_idx = increase_dict(split_idx, split_idx['rot_crit'], 4)
    else:
        crit_dict['train_rot_crit'] = split[split_idx['rot_crit']]
    # flow_crit: can be none, mse or mae
    crit_dict['train_flow_crit'] = split[split_idx['flow_crit']]
    if crit_dict['train_flow_crit'] in ['mse', 'mae']:
        crit_dict['flow_factor'] = get_float_from_split(split[split_idx['flow_factor_1']], split[split_idx['flow_factor_2']])
        split_idx = increase_dict(split_idx, split_idx['flow_crit'], 3)
    # target_t_crit: patch or none.
    crit_dict['train_target_t_crit'] = split[split_idx['target_t_crit']]
    if crit_dict['train_target_t_crit'] == 'patch':
        crit_dict['target_t_factor'] = get_float_from_split(split[split_idx['target_factor_1']], split[split_idx['target_factor_2']])
    return crit_dict


def get_hyper_params_dict(folder_str, attack_type):
    # eps_1_attack_iter_2_alpha_0_05_t_decay_10_beta_0_5_momentum_0_5_rmsprop_decay_0_1_rmsprop_eps_e-6
    params_dict = {}
    split_idx = {'epsilon': 1, 'epochs': 4, 'alpha_1': 6, 'alpha_2': 7, 'beta_1': 12, 'beta_2': 13,
                 't_decay': 10, 'momentum_1': 12, 'momentum_2': 13, 'rmsprop_decay_1': 13, 'rmsprop_decay_2': 14,
                 'rmsprop_eps': 17}
    split = folder_str.split('_')

    params_dict['epsilon'] = split[split_idx['epsilon']]
    params_dict['epochs'] = split[split_idx['epochs']]

    alpha = get_float_from_split(split[split_idx['alpha_1']], split[split_idx['alpha_2']])
    params_dict['alpha'] = alpha

    params_dict['t_decay'] = split[split_idx['t_decay']]
    if params_dict['t_decay'] != 'None':
        beta = get_float_from_split(split[split_idx['beta_1']], split[split_idx['beta_2']])
        params_dict['beta'] = beta
        split_idx = increase_dict(split_idx, split_idx['beta_2'], 3)

    if len(split) > split_idx['momentum_1'] and attack_type == 'momentum_pgd':
        params_dict['momentum'] = get_float_from_split(split[split_idx['momentum_1']], split[split_idx['momentum_2']])
        split_idx = increase_dict(split_idx, split_idx['momentum_2'], 3)
    if len(split) > split_idx['rmsprop_decay_1'] and attack_type == 'rmsprop_pgd':
        params_dict['rmsprop_decay'] = get_float_from_split(split[split_idx['rmsprop_decay_1']], split[split_idx['rmsprop_decay_2']])
    if len(split) > split_idx['rmsprop_eps'] and attack_type == 'rmsprop_pgd':
        params_dict['rmsprop_eps'] = split[split_idx['rmsprop_eps']]

    return params_dict


def get_run_params_from_path(path):
    # example: gradient_ascent/split_data/attack_pgd_norm_Linf/opt_whole_trajectory/opt_t_crit_rms_factor_1_0_rot_crit_none_flow_crit_none_target_t_crit_none/eval_rms/eps_1_attack_iter_2_alpha_0_05
    run_params = {}
    depth_dict_train = {0: 'data_dir', 1: 'none', 2: 'none', 3: 'gradient',
                        4: 'data', 5: 'attack', 6: 'trajectory',
                        7: 'train_crit', 8: 'eval_crit', 9: 'hyperparameters'}

    depth_dict_const = {0: 'data_dir', 1: 'none', 2: 'none', 3: 'attack'}

    path = path.split('tartanvo_1914/')[1]
    folders = path.split('/')
    depth_dict = depth_dict_train if folders[0] == 'VO_adv_project_train_dataset_8_frames' else depth_dict_const

    # construct params:
    for i, folder_str in enumerate(folders):
        if depth_dict[i] == 'data_dir':
            run_params['data_dir'] = folder_str
        if depth_dict[i] == 'gradient':
            grad_dict = get_gradient_from_folder_str(folder_str)
            run_params = dict(**run_params, **grad_dict)
        if depth_dict[i] == 'data':
            data_dict = get_data_params_from_folder_str(folder_str)
            run_params = dict(**run_params, **data_dict)
        if depth_dict[i] == 'attack':
            attack_dict = get_attack_from_folder_str(folder_str)
            run_params = dict(**run_params, **attack_dict)
        if depth_dict[i] == 'trajectory':
            run_params['trajectory'] = folder_str
        if depth_dict[i] == 'train_crit':
            train_crit_dict = get_train_crit_dict(folder_str)
            run_params = dict(**run_params, **train_crit_dict)
        if depth_dict[i] == 'eval_crit':
            run_params['eval_crit'] = get_eval_crit_from_folder_str(folder_str)
        if depth_dict[i] == 'hyperparameters':
            hyperparams_dict = get_hyper_params_dict(folder_str, run_params['attack'])
            run_params = dict(**run_params, **hyperparams_dict)

    return run_params


def parse_result_single(curr_dir, curr_children, result_files, traj_len=8):
    result = {'res_dir': curr_dir}
    # parse path in order to infer run parameters:
    run_params_dict = get_run_params_from_path(curr_dir)
    # merge result dict with criterions dict:
    result = dict(**result, **run_params_dict)
    # aggregate results from train/eval/test/whole dataset:
    dir_names = ['', 'train', 'eval', 'test']
    for dir_name in dir_names:
        res_dir = curr_dir + '/' + dir_name
        if dir_name == '':
            res_paths = [os.path.join(curr_dir, res_file) for res_file in result_files]
            # result['time'] = datetime.fromtimestamp(os.path.getctime(res_paths[0]))
            dir_name = 'all' if dir_names[-1] in curr_children else dir_names[-1]
        elif dir_name in curr_children:
            res_paths = [os.path.join(res_dir, res_file) for res_file in result_files]
        else:
            continue

        crit_dict = get_criteria_res_from_paths(res_paths, dir_name, traj_len)
        result = dict(**result, **crit_dict)

    return result


def get_criteria_res_from_paths(res_paths, dir_name, traj_len=8):
    crits_dict = {}
    for res_path in res_paths:
        if os.path.exists(res_path):
            # cols of results*.csv file
            fieldnames = get_fieldnames_from_path(res_path)
            # cols of criterions:
            crit_fieldnames = fieldnames[5:]
            crit_dict = {dir_name + '_' + crit_name: [] for crit_name in crit_fieldnames}

            with open(res_path, 'r') as f:
                reader = csv.DictReader(f, fieldnames=fieldnames)
                for row in reader:
                    # interested only in the whole cumulative trajectory result.
                    if row['frame_idx'] == str(traj_len - 1):
                        # fill in the crit values of each trajectory:
                        for crit_name in crit_fieldnames:
                            crit_dict[dir_name + '_' + crit_name].append(float(row[crit_name]))
            # aggregate all values as a mean:
            crit_dict = {crit_name: np.mean(crit_dict[crit_name]) for crit_name in crit_dict}
            # merge result dict with criterions dict:
            crits_dict = dict(**crits_dict, **crit_dict)

    return crits_dict


def list_dirs(dirs):
    """
    DEBUG FUNC
    """
    print('listing directories:')
    for dir in dirs:
        print(dir + '\n')


def sort_filter_print_results(results, sort_crit, filter_crit=None, to_print=True):
    """
    DEBUG FUNC.
    Use sort_crit to sort the results by column, for example by time.
    Use filter_crit to print results that fit criterion, for example: print only stochastic gradient ascent results.
    returns the filtered and sorted results.
    Use to_print=False for sorting and filtering only.
    """
    if to_print:
        print('printing results by sort prams order:')

    filter_crit = filter_crit if filter_crit is not None else {}
    filtered_results = []
    # keys to print in the following order:
    keys_to_print = ['time', 'gradient', 'attack', 'alpha', 't_decay', 'beta', 'data', 'train_data', 'eval_data', 'test_data',
                     'minibatch_size', 'epochs', 'norm', 'momentum', 'rmsprop_decay', 'rmsprop_eps',
                     'train_t_crit', 't_factor', 'train_rot_crit', 'rot_factor', 'train_flow_crit',
                     'flow_factor', 'train_target_t_crit', 'target_t_factor', 'eval_crit',
                     'test_clean_rms', 'test_adv_rms', 'test_adv_ratio_rms']

    if sort_crit != 'none' and len(results) > 0 and sort_crit in results[0]:
        results = sorted(results, key=lambda x: x[sort_crit])

    for result in results:
        accept_result = True
        for filter_key in filter_crit:
            if filter_key in result and result[filter_key] != filter_crit[filter_key]:
                accept_result = False
                break
        if accept_result:
            filtered_results.append(result)
            if to_print:
                print('printing single result:')
                print('{')
                pad = ''
                # filter keys and print in order:
                for key in keys_to_print:
                    if key in result:
                        print(f'{pad}key={key}, value={result[key]}')
                        pad += ' '
                print('}')

    return filtered_results


def parse_results(traj_len=8):
    # root = './results/kitti_custom/tartanvo_1914/VO_adv_project_train_dataset_8_frames/train_attack/universal_attack'
    root = './results/kitti_custom/tartanvo_1914'
    csv_files = ['results_mean_partial_rms.csv', 'results_rms.csv', 'results_target_mean_partial_rms.csv', 'results_target_rms.csv']
    results = []
    dirs_to_visit = [os.path.join(root, dir) for dir in os.listdir(root) if os.path.isdir(os.path.join(root, dir))]
    while len(dirs_to_visit) != 0:
        curr_dir = dirs_to_visit.pop(0)
        curr_children = os.listdir(curr_dir)

        if csv_files[0] in curr_children or 'test' in curr_children:  # got to the bottom of a directory graph:
            results.append(parse_result_single(curr_dir, curr_children, csv_files, traj_len))
        else:
            # expand bfs:
            dirs_to_visit.extend([os.path.join(curr_dir, dir) for dir in curr_children if os.path.isdir(os.path.join(curr_dir, dir))])

    return results


if __name__ == '__main__':
    args = parse_args()
    filter_crit = construct_filter_crit(args.filter_keys, args.filter_vals)
    x_keys = get_x_keys(args.x_keys)
    results = parse_results()
    #print(results)
    results = sort_filter_print_results(results, sort_crit=args.sort_crit, filter_crit=filter_crit, to_print=(not args.dont_print))
    #print(results)
    if args.title != '':  # plot creation was promped.
        plot_by_ylabel(results, x_keys, args.y_label, args.x_label, args.title)
