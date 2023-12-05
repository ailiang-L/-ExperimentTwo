import yaml
import numpy as np


def load_parameters():
    with open('../config/parameters.yaml', 'r') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    config['communication_config']['eta1'] = np.float32(config['communication_config']['eta1'])
    config['communication_config']['eta2'] = np.float32(config['communication_config']['eta2'])
    config['communication_config']['eta_LoS'] = np.float32(config['communication_config']['eta_LoS'])
    config['communication_config']['eta_NLoS'] = np.float32(config['communication_config']['eta_NLoS'])
    config['communication_config']['fc'] = np.float32(config['communication_config']['fc'])
    config['communication_config']['eta3'] = np.float32(config['communication_config']['eta3'])
    config['communication_config']['eta4'] = np.float32(config['communication_config']['eta4'])
    config['communication_config']['d0'] = np.float32(config['communication_config']['d0'])
    config['communication_config']['Lcvv'] = np.float32(config['communication_config']['Lcvv'])
    config['communication_config']['L0vv'] = np.float32(config['communication_config']['L0vv'])
    config['communication_config']['p_noise'] = np.float32(config['communication_config']['p_noise'])

    config['vehicle_path_config']['run_time'] = np.float32(config['vehicle_path_config']['run_time'])
    config['vehicle_path_config']['car_speed'] = np.float32(config['vehicle_path_config']['car_speed'])
    config['vehicle_path_config']['time_slot'] = np.float32(config['vehicle_path_config']['time_slot'])
    config['vehicle_path_config']['forward_probability'] = np.float32(
        config['vehicle_path_config']['forward_probability'])

    config['uav_config']['pos'][0] = np.array(config['uav_config']['pos'][0], dtype=np.float32)
    config['uav_config']['pos'][1] = np.array(config['uav_config']['pos'][1], dtype=np.float32)
    config['uav_config']['pos'][2] = np.array(config['uav_config']['pos'][2], dtype=np.float32)
    config['uav_config']['pos'][3] = np.array(config['uav_config']['pos'][3], dtype=np.float32)

    config['uav_config']['bandwidth'] = np.float32(config['uav_config']['bandwidth'])
    config['uav_config']['E_n'] = np.float32(config['uav_config']['E_n'])
    config['uav_config']['P_n'] = np.float32(config['uav_config']['P_n'])
    config['uav_config']['w'] = np.float32(config['uav_config']['w'])
    config['uav_config']['C_n'] = np.float32(config['uav_config']['C_n'])

    config['vehicle_config']['bandwidth'] = np.float32(config['vehicle_config']['bandwidth'])
    config['vehicle_config']['E_n'] = np.float32(config['vehicle_config']['E_n'])
    config['vehicle_config']['P_n'] = np.float32(config['vehicle_config']['P_n'])
    config['vehicle_config']['w'] = np.float32(config['vehicle_config']['w'])
    config['vehicle_config']['C_n'] = np.float32(config['vehicle_config']['C_n'])

    config['node_choose_config']['e_weight'] = np.float32(config['node_choose_config']['e_weight'])
    config['node_choose_config']['t_weight'] = np.float32(config['node_choose_config']['t_weight'])

    config['reward_config']['e_weight'] = np.float32(config['reward_config']['e_weight'])
    config['reward_config']['t_weight'] = np.float32(config['reward_config']['t_weight'])

    config['task_split_granularity'] = np.array(config['task_split_granularity'], dtype=np.float32)
    return config
