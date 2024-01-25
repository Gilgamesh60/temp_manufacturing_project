import numpy as np

exogenous_cost_attr = ['spot_price', 'holding_cost', 'waste_disposal_cost', 'exchange_transport_cost', 'spot_transport_cost']

exogenous_qty_attr = ['recycled_qty', 'waste_qty', 'produced_qty']


def synthetic_exo_data_generator(total_timesteps = 2048, *args, **kwargs):
    data_dict_obj = {}

    #cost attributes sampled for total timesteps from a log-normal distribution
    data_dict_obj['spot_price'] = np.random.normal(500., 50., total_timesteps)
    data_dict_obj['holding_cost'] = 0.15*np.random.normal(500., 50., total_timesteps)
    data_dict_obj['waste_disposal_cost'] = 0.05*np.random.normal(500., 50., total_timesteps)
    #ignoring spot, exchange transport costs

    data_dict_obj['produced_qty'] = np.random.normal(200., 20., total_timesteps)
    #data_dict_obj['recycled_qty'] = 0.05*np.random.normal(200., 20., total_timesteps)
    data_dict_obj['waste_qty'] = 0.10*np.random.normal(200., 20., total_timesteps)
    data_dict_obj['seller_init_inventory'] = 500.
    data_dict_obj['buyer_init_inventory'] = 400.
    return data_dict_obj

