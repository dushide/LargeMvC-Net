import datetime, pytz, time
current_time = datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d_%H:%M:%S')

# pip install pyyaml
import json

def load_config(config_name, verbose=True):
    if '.yaml' not in config_name:
        config_name += '.yaml'
    with open(config_name, 'r') as f:
        f_str = f.read()
        dic = yaml.load(f_str,Loader=yaml.FullLoader)
        dic['experiment_id'] = current_time
        # check_config(dic)
        # 如果 verbose 参数为 True，则将字典对象转换为格式化的 JSON 字符串，并使用 print 函数打印输出，最后，返回配置字典 dic。
        if verbose:
            js = json.dumps(dic, sort_keys=True, indent=4, separators=(',', ':'))
            print(js)
        return dic

def save_config(config_name, config):
    if '.yaml' not in config_name:
        config_name = config_name + '.yaml'
    with open(config_name, 'w') as f:
        f.write(yaml.dump(config, default_flow_style=False))
        print('config successfully saved to '+config_name)

def check_config(config):
    
    assert type(config['view_size']) == int
    assert type(config['n_clusters']) == int
    assert type(config['epoch']) == int


    return True