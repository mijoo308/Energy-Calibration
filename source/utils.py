import pandas as pd
import CONFIG
import os
import pickle

def metric_processing(metric_data):
    '''
    metric data (list of dict) : ALL_ECE, ALL_ACC ...
    '''
    ALL_DATA = metric_data
    ALL_AVG_METRIC = None
    ALL_AVG_METRIC = pd.DataFrame.from_dict(ALL_DATA[0], orient='index').transpose()
    for i in range(1,6):
        data = metric_data[i]
        data = pd.DataFrame.from_dict(data, orient='index')
        data = data.mean()
        ALL_AVG_METRIC = pd.concat([ALL_AVG_METRIC, data],axis=1)

    ALL_AVG_METRIC = pd.DataFrame(ALL_AVG_METRIC)
    ALL_AVG_METRIC.columns=['0','1','2','3','4','5']

    return ALL_AVG_METRIC


def ece_result(network, dataset, max_level, result_dir, calib_methods):
    ALL_ECE = []
    for level in range(max_level): # level
        ECE = {}
        for c_type in CONFIG.CORRUPTION_TYPE: # corruption type 
            if level == 0 or level == 6:
                c_type = 'None'
                file_name = f'{network}_{dataset}_{c_type}_{level}_result.pkl'
            else:
                file_name = f'{network}_{dataset}_{c_type}_{level}_result.pkl'

            ECE[c_type]= {}

            ''' load logit file ''' 
            result = None
            result_path = os.path.join(result_dir, file_name)
            with open(result_path, 'rb') as f:
                result = pickle.load(f)

            for calib in calib_methods:
                ECE[c_type][calib] = result['ECE'][calib]
            
            if level == 0:
                break

        ALL_ECE.append(ECE)
    
    ALL_ECE = metric_processing(ALL_ECE)

    ALL_ECE = ALL_ECE.applymap(lambda x: round(x, 2))
    ALL_ECE['Avg.'] = ALL_ECE.mean(axis=1).round(2)
    ALL_ECE = ALL_ECE.applymap(lambda x: f"{x:.2f}")

    return ALL_ECE
