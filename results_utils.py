import matplotlib
from matplotlib import pyplot as plt 
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
import pickle

def extract_r2_data(path, data='max'):
    r2_key = 'r2_'+data
    r2_state='r2'+data
    r2_data = {'subject':[], 'film':[], 'model':[], 'n_hidden':[], 'batch_size':[], r2_key:[]}
    for root, directory, files in sorted(os.walk(path)):
        if len(files)>0:

            state_files = sorted([name for name in files if name[-3:]=='.pt'])
            for file_model in state_files:
                path_file = os.path.join(root, file_model)

                a = path_file.split('/')
                r2_data['subject'].append(a[5])
                r2_data['film'].append(a[6])
                r2_data['model'].append(a[7])
                batch_size = file_model[-6:-3]
                r2_data['batch_size'].append(int(batch_size.replace('_', '')))

                b = torch.load(path_file, map_location='cpu')
                r2_data['n_hidden'].append(b['nhidden'])
                r2_data['r2_max'].append(b[r2_state])
                print(file_model)

    c = pd.DataFrame.from_dict(r2_data)
    save_path = os.path.join(path, 'r2_dataframe.p')
    pickle.dump(c, open(save_path, "wb" ))
    return c, save_path

path = '/home/maelle/Results/full_run'
#dataframe, path = extract_r2_data(path, data='max')

save_path = os.path.join(path, 'r2_dataframe.p')
dataframe = pickle.load(open(save_path, 'rb'))

for subject in ['sub_01', 'sub_02', 'sub_03', 'sub_04'] : 
    sub = dataframe.loc[dataframe['subject'] == subject]

    f = plt.figure(figsize=(20,30))

    for i, (model, hidden_layer) in enumerate(zip(['model_0','model_1','model_2'], [[0], [1500,500,1000], [1500,500,1000]])):
        sub_model = sub.loc[sub['model']== model]
        ax = plt.subplot(3,1,i+1)
        mask = sub_model['film'].isin(['all_movies'])

        all_movies = sub_model[mask]
        films_alone = sub_model[~mask]

        films = films_alone.film.unique()
        for film in films :
            film_serie = films_alone.loc[(films_alone['film']==film) & (films_alone['n_hidden']==hidden_layer[0])]
            mask_batch_size = film_serie['batch_size'].isin(range(30, 130, 10))
            plouf = film_serie[mask_batch_size]       
            
            plt.plot(range(30, 130, 10), plouf['r2_max'])
        
        plt.axhline(y=0.08, linestyle = 'dotted', alpha=0.8)
        legend = films

    #     for n_hidden in hidden_layer:
    #         temp = films_alone.loc[sub_model['n_hidden']==n_hidden]

    #         r2min = []
    #         r2max = []
    #         r2mean = []
    #         for batch_size in range(30, 130, 10):
    #             temp2 = temp.loc[films_alone['batch_size']==batch_size]
    #             temp3 = temp2['r2_max']
    #             r2min.append(temp3.min())
    #             r2max.append(temp3.max())
    #             r2mean.append(temp3.mean())
            
    #         y_error_high = np.array(r2max) - np.array(r2mean)
    #         y_error_low = np.array(r2mean) - np.array(r2min)
    #         y_error = np.concatenate((y_error_low.reshape(1,-1), y_error_high.reshape(1,-1)), axis = 0)

    #         plt.errorbar(range(30, 130, 10), r2mean, yerr=y_error)



    #     plt.axhline(y=0.08, linestyle = 'dotted', alpha=0.8)
    #     legend = ['0.8 indicator']
    #     legend += (['n_hidden : '+str(n_hidden) for n_hidden in hidden_layer])
    
        plt.legend(legend)
        plt.title('r2 max for differents sizes of batch for 3 films, for {} neurons in {} hidden layer(s) in {}'.format(hidden_layer[0], model, subject))
    
    p = os.path.join(path, '{}_graph_({}_neurons)_for_films.jpg'.format(subject, hidden_layer[0]))
    plt.savefig(p)
    plt.close()