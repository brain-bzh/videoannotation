import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
import pickle

from nilearn.plotting import plot_stat_map
from nilearn.regions import signals_to_img_labels


def extract_r2_data_from_pt(path, data='max'):
    r2_key = 'r2_'+data
    r2_state='r2'+data
    r2_data = {'subject':[], 'film':[], 'model':[], 'n_hidden':[], 'batch_size':[], r2_key:[]}
    for root, directory, files in sorted(os.walk(path)):
        if len(files)>0:

            state_files = sorted([name for name in files if name[-3:]=='.pt'])
            for file_model in state_files:
                path_file = os.path.join(root, file_model)

                a = path_file.split('/')
                if a[6] != 'all_movies':
                    r2_data['subject'].append(a[5])
                    r2_data['film'].append(a[6])
                    r2_data['model'].append(a[7])
                    batch_size = file_model[-6:-3]
                    r2_data['batch_size'].append(int(batch_size.replace('_', '')))

                    b = torch.load(path_file, map_location='cpu')
                    r2_data['n_hidden'].append(b['nhidden'])
                    r2_data['r2_max'].append(b[r2_state])
                    print(path_file)

    c = pd.DataFrame.from_dict(r2_data)
    save_path = os.path.join(path, 'r2_dataframe.p')
    pickle.dump(c, open(save_path, "wb" ))
    return c, save_path

def select_by_extension(path, extension='.nii.gz'):
    files_dict = {'path':[], 'subject':[], 'film':[], 'model':[], 'n_hidden':[], 'batch_size':[]}
    ext_size = len(extension)
    for root, directory, files in sorted(os.walk(path)):
        if len(files)>0:
            state_files = sorted([name for name in files if name[-ext_size:]==extension])
            for file_model in state_files:
                path_file = os.path.join(root, file_model)
                args_dir = path_file.split('/')
                args_file = []

                files_dict['subject'].append(args_dir[5])
                files_dict['film'].append(args_dir[6])
                files_dict['model'].append(args_dir[7])
                n_hidden = args_dir[8]
                files_dict['n_hidden'].append(int(n_hidden.replace('hidden_', '')))
                batch_size = file_model[-10:-7]
                files_dict['batch_size'].append(int(batch_size.replace('_', '')))
                files_dict['path'].append(path_file)

    c = pd.DataFrame.from_dict(files_dict)
    save_path = os.path.join(path, 'niigz_dataframe.p')
    pickle.dump(c, open(save_path, "wb" ))
    return c, save_path

def plot_by_film(dataframe, data_range) : 
    films = dataframe.film.unique()
    for film in films :
        film_df = dataframe.loc[(dataframe['film']==film) & (dataframe['n_hidden']==hidden_layer[0])]
        mask_batch_size = film_df['batch_size'].isin(data_range)
        df_in_range  = film_df[mask_batch_size]       
        plt.plot(range(30, 130, 10), df_in_range['r2_max'])
    
    plt.axhline(y=0.08, linestyle = 'dotted', alpha=0.8)
    legend = films
    return legend

def plot_by_hidden_layer_size(dataframe, hidden_layer_size, batch_range, error_plot = True):
    for n_hidden in hidden_layer_size:
        temp = dataframe.loc[dataframe['n_hidden']==n_hidden]

        r2min = []
        r2max = []
        r2mean = []
        for batch_size in batch_range:
            temp2 = temp.loc[temp['batch_size']==batch_size]
            temp3 = temp2['r2_max']
            r2min.append(temp3.min())
            r2max.append(temp3.max())
            r2mean.append(temp3.mean())
        
        y_error_high = np.array(r2max) - np.array(r2mean)
        y_error_low = np.array(r2mean) - np.array(r2min)
        y_error = np.concatenate((y_error_low.reshape(1,-1), y_error_high.reshape(1,-1)), axis = 0)
        
        if error_plot : 
            plt.errorbar(batch_range, r2mean, yerr=y_error)
        else : 
            plt.plot(batch_size, r2mean)

    plt.axhline(y=0.08, linestyle = 'dotted', alpha=0.8)
    legend = ['0.8 indicator']
    legend += (['n_hidden : '+str(n_hidden) for n_hidden in hidden_layer])
    return legend

path = '/home/maelle/Results/full_run'
#dataframe, save_path = extract_r2_data_from_pt(path, data='max')
dataframe, save_path = select_by_extension(path, extension='.nii.gz')

print(dataframe)

save_path = os.path.join(path, 'r2_dataframe.p')
dataframe = pickle.load(open(save_path, 'rb'))

for subject in ['sub_01', 'sub_02', 'sub_03', 'sub_04'] : 
    sub = dataframe.loc[dataframe['subject'] == subject]

    f = plt.figure(figsize=(20,30))

    for i, (model, hidden_layer) in enumerate(zip(['model_0','model_1','model_2'], [[0], [500,1000,1500], [500,1000,1500]])):
        sub_model = sub.loc[sub['model']== model]
        ax = plt.subplot(3,1,i+1)
        mask = sub_model['film'].isin(['all_movies'])

        all_movies = sub_model[mask]
        films_alone = sub_model[~mask]

        #legend = plot_by_film(films_alone, range(30, 130, 10))
        #legend = plot_by_hidden_layer_size(films_alone, hidden_layer, range(30,130,10))
    

        r2_img = signals_to_img_labels(r2model.reshape(1,-1),mistroifile)

        # plot_stat_map(r2_img,display_mode='z',cut_coords=8,figure=f,axes=ax)
        # f.savefig(str_bestmodel_plot)
        # r2_img.to_filename(str_bestmodel_nii)
        # plt.close()

        # plt.legend(legend)
        
        #plt.title('r2 max for differents sizes of batch for 3 films, for {} neurons in {} hidden layer(s) in {}'.format(hidden_layer[0], model, subject))
        #plt.title('r2 max for differents sizes of batch for 3 films, for {} hidden layer(s) in {}'.format(model, subject))

    #save_name = '{}_graph_({}_neurons)_for_films_test.jpg'.format(subject, hidden_layer[0])
    #save_name = '{}_graph_with_error.jpg'.format(subject)

    # p = os.path.join(path, save_name)
    # plt.savefig(p)
    # plt.close()