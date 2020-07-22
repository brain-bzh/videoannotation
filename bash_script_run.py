import os

#path = '/media/brain/Elec_HD/cneuromod/movie10/stimuli/life'
#path = '/home/maelle/Database/cneuromod/movie10/stimuli/life'

#fmripath = '/home/brain/nico/sub-01'
#fmripath = '/home/maelle/Database/movie10_parc/sub-01'

result_path = '/home/maelle/Results/full_run'
#result_path = '/home/brain/Results/HRF_response'

subjects_path = {'sub_01':'/home/maelle/Database/movie10_parc/sub-01',
                'sub_02':'/home/maelle/Database/movie10_parc/sub-02',
                'sub_03':'/home/maelle/Database/movie10_parc/sub-03',
                'sub_04':'/home/maelle/Database/movie10_parc/sub-04',
                'sub_05':None,
                'sub_06':None
                }

movies_path = {#'life': '/home/maelle/Database/cneuromod/movie10/stimuli/life',
            #'bourne_supremacy':'/home/maelle/Database/cneuromod/movie10/stimuli/bourne_supremacy',
            #'hidden_figures':'/home/maelle/Database/cneuromod/movie10/stimuli/hidden_figures',
            #'wolf_of_wall_street':'/home/maelle/Database/cneuromod/movie10/stimuli/wolf_of_wall_street', 
            'all_movies':'/home/maelle/Database/cneuromod/movie10/stimuli'
            }

def create_directory_if_necessary(path, name):
    new_path = os.path.join(path, name)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    return new_path

#for each subject
for subject, sub_path in subjects_path.items():
    if sub_path is not None : 
        sub_result_path = create_directory_if_necessary(result_path, subject)

        #for each movie (+ one cdt : all movies as inputs for training)
        for movie, mv_path in movies_path.items():
            if mv_path is not None:
                mv_result_path = create_directory_if_necessary(sub_result_path,movie)
            
            #for each model (no hidden layer, 1 hd or 2 hd)
            for model in range(3):
                if model==0:
                    hidden_list = [0]
                else : 
                    hidden_list = [500, 1000, 1500]

                model_result_path = create_directory_if_necessary(mv_result_path, 'model_'+str(model))
                for hidden in hidden_list:
                    hidden_result_path = create_directory_if_necessary(model_result_path, 'hidden_'+str(hidden))
                    print(hidden_result_path)
                    for batch in range(10, 151, 10):
                        cmd = 'python3 train_encoding_pretrained.py --movie '+mv_path \
                            +' --subject '+sub_path \
                            +' --save_path '+hidden_result_path \
                            +' --epochs '+str(5000) \
                            +' --lr '+str(0.01) \
                            +' --delta '+str(1e-1) \
                            +' --epsilon '+str(1e-2) \
                            +' --model '+str(model) \
                            +' --batch '+str(batch) \
                            +' --hidden '+str(hidden)                  
                        os.system(cmd)
                


#python train_encoding_pretrained.py --epochs 5000 --lr 0.01 --batch 70 --delta 1e-1 --epsilon 1e-2 --model 1 --hidden 500
    