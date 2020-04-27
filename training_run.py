import os

#path = '/media/brain/Elec_HD/cneuromod/movie10/stimuli/life'
#path = '/home/maelle/Database/cneuromod/movie10/stimuli/life'

#fmripath = '/home/brain/nico/sub-01'
#fmripath = '/home/maelle/Database/movie10_parc/sub-01'

#result_path = '/home/maelle/Results/training_model_test'
result_path = '/home/brain/Results/HRF_response'

subjects_path = {'sub_01':'/home/brain/nico/sub-01',
                'sub_02':'/home/brain/nico/sub-02',
                'sub_03':'/home/brain/nico/sub-03',
                'sub_04':'/home/brain/nico/sub-04',
                'sub_05':None,
                'sub_06':None
                }

movies_path = {'life': '/media/brain/Elec_HD/cneuromod/movie10/stimuli/life',
            'bourne_supremacy':'/media/brain/Elec_HD/cneuromod/movie10/stimuli/bourne_supremacy',
            'hidden_figures':'/media/brain/Elec_HD/cneuromod/movie10/stimuli/hidden_figures',
            'wolf_of_wall_street':'/media/brain/Elec_HD/cneuromod/movie10/stimuli/wolf_of_wall_street', 
            'all_movies':'/media/brain/Elec_HD/cneuromod/movie10/stimuli'
            }

for subject, sub_path in subjects_path.items():

    if sub_path is not None : 
        sub_result_path = os.path.join(result_path, subject)
        if not os.path.exists(sub_result_path):
            os.mkdir(sub_result_path)

        for movie, mv_path in movies_path.items():

            if mv_path is not None:
                mv_result_path = os.path.join(sub_result_path, movie)
                if not os.path.exists(mv_result_path):
                    os.mkdir(mv_result_path)

                print(mv_result_path)

                for hidden in [500, 1000]:
                    for batch in range(10, 151, 20):
                        cmd = 'python3 train_encoding_pretrained.py --movie '+mv_path \
                            +' --subject '+sub_path \
                            +' --save_path '+mv_result_path \
                            +' --epochs '+str(5000) \
                            +' --lr '+str(0.01) \
                            +' --delta '+str(1e-1) \
                            +' --epsilon '+str(1e-2) \
                            +' --model '+str(1) \
                            +' --batch '+str(batch) \
                            +' --hidden '+str(hidden)                  
                        os.system(cmd)
                


#python train_encoding_pretrained.py --epochs 5000 --lr 0.01 --batch 70 --delta 1e-1 --epsilon 1e-2 --model 1 --hidden 500
    