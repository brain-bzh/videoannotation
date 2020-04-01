import os

#path = '/media/brain/Elec_HD/cneuromod/movie10/stimuli/bourne_supremacy'
#path = '/home/maelle/Database/cneuromod/movie10/stimuli/life'

#fmripath = '/home/brain/nico/sub-01'
#fmripath = '/home/maelle/Database/movie10_parc/sub-01'

#result_path = '/home/maelle/Results'
result_path = '/home/brain/Results'

subjects_path = {'sub_01':'/home/brain/nico/sub-01',
                'sub_02':'/home/brain/nico/sub-02',
                'sub_03':'/home/brain/nico/sub-03',
                'sub_04':'/home/brain/nico/sub-01',
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

                for hidden in [100, 500, 1000, 5000, 10000]:
                    for audiopad in range(3):
                        cmd = 'python3 train_encoding_pretrained.py --movie '+mv_path \
                            +' --subject '+sub_path \
                            +' --audiopad '+str(audiopad) \
                            +' --hidden '+str(hidden) \
                            +' --save_path '+mv_result_path
                        
                        os.system(cmd)
                    



    