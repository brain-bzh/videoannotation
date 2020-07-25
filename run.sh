### Usage : put the neuromod stimuli folder as argument of this script 
BASE_DIR=$1

for FILM in $(ls $BASE_DIR)
do 
FILM_DIR="$BASE_DIR$FILM/"

for OUTPUT in $(ls $FILM_DIR*.mkv)
do
	python run_generate_proba_fm_gpu.py $OUTPUT
done
done