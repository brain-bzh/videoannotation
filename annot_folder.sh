FILM_DIR="$1/"

for OUTPUT in $(ls $FILM_DIR*.mkv)
do
	python run_generate_srt_fm.py $OUTPUT
done
done