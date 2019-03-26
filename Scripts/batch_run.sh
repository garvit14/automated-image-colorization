#!/bin/sh

input_folder="./final_results_quote/ILSVRC2013_DET_val/"
input_bw_folder="./final_results_quote/ILSVRC2013_DET_val_l/"
output_folder="./final_results_quote/ILSVRC2013_DET_val_output/"
output_intermediate="./final_results_quote/ILSVRC2013_DET_val_output_intermediate/"

files=`ls $input_folder` 

i=0
for file in $files
do
	i=$((i+1))
	echo $i $file

	output1=$(python -W ignore 1_default_colorization.py "$input_folder$file" "$output_intermediate$file")
	output2=$(python -W ignore 2_get_coordinates.py $output1 2 16)
	python -W ignore 3_interactive_coloring.py "$input_bw_folder$file" "$output_intermediate$file" $output2 "$output_folder$file"
done