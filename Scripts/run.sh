#!/bin/sh


output1=$(python -W ignore 1_default_colorization.py ./check/imgs/16068.jpg ./check/output_intermediate/16068.jpg)
output2=$(python -W ignore 2_get_coordinates.py $output1 2 16)
python -W ignore 3_interactive_coloring.py ./check/imgs_bw/16068.jpg ./check/output_intermediate/16068.jpg $output2 ./check/output_final/16068.jpg