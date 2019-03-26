import cv2
import os
import matplotlib.pyplot as plt
input_folder = './final_results_quote/ILSVRC2013_DET_val/'
output_folder = './final_results_quote/ILSVRC2013_DET_val_l/'
files = os.listdir(input_folder)
i = 0
# for file in files:
# 	img = cv2.imread(input_folder + file)
# 	img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 	plt.imsave(output_folder + file, img_bw, cmap='gray')
# 	print(i)
# 	i+=1

for file in files:
	img = cv2.imread(input_folder + file)
	img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	plt.imsave(output_folder + file, img_bw[:,:,0], cmap='gray')
	print(i)
	i+=1