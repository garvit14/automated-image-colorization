from data import colorize_image as CI
import matplotlib.pyplot as plt
import numpy as np
import skimage.color as color
import sys
import cv2

# function to get image of given dimensions
def getResizedImage(input_path, dim=256):
    img = cv2.imread(input_path, 1)
    img = cv2.resize(img, (dim, dim))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def put_point(input_ab,mask,loc,p,val):
    # input_ab    2x256x256    current user ab input (will be updated)
    # mask        1x256x256    binary mask of current user input (will be updated)
    # loc         2 tuple      (h,w) of where to put the user input
    # p           scalar       half-patch size
    # val         2 tuple      (a,b) value of user input
    input_ab[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = np.array(val)[:,np.newaxis,np.newaxis]
    mask[:,loc[0]-p:loc[0]+p+1,loc[1]-p:loc[1]+p+1] = 1
    return (input_ab,mask)

# Choose gpu to run the model on
gpu_id = -1

# Initialize colorization class
colorModel = CI.ColorizeImageCaffe(Xd=256)

# Load the model
colorModel.prep_net(gpu_id,'./models/reference_model/deploy_nodist.prototxt','./models/reference_model/model.caffemodel')

# Load the black and white image
colorModel.load_image(sys.argv[1])

# Load the colored reference image
referenceImage = getResizedImage(sys.argv[2])

# extract ab channels of the resized image
referenceImage_lab = color.rgb2lab(referenceImage)

# initialize with no user inputs
input_ab = np.zeros((2,256,256))
mask = np.zeros((1,256,256))

# add points in the image
numberOfPoints = int(sys.argv[3])
for i in range(numberOfPoints):
	coordinate = sys.argv[4+i]
	coordinate = coordinate[1:-1]
	xx = int(list(coordinate.split(','))[0])
	yy = int(list(coordinate.split(','))[1])
	color_to_put = referenceImage_lab[xx,yy]
	(input_ab, mask) = put_point(input_ab,mask,[xx,yy],3,[color_to_put[1], color_to_put[2]])

# call forward
img_out = colorModel.net_forward(input_ab,mask)

# get mask, input image, and result in full resolution
mask_fullres = colorModel.get_img_mask_fullres() # get input mask in full res
img_in_fullres = colorModel.get_input_img_fullres() # get input image in full res
img_out_fullres = colorModel.get_img_fullres() # get image at full resolution

plt.imsave(sys.argv[4+numberOfPoints],img_out_fullres)

# display the output
# intermediate_image = cv2.imread(sys.argv[2])
# intermediate_image = cv2.cvtColor(intermediate_image, cv2.COLOR_BGR2RGB)
# plt.title("BW image / Output of first model / Final Output")
# plt.imshow(np.concatenate((img_in_fullres, intermediate_image, img_out_fullres), axis=1))
# plt.show()