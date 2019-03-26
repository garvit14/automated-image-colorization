import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import sys

# calculate Euclidean distance between two points
def euc_dist(point1, point2):
    dist = np.linalg.norm(point1-point2)
    return dist

# function to get image of given dimensions
def getResizedImage(input_path, dim=256):
    img = cv2.imread(input_path)
    img = cv2.resize(img, (dim, dim))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# K-means clustering
class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    ACTUAL_HEIGHT = None
    ACTUAL_WIDTH = None
    
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self, algo="kmeans"):
    
        #read image
        #img = cv2.imread(self.IMAGE)
        img = getResizedImage(self.IMAGE, 256)
                
        self.ACTUAL_HEIGHT = img.shape[0]
        self.ACTUAL_WIDTH = img.shape[1]
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        if(algo=='kmeans'):
            #using k-means to cluster pixels
            model = KMeans(n_clusters = self.CLUSTERS)
        else:
            
            # using mean-shift for clustering pixels
            bandwidth = estimate_bandwidth(img)
            model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        
        model.fit(img)

        #the cluster centers are our dominant colors.
        self.COLORS = model.cluster_centers_


        #save labels
        self.LABELS = model.labels_
        
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)
    
    # get the point whose color is most similar to centroid
    def getNearestPointTocentroid(self):
        closest, _ = pairwise_distances_argmin_min(self.COLORS, self.IMAGE)
        return self.IMAGE.take(closest, axis=0)
    
    # get coordinates of point which whose color is most similar to input point
    def getNearestPoint(self, point):
        closestPoints = []
        
        for color in self.IMAGE:
            # all points with a euclidean distance less than 1
            if euc_dist(point,color)<1:
                closestPoints.append(color)
        
        # discard duplicates from closestPoints list
        closestPoints = list(np.unique(closestPoints, axis=0))
        
        coordinateList = []
        
        # get x and y coordinates
        for color in closestPoints:
            coordinateList.append(self.getCoordinates(color))
            
        coordinateList.sort()
        
        coordi = coordinateList[len(coordinateList)/2] 
        
        actualImage = self.IMAGE.reshape(self.ACTUAL_HEIGHT, self.ACTUAL_WIDTH, 3)
        
        # return median
        return coordi
            
    # given an RGB value, finds the coordinate of the point with that RGB value
    def getCoordinates(self, point):
        actualImage = self.IMAGE.reshape(self.ACTUAL_HEIGHT, self.ACTUAL_WIDTH, 3)
        templist = []
        for i in range(0,self.ACTUAL_HEIGHT,1):
            for j in range(0,self.ACTUAL_WIDTH,1):
                if euc_dist(actualImage[i,j], point) == 0:
                    templist.append([i,j])
        
        # return median
        return templist[len(templist)/2]
    
    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        
    def calculate_davies_bouldin_score(self):
        return davies_bouldin_score(self.IMAGE, self.LABELS)
        
    def plotHistogram(self):
       
        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)
       
        #create frequency count tables    
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        
        #appending frequencies to cluster centers
        colors = self.COLORS
        
        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()] 
        
        #creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0
        
        #creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500
            
            #getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]
            
            #using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
            start = end	
        
        #display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()

# Analyze k-means for different number of clusters
optimumClusters = 5
validation_scores = []
img = sys.argv[1]
min_clusters = int(sys.argv[2])
max_clusters = int(sys.argv[3])
for i in range(min_clusters, max_clusters, 1):
    clusters = i
    dc = DominantColors(img, clusters)
    colors = dc.dominantColors()
    db_score = dc.calculate_davies_bouldin_score()
    #db_score = dc.calculate_silhoutte_score()
    #db_score = dc.calculate_calinski_harabaz_score()
    validation_scores.append(db_score)
optimumClusters = np.argmin(validation_scores) + min_clusters
print(optimumClusters)



dc = DominantColors(img, optimumClusters)
colors = dc.dominantColors()

# get the points nearest to the centroid
points = dc.getNearestPointTocentroid()

# get coordinates of the points
for point in points:
	coordinate = dc.getNearestPoint(point)
	print("["+str(coordinate[0])+","+str(coordinate[1])+"]")