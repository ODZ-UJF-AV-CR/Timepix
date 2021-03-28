import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import numpy.linalg as la
from scipy import signal
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse
from skimage.morphology import skeletonize
import networkx as nx
from scipy.optimize import leastsq
from scipy import optimize

#A Cluster class
class Cluster:
    def __init__(self, matrix, mass, width, filename, frame_number, track_index, frame_time, acquisition_time):
        self.chip_mass = mass
        self.chip_width = width
        self.filename = filename
        self.frame_number = frame_number
        self.track_index = track_index
        frame_number = str(frame_number).zfill(6)
        track_index = str(track_index).zfill(2)
        self.id = frame_number + '_' + str(track_index)
        self.time = frame_time
        self.acquisition_time = acquisition_time
        self.track = matrix

        self.Calculate_features()

        '''
        self.elevation = self.elevation()
        self.path = self.Path()
        self.dose_Si = self.Dose_Si()
        self.Calculate()
        '''

    def Calculate_features(self):
        # Extracts the features important for the classification of clusters
        x = list(map(int, self.track[0::3]))
        y = list(map(int, self.track[1::3]))
        energy = list(map(float, self.track[2::3]))
        points = list(zip(x, y))

        self.volume = sum(energy)
        self.size = len(energy)
        self.height = max(energy)      

        self.energy_per_pixel = self.volume / self.size
        self.lower_decile = np.quantile(energy, 0.1)
        self.upper_decile = np.quantile(energy, 0.9)
        self.median = np.quantile(energy, 0.5)
        self.hull_width = max(x) - min(x) + 1
        self.hull_height = max(y) - min(y) + 1
        self.hull_area = self.hull_width * self.hull_height
        self.hull_occupancy = self.size / self.hull_area
        self.binary_image = self.Make_binary_image(x, y)
        self.image = self.Make_image(x, y, energy)        
        self.skeleton = skeletonize(self.binary_image)
        self.SCHR = np.sum(self.skeleton) / self.hull_area
        
        '''
        if self.size > 3:
            try:
                self.linearity, self.eigen_ratio = self.Linearize(x, y)
            except:
                self.linearity, self.eigen_ratio = 1, np.inf
        else:
            self.linearity, self.eigen_ratio = 1, np.inf
        '''
        #print(self.binary_image)
        self.average_neighbours = self.Neighbours(points)
        self.centroid = self.Centroid(x, y)
        self.radius = self.Radius(x, y)
        self.diameter = 2 * self.radius
        self.density = self.Density()
        self.line_residual, self.best_fit_theta = self.Line_residual(x, y)
        self.width = self.Width()
        
        best_fit_circle = self.find_best_fit_circle(x, y) # x, y, radius, residuals
        self.curvature_radius = best_fit_circle[2]
        self.circle_residual = best_fit_circle[3]


        '''
        if self.size > 30:
            fig = plt.figure(figsize=(18,9))
            ax1 = fig.add_subplot(2,1,1)
            ax2 = fig.add_subplot(2,1,2)
            ax1.imshow(self.binary_image)
            ax2.imshow(self.skeleton)
            plt.pause(2)
            plt.close()
        '''
        #self.crossing = ss

    def Neighbours(self, points):
        #print(points)
        n_ns = []
        for x, y in points:
            z = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y-1), (x+1, y), (x+1, y+1)]
            n_ns.append(len(set(z).intersection(points)))
        #print(n_ns)
        return np.mean(n_ns)
        
    def Centroid(self, x, y):
        return (np.mean(x), np.mean(y))
        
    def Radius(self, x, y):
        x_cent, y_cent = self.centroid
        x_list = x - x_cent
        y_list = y - y_cent
        x_list = map(lambda x: x ** 2, x_list)
        y_list = map(lambda x: x ** 2, y_list)
        z_list = [a + b for a, b in zip(x_list, y_list)]
        z_list = map(math.sqrt, z_list)
        return max(z_list)
    
    def Density(self):
        # Special case for single pixel: radius equals to 0
        if self.size == 1:
            return 1
        else:
            return self.size / (np.pi * (self.radius ** 2))
    
    def Line_residual(self, x, y):
        # return angle theta anticlockwise from x axis, with the line passing through the cluster centroid

        # Special case for single pixel: horizontal line, completely linear!
        if self.size == 1:
            return (0, 0)
        
        # Otherwise, use leastsq to estimate a line of best fit
        # x axis as inital guess
        first_guess_theta = 0.1
        # Use scipy's regression function to magic this into a good LoBF
        best_fit_theta = leastsq(self.residuals, first_guess_theta, args = (np.array(y), np.array(x)))[0] % (np.pi)
        #print np.degrees(best_fit_theta)
        squiggliness = np.sum([point_line_distance(p, self.centroid, best_fit_theta)**2 for p in zip(x, y)])
        return squiggliness, best_fit_theta[0]
    
    def residuals(self, theta, y, x):
        return point_line_distance((x,y), self.centroid, theta) 
    
    def Width(self):
        if self.size == 1:
            return 0
        else:
            return self.size / (2 * self.diameter)
    
    def find_best_fit_circle(self, x, y):
        if self.size == 1:
            return 0, 0, 0, 0

        # The cluster centroid is often a very bad first guess for the circle centre, so try with a couple of others...
        x_cent, y_cent = self.centroid
        d = self.diameter
        th = np.radians(self.best_fit_theta)
        p1 = (x_cent + d*np.cos(th - (np.pi/2)), y_cent + d*np.sin(th - (np.pi/2)))
        p2 = (x_cent + d*np.cos(th + (np.pi/2)), y_cent + d*np.sin(th + (np.pi/2)))
        test_circles = [leastsq_circle(x, y, test_point) for test_point in [self.centroid, p1, p2]]
        # circle[3] is being minimised
        test_circles.sort(key = lambda circle: circle[3])
        return test_circles[0]    

    def Linearize(self, X, Y):
        a_points = np.array(list(zip(X, Y)))
        A, c = self.mvee(a_points)
        w, _ = la.eig(A)
        linearity = np.max(w) / sum(w)
        eigen_ratio = np.max(w) / np.min(w)
        return linearity, eigen_ratio

    def mvee(self, points, tol = 0.001):
        """
        Find the minimum volume ellipse.
        Return A, c where the equation for the ellipse given in "center form" is
        (x-c).T * A * (x-c) = 1
        """
        points = np.asmatrix(points)
        N, d = points.shape
        Q = np.column_stack((points, np.ones(N))).T
        err = tol + 1.0
        u = np.ones(N) / N
        while err > tol:
            # assert u.sum() == 1 # invariant
            X = Q * np.diag(u) * Q.T
            M = np.diag(Q.T * la.inv(X) * Q)
            jdx = np.argmax(M)
            step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
            new_u = (1 - step_size) * u
            new_u[jdx] += step_size
            err = la.norm(new_u - u)
            u = new_u
        c = u * points
        A = la.inv(points.T * np.diag(u) * points - c.T * c) / d
        return np.asarray(A), np.squeeze(np.asarray(c))

    def Make_image(self, x, y, energy):
        #print(x, y)
        norm_x = np.array(x) - min(x) + 1
        norm_y = np.array(y) - min(y) + 1
        img = np.zeros((self.hull_width + 2, self.hull_height + 2))
        for xx, yy, en in zip(norm_x, norm_y, energy):
            img[xx, yy] = en
        return img
    
    def Make_binary_image(self, x, y):
        #print(x, y)
        norm_x = np.array(x) - min(x) + 1
        norm_y = np.array(y) - min(y) + 1
        img = np.zeros((self.hull_width + 2, self.hull_height + 2))
        for xx, yy in zip(norm_x, norm_y):
            img[xx, yy] = 1
        return img

    def Mean_chord(self):
            return (4 * (14080 * 14080 * width)) / (2 * 14080 * 14080 + 4 * 14080 * width)

    def Path(self):
        condition = 1
        if (condition == 0):
            return self.Mean_chord
        else:
            return self.width / float(np.cos(np.radians(self.elevation)))

    def Dose_Si(self):
        electron_charge = 1.60218E-19
        return (electron_charge * 1000 * self.volume) / self.mass

    def Calculate(self):
        if (self.size > 4):
            self.let_Si = self.volume / self.path
            self.let_H2O = pow(10, -0.1655 + 1.0213 * np.log10(self.let_Si))
            self.ratio = self.let_H2O / self.let_Si
            if (self.let_H2O <= 10):
                self.quality_Factor = 1
            elif (self.let_H2O > 10 and self.let_H2O < 100):
                self.quality_Factor = 0.32 * self.let_H2O - 2.2
            elif (self.let_H2O >= 100):
                self.quality_Factor = 300 * pow(self.let_H2O,-0.5)
            self.dose_H2O = self.dose_Si * 6.2 * self.ratio
            self.dose_Equivalent = self.dose_H2O * self.quality_Factor
        else:
            self.let_Si = 0
            self.let_H2O = 0
            self.ratio = 0
            self.quality_Factor = 1
            self.dose_H2O = self.dose_Si * 1.11
            self.dose_Equivalent = self.dose_H2O

        if (self.size > 4):
            self.let_Si = self.volume / self.path
            self.let_H2O = pow(10, -0.2902 + 1.025 * np.log10(self.let_Si))
            self.ratio = self.let_H2O / self.let_Si
            if (self.let_H2O <= 10):
                self.quality_Factor = 1
            elif (self.let_H2O > 10 and self.let_H2O < 100):
                self.quality_Factor = 0.32 * self.let_H2O - 2.2
            elif (self.let_H2O >= 100):
                self.quality_Factor = 300 * pow(self.let_H2O,-0.5)
            self.dose_H2O = self.dose_Si * 2.328 * self.ratio
            self.dose_Equivalent = self.dose_H2O * self.quality_Factor
        else:
            self.let_Si = 0
            self.let_H2O = 0
            self.ratio = 0
            self.quality_Factor = 1
            self.dose_H2O = self.dose_Si * 1.11
            self.dose_Equivalent = self.dose_H2O

    def return_time(self):
        string = datetime.datetime.utcfromtimestamp(self.time).strftime('%c')
        return datetime.datetime.strptime(string, '%c')

    def get_filename(self):
        return self.filename

    def get_frame_number(self):
        return self.frame_number

    def get_track_index(self):
        return self.track_index

    def get_id(self):
        return self.id
    
    def get_size(self):
        return self.size

    def get_features(self):
        lst = [self.time, self.volume, self.size, self.height, self.energy_per_pixel, self.lower_decile, self.upper_decile, self.median, self.hull_width, self.hull_height, self.hull_area, self.hull_occupancy, self.linearity, self.eigen_ratio, self.SCHR]
        return lst
    
    def get_lucid_features(self):
        return [self.time, self.frame_number, self.track_index, self.volume, self.size, self.height, self.average_neighbours, self.radius, self.diameter, self.density, self.line_residual, self.width, self.curvature_radius, self.circle_residual]

    def get_image(self):
        return self.image
    
    def get_binary_image(self):
        return self.binary_image

    def get_skeleton(self):
        return self.skeleton
    
def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y, centre_estimate):
    centre, ier = optimize.leastsq(f, centre_estimate, args=(x,y))
    xc, yc = centre
    Ri       = calc_R(x, y, *centre)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def distance(point1, point2):
    # Simple 2D distance function using Pythagoras:
    # Calculates the distance between point1 (x, y) and point2 (x, y)
    return np.sqrt(((point2[0] - point1[0])**2) + ((point2[1] - point1[1])**2))

def point_line_distance(point, centroid, theta):
    x1, y1 = centroid
    x2, y2 = (centroid[0] + np.cos(theta), centroid[1] + np.sin(theta))
    x0, y0 = point
    # cheers wikipedia
    return np.fabs( (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 ) / np.sqrt( (y2-y1)**2 + (x2-x1)**2 )