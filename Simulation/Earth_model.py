from Simulation.Parameters import SET_PARAMS
import math
import numpy as np
import Simulation.igrf_utils as igrf_utils
from scipy import interpolate
import csv
from Simulation.utilities import crossProduct
import os
from PyAstronomy import pyasl

# planets = load('de421.bsp')
# earth, moon = planets['earth'], planets['moon']

# ts = load.timescale()
# t = ts.now()
# position = earth.at(t).observe(moon)
# ra, dec, distance = position.radec()

# print(ra)
# print(dec)
# print(distance)

def Coordinat_Transformation(RA, DEC): #, alfa0, delta0, phi):
    #Returns the cartesian coordinates of the star
    #alfa0 and delta0 are the boresight orientation
    #phi is the rotation angle around the boresight (around the Z-axis of star camera coordinates).
    #RA and DEC are the stars Euler coordinates
    ECI = np.array([np.cos(RA)*np.cos(DEC),np.sin(RA)*np.cos(DEC),np.sin(DEC)])
    # m1 = np.array([[np.cos(phi), np.sin(phi), 0], [-1*np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    # m2 = np.array([[1, 0, 0],[0, np.sin(delta0), np.cos(delta0)], [0, -1*np.cos(delta0), np.sin(delta0)]])
    # m3 = np.array([[-1*np.sin(alfa0), np.cos(alfa0), 0], [-1*np.cos(alfa0), -1*np.sin(alfa0), 0], [0, 0, 1]])
    # M = np.matmul(np.matmul(m1,m2),m3)
    return ECI #np.matmul(M,ECI)

IGRF_FILE = r'Simulation/Simulation_data/IGRF13.shc'
igrf = igrf_utils.load_shcfile(IGRF_FILE, None)
f = interpolate.interp1d(igrf.time, igrf.coeffs)

pi = math.pi

def ecef2lla(R):
    # x, y and z are scalars or vectors in meters
    x, y, z = R
    x = np.array([x]).reshape(np.array([x]).shape[-1], 1)
    y = np.array([y]).reshape(np.array([y]).shape[-1], 1)
    z = np.array([z]).reshape(np.array([z]).shape[-1], 1)

    a=6378137
    a_sq=a**2
    e = 8.181919084261345e-2
    e_sq = 6.69437999014e-3

    f = 1/298.257223563
    b = a*(1-f)

    # calculations:
    r = np.sqrt(x**2 + y**2)
    ep_sq  = (a**2-b**2)/b**2
    ee = (a**2-b**2)
    f = (54*b**2)*(z**2)
    g = r**2 + (1 - e_sq)*(z**2) - e_sq*ee*2
    c = (e_sq**2)*f*r**2/(g**3)
    s = (1 + c + np.sqrt(c**2 + 2*c))**(1/3.)
    p = f/(3.*(g**2)*(s + (1./s) + 1)**2)
    q = np.sqrt(1 + 2*p*e_sq**2)
    r_0 = -(p*e_sq*r)/(1+q) + np.sqrt(0.5*(a**2)*(1+(1./q)) - p*(z**2)*(1-e_sq)/(q*(1+q)) - 0.5*p*(r**2))
    u = np.sqrt((r - e_sq*r_0)**2 + z**2)
    v = np.sqrt((r - e_sq*r_0)**2 + (1 - e_sq)*z**2)
    z_0 = (b**2)*z/(a*v)
    h = u*(1 - b**2/(a*v))
    phi = np.arctan((z + ep_sq*z_0)/r)
    lambd = np.arctan2(y, x)


    return phi*180/np.pi, lambd*180/np.pi, h

class Moon:
    def __init__(self):
        self.J_t, self.fr = SET_PARAMS.J_t, SET_PARAMS.fr
        if SET_PARAMS.UsePredeterminedPositionalData:
            self.preData = np.genfromtxt('PreMoon.csv', delimiter=',')
        else:
            if os.path.exists('PreMoon.csv'):
                os.remove('PreMoon.csv')
                print("Remove PreMoon.csv")
            else:
                print("PreMoon.csv does not exist") 

    def moonPosition(self, t):
        if SET_PARAMS.UsePredeterminedPositionalData:
            moonVec = self.preData[t-1]
        else:
            ra, dec, distance, geoLong, geoLat = pyasl.moonpos(self.J_t + self.fr + t/86400)

            moonVec = Coordinat_Transformation(ra[0], dec[0])

            # writing to csv file
            with open('PreMoon.csv', 'a') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(moonVec)

        return moonVec

class orbit:
    def __init__(self):
        self.w_earth = SET_PARAMS.wo
        self.a_G0 = SET_PARAMS.a_G0

    def EFC_to_EIC(self, t):
        a_G = self.w_earth * t + self.a_G0   # angle in radians form the greenwich
        A = np.array(([[np.cos(a_G), -np.sin(a_G), 0.0], [np.sin(a_G), np.cos(a_G), 0.0], [0.0,0.0,1.0]]))
        return A

    def EIC_to_ORC(self, position_vector, velocity_vector):
        # position vector - Height from center of earth, 
        position_vector = position_vector/np.linalg.norm(position_vector)
        velocity_vector = velocity_vector/np.linalg.norm(velocity_vector)
        c = -position_vector   # position vector must be measured by sensors
        b = crossProduct(velocity_vector, position_vector)/(np.linalg.norm(crossProduct(velocity_vector, position_vector)))
        a = crossProduct(b,c)
        A = np.array((a,b,c)).T
        return A

class Earth:   
    def __init__(self):
        self.V = np.zeros((2))
        self.coeffs = f(2021) 
        if SET_PARAMS.UsePredeterminedPositionalData:
            self.preData = np.genfromtxt('PreMagnetometer.csv', delimiter=',')
        else:
            if os.path.exists('PreMagnetometer.csv'):
                os.remove('PreMagnetometer.csv')
                print("Remove PreMagnetometer.csv")
            else:
                print("PreMagnetometer.csv does not exist") 

    def scalar_potential_function(self, latitude, longitude, altitude, t):
        rs = altitude[0,0]
        theta = 90 - latitude[0,0]
        lambda_ = longitude[0,0]

        if SET_PARAMS.UsePredeterminedPositionalData:
            B = self.preData[t-1]
        else:
            B = igrf_utils.synth_values(self.coeffs, rs, theta, lambda_, 10, 3)
            B = np.array((B[0],B[1],B[2]))
            # writing to csv file
            with open('PreMagnetometer.csv', 'a') as csvfile:
                # creating a csv writer object
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(B)

        return B