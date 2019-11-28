from math import sin, cos, sqrt, atan2, radians
import math
import matplotlib.pyplot as plt

def distance(lat1,lon1,lat2,lon2):
    R = 6371.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))    
    distance = R * c
    return distance * 1000.0

gpsDatafileName = "C:/Users/karthik/Desktop/Assignment/sensor_data/gpsValues.txt"
gpsValuesFileHandle = open(gpsDatafileName,'r')
GPSDATA,gpsTimeStamp,gpsLatitude,gpsLongitude,gpsAltitude,gpsBearing,gpsSpeed = ([] for i in range(7))
for gpsData in gpsValuesFileHandle:
    gpsData = gpsData.rstrip()
    GPSDATA.append(gpsData.split(','))
for i in range(1,len(GPSDATA)):  #ignoring erronious data points
    gpsTimeStamp.append(float(GPSDATA[i][0]))
    gpsLatitude.append(float(GPSDATA[i][1]))
    gpsLongitude.append(float(GPSDATA[i][2]))
    gpsBearing.append(float(GPSDATA[i][4]))

x = [0]
y = [0]

for i in range(len(gpsTimeStamp)-1):
    d = distance(gpsLatitude[i],gpsLongitude[i],gpsLatitude[i+1],gpsLongitude[i+1])
    x.append(d*cos(((gpsBearing[i] + gpsBearing[i+1])/2.0)*(math.pi/180.0)) + x[-1])
    y.append(d*sin(((gpsBearing[i] + gpsBearing[i+1])/2.0)*(math.pi/180.0)) + y[-1])

plt.scatter(y,x)
plt.plot(y,x)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

