import matplotlib.pyplot as plt
import numpy as np
import math
import gmplot
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
from statistics import mean
from madgwickahrs import MadgwickAHRS

# Notation used coming from: https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/
def prediction(X_hat_t_1, P_t_1, F_t, B_t, U_t, Q_t):
    X_hat_t = F_t.dot(X_hat_t_1) + (B_t.dot(U_t))
    P_t = np.diag(np.diag(F_t.dot(P_t_1).dot(F_t.transpose()))) + Q_t
    return X_hat_t, P_t
def update(X_hat_t, P_t, Z_t, R_t, H_t):
    K_prime = P_t.dot(H_t.transpose()).dot(np.linalg.inv(H_t.dot(P_t).dot(H_t.transpose()) + R_t))
    X_t = X_hat_t + K_prime.dot(Z_t - H_t.dot(X_hat_t))
    P_t = P_t - K_prime.dot(H_t).dot(P_t)
    return X_t, P_t
def rotaitonMatrix(heading, attitude, bank):
    '''
    :returns: rotation array in numpy format
    [m00 m01 m02]
    [m10 m11 m12]
    [m20 m21 m22]
    '''
    ch = math.cos(heading)
    sh = math.sin(heading)
    ca = math.cos(attitude)
    sa = math.sin(attitude)
    cb = math.cos(bank)
    sb = math.sin(bank)
    m00 = ch * ca
    m01 = sh * sb - ch * sa * cb
    m02 = ch * sa * sb + sh * cb
    m10 = sa
    m11 = ca * cb
    m12 = -ca * sb
    m20 = -sh * ca
    m21 = sh * sa * cb + ch * sb
    m22 = -sh * sa * sb + ch * cb
    return np.array([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
def getDistance(lat1, lon1, lat2, lon2):
    '''
    refernce: http://code.activestate.com/recipes/577594-gps-distance-and-bearing-between-two-gps-points/
    '''
    R = 6371.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance * 1000.0
def getBearing(lat1,lon1,lat2,lon2):
    '''
    reference: http://code.activestate.com/recipes/577594-gps-distance-and-bearing-between-two-gps-points/
    '''
    dLon = lon2 - lon1
    y = sin(dLon) * cos(lat2)
    x = cos(lat1) * sin(lat2) \
        - sin(lat1) * cos(lat2) * cos(dLon)
    return atan2(y, x)

samples = 30000

# Defining file locations
gpsDatafileName = "sensor_data/gpsValues.txt"
accDatafileName = "sensor_data/accelerometerValues.txt"
gyroDatafileName = "sensor_data/gyroscopeValues.txt"
magDatafileName = "sensor_data/magneticValues.txt"

# defining file handles
gpsValuesFileHandle = open(gpsDatafileName, 'r')
accValuesFileHandle = open(accDatafileName, 'r')
gyroValuesFileHandle = open(gyroDatafileName, 'r')
magValuesFileHandle = open(magDatafileName, 'r')

# Initializing lists to hold data
GPSDATA, gpsTimeStamp, gpsLatitude, gpsLongitude, gpsAltitude, gpsBearing, gpsSpeed = ([] for i in range(7))
ACCDATA, accTimeStamp, accX, accY, accZ = ([] for i in range(5))
GYRODATA, gyroTimeStamp, gyroX, gyroY, gyroZ = ([] for i in range(5))
MAGDATA, magTimeStamp, magX, magY, magZ = ([] for i in range(5))

print("Reading files...")
# reading text files
for gpsData in gpsValuesFileHandle:
    gpsData = gpsData.rstrip()
    GPSDATA.append(gpsData.split(','))
for i in range(6, len(GPSDATA)):  # ignoring erronious data points
    gpsTimeStamp.append(float(GPSDATA[i][0]))
    gpsLatitude.append(float(GPSDATA[i][1]))
    gpsLongitude.append(float(GPSDATA[i][2]))
    gpsBearing.append(float(GPSDATA[i][4]))
for accData in accValuesFileHandle:
    accData = accData.rstrip()
    ACCDATA.append(accData.split(','))
for i in range(1, len(ACCDATA)):
    accTimeStamp.append(float(ACCDATA[i][0]))
    accX.append(float(ACCDATA[i][1]))
    accY.append(float(ACCDATA[i][2]))
    accZ.append(float(ACCDATA[i][3]))
for gyroData in gyroValuesFileHandle:
    gyroData = gyroData.rstrip()
    GYRODATA.append(gyroData.split(','))
for i in range(1, len(GYRODATA)):
    gyroTimeStamp.append(float(GYRODATA[i][0]))
    gyroX.append(float(GYRODATA[i][1]))
    gyroY.append(float(GYRODATA[i][2]))
    gyroZ.append(float(GYRODATA[i][3]))
for magData in magValuesFileHandle:
    magData = magData.rstrip()
    MAGDATA.append(magData.split(','))
for i in range(1, len(MAGDATA)):
    magTimeStamp.append(float(MAGDATA[i][0]))
    magX.append(float(MAGDATA[i][1]))
    magY.append(float(MAGDATA[i][2]))
    magZ.append(float(MAGDATA[i][3]))

print("Interpolating...")
# interpolating data to 30000 points
# using numpy linear interpolation
gpsTimeStampInterp = np.linspace(gpsTimeStamp[0], gpsTimeStamp[len(gpsTimeStamp) - 1], samples)
gpsLatitudeInterpolated = np.interp(gpsTimeStampInterp, gpsTimeStamp, gpsLatitude)
gpsLongitudeInterpolated = np.interp(gpsTimeStampInterp, gpsTimeStamp, gpsLongitude)
gpsBearingInterpolated = np.interp(gpsTimeStampInterp, gpsTimeStamp, gpsBearing)

# taking GPS time stamps as they GPS is absolute data
accTimeStampInterp = np.linspace(gpsTimeStampInterp[0], gpsTimeStampInterp[len(gpsTimeStampInterp) - 1], samples)
accXInterpolated = np.interp(accTimeStampInterp, accTimeStamp, accX)
accYInterpolated = np.interp(accTimeStampInterp, accTimeStamp, accY)
accZInterpolated = np.interp(accTimeStampInterp, accTimeStamp, accZ)

gyroTimeStampInterp = np.linspace(gpsTimeStampInterp[0], gpsTimeStampInterp[len(gpsTimeStampInterp) - 1], samples)
gyroXInterpolated = np.interp(gyroTimeStampInterp, gyroTimeStamp, gyroX)
gyroYInterpolated = np.interp(gyroTimeStampInterp, gyroTimeStamp, gyroY)
gyroZInterpolated = np.interp(gyroTimeStampInterp, gyroTimeStamp, gyroZ)

magTimeStampInterp = np.linspace(gpsTimeStampInterp[0], gpsTimeStampInterp[len(gpsTimeStampInterp) - 1], samples)
magXInterpolated = np.interp(magTimeStampInterp, magTimeStamp, magX)
magYInterpolated = np.interp(magTimeStampInterp, magTimeStamp, magY)
magZInterpolated = np.interp(magTimeStampInterp, magTimeStamp, magZ)

dt = [(gpsTimeStampInterp[i + 1] - gpsTimeStampInterp[i]) * 0.000000001 for i in range(len(gpsTimeStampInterp) - 1)]
dt.insert(0, 0)

accXAbsolute = []
accYAbsolute = []

print("Calculating absolute acc values...")
heading = MadgwickAHRS(sampleperiod=mean(dt))
for i in range(samples):
    gyroscope = [gyroZInterpolated[i], gyroYInterpolated[i], gyroXInterpolated[i]]
    accelerometer = [accZInterpolated[i], accYInterpolated[i], accXInterpolated[i]]
    magnetometer = [magZInterpolated[i], magYInterpolated[i], magXInterpolated[i]]
    heading.update(gyroscope, accelerometer, magnetometer)
    ahrs = heading.quaternion.to_euler_angles()
    roll = ahrs[0]
    pitch = ahrs[1]
    yaw = ahrs[2] + (3.0 * (math.pi / 180.0))  # adding magenetic declination
    ACC = np.array([[accZ[i]], [accY[i]], [accX[i]]])
    ACCABS = np.linalg.inv(rotaitonMatrix(yaw, pitch, roll)).dot(ACC)
    accXAbsolute.append(-1 * ACCABS[0, 0])
    accYAbsolute.append(-1 * ACCABS[1, 0])

# Transition matrix
F_t = np.array([[1, 0, dt[0], 0], [0, 1, 0, dt[0]], [0, 0, 1, 0], [0, 0, 0, 1]])
# Initial State cov
P_t = np.identity(4) * 400
# Process cov
Q_t = np.array([[1000, 0, 100, 0], [0, 1000, 0, 100], [100, 0, 1000, 0], [0, 100, 0, 1000]]) * 0.65
# Control matrix
B_t = np.array([[0.5 * dt[0] ** 2, 0, 0, 0], [0, 0.5 * dt[0] ** 2, 0, 0], [0, 0, dt[0], 0], [0, 0, 0, dt[0]]])
# Control vector
U_t = np.array([[accXAbsolute[0]], [accYAbsolute[0]], [accXAbsolute[0]], [accYAbsolute[0]]])
# Measurment Matrix
H_t = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
# Measurment cov
R_t = np.identity(2)
# Initial State
X_hat_t = np.array([[gpsLongitudeInterpolated[0]], [gpsLatitudeInterpolated[0]], [0], [0]])

Xfilter = []
Yfilter = []

print("Running kalman filter...")
for i in range(len(dt)):
    X_hat_t, P_hat_t = prediction(X_hat_t, P_t, F_t, B_t, U_t, Q_t)  # STATE PREDICTION (IMU)
    Z_t = np.array([[gpsLongitudeInterpolated[i]], [gpsLatitudeInterpolated[i]]])
    Z_t = Z_t.reshape(Z_t.shape[0], -1)
    X_t, P_t = update(X_hat_t, P_hat_t, Z_t, R_t, H_t)  # MEASUREMENT UPDATE (GPS)
    X_hat_t = X_t
    P_hat_t = P_t
    F_t = np.array([[1, 0, dt[i], 0], [0, 1, 0, dt[i]], [0, 0, 1, 0], [0, 0, 0, 1]])
    B_t = np.array([[0.5 * dt[i] ** 2, 0, 0, 0], [0, 0.5 * dt[i] ** 2, 0, 0], [0, 0, dt[i], 0], [0, 0, 0, dt[i]]])
    U_t = np.array([[accXAbsolute[i]], [accYAbsolute[i]], [accXAbsolute[i]], [accYAbsolute[i]]])
    Xfilter.append(X_t[0, 0])
    Yfilter.append(X_t[1, 0])

# reducing data points to 100 for plotting on google maps
dtReduced = np.linspace(gpsTimeStamp[0], gpsTimeStamp[len(gpsTimeStamp) - 1], 100)
XfilterInterp = np.interp(dtReduced, gpsTimeStampInterp, Xfilter)
YfilterInterp = np.interp(dtReduced, gpsTimeStampInterp, Yfilter)

print("Plotting on google maps...")
gmap1 = gmplot.GoogleMapPlotter(gpsLatitude[0], gpsLongitude[0], 13)
gmap1.apikey = "AIzaSyDvvwPIEA8T9IUxPKaRZ6gp2f6xRtBYICU"
gmap1.scatter(gpsLatitude, gpsLongitude, color='r', size=0.3, marker=False)
gmap1.scatter(YfilterInterp, XfilterInterp, color='g', size=0.6, marker=False)
gmap1.draw("results.html")

print("Done...")

while True:
    ans = input("Get distance / bearing data? [Y-N]:")
    if ans == 'Y':
        try:
            f1 = input("Enter Frame 1 ID [1-100]:")
            f2 = input("Enter Frame 1 ID [1-100]:")
            d = getDistance(YfilterInterp[int(f1)],XfilterInterp[int(f1)],YfilterInterp[int(f2)],XfilterInterp[int(f2)])
            b = getBearing(YfilterInterp[int(f1)],XfilterInterp[int(f1)],YfilterInterp[int(f2)],XfilterInterp[int(f2)])
            print("Distance:", d)
            print("Bearing:", (b * 180.0 / math.pi))
        except Exception as e:
            print(e)
            print("Error, try again...")
            break
    elif ans == 'N':
        print("Exiting...")
        break
    else:
        print("Invalid Input")
        break
