import numpy as np

def Reflection(V, N):
    R = V - 2 * N.T * (np.dot(V, N))
    R = R/np.linalg.norm(R)
    return R


####################################################################################################################
# THIS FUNCTION IS SPECIFIC FOR THE SUN REFLECTION AND DETERMINING THE INTERSECTION BETWEEN THE VECTOR AND A PLANE #
####################################################################################################################
def Intersection(Plane, vector, pointFrom):
    vector = np.array(vector)

    # if (vector[0] < 0 and pointFrom[0] > 0) or (vector[0] > 0 and pointFrom[0] < 0):
    #     print("Vector:", vector)
    #     print(pointFrom)
    #     print(Plane)

    d = np.sum(np.array(Plane[:3]) * vector)



    t0 = np.sum(np.array(Plane[:3]) * np.array(pointFrom))

    t = (Plane[3] - t0)/d


    vectorMoved = vector * t

    position = pointFrom + vectorMoved

    # if (vector[0] < 0 and pointFrom[0] > 0) or (vector[0] > 0 and pointFrom[0] < 0):
    #     print(d)
    #     print(t0)

    #     print(t)

    #     print(position)

    #     print(int(np.sum(position * Plane[:3])))

    #     print(Plane[3])

    #     print(np.sum(position * Plane[:3]) == Plane[3])

    # if round(np.sum(position * Plane[:3]),4) == Plane[3]:
    #     print("within Plane")

    # Plane[0] * (vector[0]) + Plane[1] * (vector[1]) + Plane[2] * (vector[2])

    return position

####################################################################################
# THIS FUNCTION IS USED TO DETERMINE WHETHER A POINT IS BETWEEN TWO PARALLEL LINES #
####################################################################################
def PointWithinParallelLines(Line1, Line2, Point):
    z1 = Line1[0]*Point[0] + Line1[1]
    z2 = Line2[0]*Point[0] + Line2[1]
    # print(Line1, Line2, Point)
    # print(z1, z2)

    if (Point[1] < z1 and Point[1] > z2) or (Point[1] > z1 and Point[1] < z2):
        return True
    else:
        return False

################################
# FIND THE EQUATION FOR A LINE #
################################
def lineEquation(PointFrom, PointTo):
    m = (PointTo[0] - PointFrom[0])/(PointTo[1] - PointFrom[1])
    c = PointTo[0] - m*PointTo[1]
    return m, c

################################
# FIND THE EQUATION FOR A LINE #
################################
def line2Equation(PointFrom, PointTo):
    m = (PointTo[1] - PointFrom[1])/(PointTo[0] - PointFrom[0])
    c = PointTo[1] - m*PointTo[0]
    return m, c

#####################################
# CALCULATE CROS PRODUCT OF VECTORS #
#####################################
def crossProduct(a, b):
    x = a[1]*b[2] - a[2]*b[1]
    y = a[2]*b[0] - a[0]*b[2]
    z = a[0]*b[1] - a[1]*b[0]

    vector = np.array([x, y, z])

    # if np.isnan(vector).any():
    #     print("Nan Vector")

    return vector

def NormalizeVector(Vector):
    norm = np.linalg.norm(Vector)

    if norm != 0:
        Vector = Vector/norm
    
    return Vector