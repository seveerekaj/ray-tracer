import numpy as np

def normalize(v):
    return v / np.sqrt(np.sum(v ** 2))

class PhongObj:
    def __init__(self, kd, ks, ka, od, os, kgls, refl):
        self.kd = kd
        self.ks = ks
        self.ka = ka
        self.od = np.array(od)
        self.os = np.array(os)
        self.kgls = kgls
        self.refl = refl

class Sphere(PhongObj):
    def __init__(self, center, radius, kd, ks, ka, od, os, kgls, refl=0):
        super().__init__(kd, ks, ka, od, os, kgls, refl)
        self.xc = center[0]
        self.yc = center[1]
        self.zc = center[2]
        self.radius = radius

    def getSphereNormal(self, intersectPoint):
        xi, yi, zi = intersectPoint
        return normalize(np.array([
            (xi - self.xc),
            (yi - self.yc),
            (zi - self.zc),
        ]))

class Triangle(PhongObj):
    def __init__(self, v1, v2, v3, kd, ks, ka, od, os, kgls, refl):
        super().__init__(kd, ks, ka, od, os, kgls, refl)
        self.v1 = np.array(v1)
        self.v2 = np.array(v2)
        self.v3 = np.array(v3)
        # compute supporting plane of triangle
        self.n = normalize(np.cross(self.v2 - self.v1, self.v3 - self.v1))
        self.d = np.dot(self.n, self.v1)

