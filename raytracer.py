from PIL import Image
import numpy as np
from objects import Sphere, Triangle

NUM_CHANNELS_RGB = 3
RAY_START_OFFSET = np.power(2.0, -16)
MAX_RAY_TREE_DEPTH = 5

def normalize(v):
    return v / np.sqrt(np.sum(v ** 2))

def getRayIntersect(rayStart, rayDir, t):
    return rayStart + rayDir * t

class RayTracer:
    def __init__(self, direcToLight, lightColor, ambientLight, backgroundColor):
        self.direcToLight = normalize(np.array(direcToLight))
        self.lightColor = np.array(lightColor)
        self.ambientLight = np.array(ambientLight)
        self.backgroundColor = np.array(backgroundColor)
        self.I_MAX = 500
        self.J_MAX = 500
        self.U_MIN = -0.55
        self.U_MAX = 0.55
        self.V_MIN = -0.55
        self.V_MAX = 0.55
        self.LOOK_FROM = np.array([0, 0, 1])
        self.imageArray = np.zeros((self.I_MAX, self.J_MAX, NUM_CHANNELS_RGB), dtype=np.float64)
        self.spheres = []
        self.triangles = []

    def withSpheres(self, *args):
        self.spheres.extend(args)
        return self

    def withTriangles(self, *args):
        self.triangles.extend(args)
        return self

    def viewportToWindow(self, i, j):
        u = i * ((self.U_MAX - self.U_MIN) / self.I_MAX) + self.U_MIN
        v = j * ((self.V_MAX - self.V_MIN) / self.J_MAX) + self.V_MIN
        return np.array([u, v, 0])

    def setWindowSize(self, width, height):
        self.I_MAX = width
        self.J_MAX = height
        return self

    def setCamera(self, fov, lookFrom):
        self.LOOK_FROM = np.array(lookFrom)
        self.U_MAX = self.V_MAX = np.tan(np.radians(fov / 2)) * lookFrom[2]
        self.U_MIN = self.V_MIN = -self.U_MAX
        return self

    def saveImage(self, fileName):
        # first divide by max to scale vals between [0,1] then multiply by 255 to scale vals between [0,255], then cast to uint8 for use with PIL.Image
        # self.imageArray = (self.imageArray / np.max(self.imageArray) * 255).astype('uint8')
        # print(f'max color: {np.max(self.imageArray)}')
        self.imageArray = (np.clip(self.imageArray, 0, 1) * 255).astype('uint8')
        # rotate 90 degrees because viewport origin (0,0) is bottom left, but PIL Image expects origin (0,0) to be top left
        Image.fromarray(self.imageArray).rotate(90).save(fileName)

    def raytrace(self):
        self.imageArray = np.zeros((self.I_MAX, self.J_MAX, NUM_CHANNELS_RGB), dtype=np.float64)
        for i in range(self.I_MAX):
            for j in range(self.J_MAX):
                ray = normalize(self.viewportToWindow(i, j) - self.LOOK_FROM)
                nearestTval, nearestObj = self.getNearestSphere(self.LOOK_FROM, ray)
                nearestTval, nearestObj = self.getNearestTriangle(nearestTval, nearestObj, self.LOOK_FROM, ray)

                # compute pixel color with phong model
                self.imageArray[i, j] = self.getColor(nearestTval, nearestObj, self.LOOK_FROM, ray, 0)
        return self

    def getColor(self, t, obj, rayStart, rayDir, rayTreeDepth):
        if rayTreeDepth > MAX_RAY_TREE_DEPTH:
            return 0
        if obj is None:
            return self.backgroundColor

        rayIntersect = getRayIntersect(rayStart, rayDir, t)
        if isinstance(obj, Sphere):
            normal = obj.getSphereNormal(rayIntersect)
        else:  # if it's not a sphere, it must be a triangle
            normal = obj.n

        r = 2 * np.dot(self.direcToLight, normal) * normal - self.direcToLight
        v = rayDir * -1
        ambient = np.multiply(obj.ka * self.ambientLight, obj.od)
        diffuse = np.multiply(obj.kd * self.lightColor, obj.od) * max(0, np.dot(normal, self.direcToLight))
        specular = np.multiply(obj.ks * self.lightColor, obj.os) * np.power(max(0, np.dot(v, r)), obj.kgls)

        # send reflection ray recursively
        reflRayDir = normalize(rayDir - 2 * normal * np.dot(rayDir, normal))
        reflRayStart = getRayIntersect(rayIntersect, reflRayDir, RAY_START_OFFSET)
        nearestTval, nearestObj = self.getNearestSphere(reflRayStart, reflRayDir)
        nearestTval, nearestObj = self.getNearestTriangle(nearestTval, nearestObj, reflRayStart, reflRayDir)

        color = ambient + obj.refl * self.getColor(nearestTval, nearestObj, reflRayStart, reflRayDir, rayTreeDepth + 1)
        if self.isInShadow(rayIntersect, normal):
            return color
        return color + diffuse + specular

    def isInShadow(self, intersectPoint, normal):
        # offset intersection point by very small amount in direction of surface normal
        shadowRayStart = getRayIntersect(intersectPoint, normal, RAY_START_OFFSET)
        nearestTval, nearestObj = self.getNearestSphere(shadowRayStart, self.direcToLight)
        if nearestObj is not None:
            return True
        nearestTval, nearestObj = self.getNearestTriangle(nearestTval, nearestObj, shadowRayStart, self.direcToLight)
        return nearestObj is not None

    def getNearestTriangle(self, nearestTval, nearestObj, rayStart, rayDir):
        for triangle in self.triangles:
            # compute intersection point of ray with plane (and get T)
            tDivisor = np.dot(triangle.n, rayDir)
            if tDivisor == 0: # ray is parallel to plane
                continue
            t = (triangle.d - np.dot(triangle.n, rayStart)) / tDivisor
            # plane is behind ray or we have already found some other object that is closer
            if (t < 0 or t > nearestTval):
                continue

            # run inside-outside test to determine if intersection point lies inside triangle
            q = np.array(getRayIntersect(rayStart, rayDir, t))
            if (np.dot(np.cross(triangle.v2 - triangle.v1, q - triangle.v1), triangle.n) >= 0 and
                    np.dot(np.cross(triangle.v3 - triangle.v2, q - triangle.v2), triangle.n) >= 0 and
                    np.dot(np.cross(triangle.v1 - triangle.v3, q - triangle.v3), triangle.n) >= 0):
                nearestTval, nearestObj = t, triangle

        return nearestTval, nearestObj

    def getNearestSphere(self, rayStart, rayDir):
        x0, y0, z0 = rayStart
        xd, yd, zd = rayDir
        nearestTval = np.inf
        nearestSphere = None
        # for each sphere in scene, plug ray equation into implicit sphere equation. Solve for t using quadratic (and use improvements - just solve for discriminant first, etc)
        for sphere in self.spheres:
            xc = sphere.xc
            yc = sphere.yc
            zc = sphere.zc
            r = sphere.radius
            b = 2 * (xd * x0 - xd * xc + yd * y0 - yd * yc + zd * z0 - zd * zc)
            c = x0 ** 2 - 2 * x0 * xc + xc ** 2 + y0 ** 2 - 2 * y0 * yc + yc ** 2 + z0 ** 2 - 2 * z0 * zc + zc ** 2 - r ** 2
            # keep track of which sphere has smallest positive t-value (the ray hits here first)
            discriminant = b ** 2 - 4 * c
            if discriminant < 0:
                continue
            t0 = (-b - np.sqrt(discriminant)) / 2
            if t0 > 0 and t0 < nearestTval:
                nearestTval, nearestSphere = t0, sphere
            else:
                t1 = (-b + np.sqrt(discriminant)) / 2
                if t1 > 0 and t1 < nearestTval:
                    nearestTval, nearestSphere = t1, sphere
        return nearestTval, nearestSphere

if __name__ == "__main__":
    sphere_purple = Sphere((0,0,0), 0.4, 0.7, 0.2, 0.1, (1,0,1), (1,1,1), 16)
    sphere_white = Sphere((0.45,0,-0.15), 0.15, 0.8, 0.1, 0.3, (1,1,1), (1,1,1), 4)
    sphere_red = Sphere((0,0,-0.1), 0.2, 0.6, 0.3, 0.1, (1,0,0), (1,1,1), 32)
    sphere_green = Sphere((-0.6,0,0), 0.3, 0.7, 0.2, 0.1, (0,1,0), (0.5,1,0.5), 64)
    sphere_blue = Sphere((0,-10000.5,0), 10000, 0.9, 0, 0.1, (0,0,1), (1,1,1), 16)
    sphere_orange = Sphere((0,0,0), 0.3, 0.9, 0.2, 0.1, (1,0.647,0), (1,1,1), 16, 0.5)
    sphere_yellow = Sphere((-0.25,-0.25,-0.25), 0.3, 0.6, 0.3, 0.1, (1,1,0), (1,1,1), 64, 0.3)
    sphere_cyan = Sphere((0.25,0.25,0.1), 0.3, 0.5, 0.4, 0.1, (0,1,1), (1,1,1), 128, 0.9)
    sphere_reflect = Sphere((0,0.3,-1), 0.25, 0, 0.1, 0.1, (0.75,0.75,0.75), (1,1,1), 10, 0.9)
    sphere_scene2_6_1 = Sphere((0.5,0,-0.15), 0.05, 0.8, 0.1, 0.3, (1,1,1), (1,1,1), 4, 0)
    sphere_scene2_6_2 = Sphere((0.3,0,-0.1), 0.08, 0.8, 0.8, 0.1, (1,0,0), (0.5,1,0.5), 32, 0)
    sphere_scene2_6_3 = Sphere((-0.6,0,0), 0.3, 0.7, 0.5, 0.1, (0,1,0), (0.5,1,0.5), 64, 0)
    sphere_scene2_6_4 = Sphere((0.1,-0.55,0.25), 0.3, 0, 0.1, 0.1, (0.75,0.75,0.75), (1,1,1), 10, 0.9)

    triangle_blue = Triangle((0,-0.7,-0.5), (1,0.4,-1), (0,-0.7,-1.5), 0.9, 1, 0.1, (0,0,1), (1,1,1), 4, 0)
    triangle_yellow = Triangle((0,-0.7,-0.5), (0,-0.7,-1.5), (-1,0.4,-1), 0.9, 1, 0.1, (1,1,0), (1,1,1), 4, 0)
    triangle_scene2_6_1 = Triangle((0.3,-0.3,-0.4), (0,0.3,-0.1), (-0.3,-0.3,0.2), 0.9, 0.9, 0.1, (0,0,1), (1,1,1), 32, 0)
    triangle_scene2_6_2 = Triangle((-0.2,0.1,0.1), (-0.2,-0.5,0.2), (-0.2,0.1,-0.3), 0.9, 0.5, 0.1, (1,1,0), (1,1,1), 4, 0)
    triangle_scene3_6_1 = Triangle((-1,-1,-1), (1,-1,0), (1,1,-1), 0, 0.1, 0.1, (0.75,0.75,0.75), (1,1,1), 4, 0.7)

    scene1_prog5 = RayTracer((0, 1, 0), (1, 1, 1), (0, 0, 0), (0.2, 0.2, 0.2)).withSpheres(sphere_purple)
    scene2_prog5 = RayTracer((1, 1, 1), (1, 1, 1), (0.1, 0.1, 0.1), (0.2, 0.2, 0.2)).withSpheres(sphere_red, sphere_white, sphere_green, sphere_blue)
    scene3_prog5 = RayTracer((-1, 1, 1), (1, 1, 1), (0.2, 0.2, 0.2), (0.5137, 0.9608, 0.1725)).withSpheres(sphere_orange, sphere_yellow, sphere_cyan)

    scene1_prog6 = RayTracer((0,1,0), (1,1,1), (0,0,0), (0.2,0.2,0.2)) \
        .withSpheres(sphere_reflect) \
        .withTriangles(triangle_yellow, triangle_blue)
    scene2_prog6 = RayTracer((1,0,0), (1,1,1), (0.1,0.1,0.1), (0.2,0.2,0.2)) \
        .withSpheres(sphere_scene2_6_1, sphere_scene2_6_2, sphere_scene2_6_3, sphere_scene2_6_4) \
        .withTriangles(triangle_scene2_6_1, triangle_scene2_6_2)
    scene3_prog6 = RayTracer((-1,-1,1), (1,1,1), (0.3,0.3,0.3), (0.9215, 0.9568, 1)) \
        .withSpheres(sphere_orange, sphere_yellow, sphere_cyan) \
        .withTriangles(triangle_scene3_6_1)

    # scene1_prog5.raytrace().saveImage("program_5-scene_1_jake.png")
    # scene2_prog5.raytrace().saveImage("program_5-scene_2_jake.png")
    # scene3_prog5.raytrace().saveImage("program_5-scene_3_jake.png")
    scene1_prog6.raytrace().saveImage("program_6-scene_1_jake.png")
    scene2_prog6.raytrace().saveImage("program_6-scene_2_jake.png")
    scene3_prog6.raytrace().saveImage("program_6-scene_3_jake.png")