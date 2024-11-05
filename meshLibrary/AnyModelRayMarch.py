import numpy as np
import trimesh
import os

class ModelRayTrace:

    def __init__(self, filepath):
        self.combinedMesh = trimesh.Trimesh()
        for filename in os.listdir(filepath):
            p = os.path.join(filepath, filename)
            # print(p)
            newgeometry = trimesh.load(p, force="mesh")
            print(type(newgeometry))
            self.combinedMesh = self.combinedMesh + newgeometry
        self.rays = self.combinedMesh.ray

    def RayTrace(self, origins, directions):
        ''' Traces rays on the Connected House geometry
        origins: list or array of origins of rays (N x 3)
        directions: list or array of directions of rays (N x 3)        Returns:
        (didHit, locations, normals)
        didHit: array of 1 if the ray hit something, 0 otherwise (1 x N)
        locations: locations of ray intersection with mesh (N x 3)
        normals: normals of the face the ray intersected with (N x 3)'''
        origins = np.asarray(origins)
        directions = np.asarray(directions)
        if (origins.shape != directions.shape):
            raise ValueError(f"mismatch in shape of origins and directions. Origins: {origins.shape}  Directions: {directions.shape}")
        if (origins.shape[1] != 3):
            raise ValueError("origins has the wrong shape: " + origins.shape)
        if (directions.shape[1] != 3):
            raise ValueError("directions has the wrong shape: " + directions.shape)
        intersections_out, ray_idx, face_idx = self.rays.intersects_location(ray_origins=origins, ray_directions=directions, multiple_hits=False)
        normals_out = self.combinedMesh.face_normals[face_idx]
        intersections = np.zeros_like(origins)
        normals = np.zeros_like(origins)
        normals[ray_idx] = normals_out
        a = np.sum(np.multiply(normals, directions), axis=1)
        normals[a>0] *= -1
        intersections[ray_idx] = intersections_out
        didHit = np.zeros(origins.shape[0])
        didHit[ray_idx] = 1
        return (didHit, intersections, normals)
        
        

# chrt = ConnectedHouseRayTrace()
# # origins = np.array([[0, 0, 0, 0]])
# print(chrt.RayTrace([[1000, 0, 0], [1, 0, 0]], [[1, 0, 0], [0, 1, 0]]))

# mrt = ModelRayTrace("Connected Home Api\All Home Office")

# print(mrt.RayTrace([[0, 0, 0]], [[1, 0, 0]]))