import numpy as np
import trimesh
import os

class ModelRayTrace:
    def getMaterial(self, filename):
        parts = filename.split(" ")
        final = parts[-1].split(".")[0]
        if final.endswith("(2)"):
          return final[:-3]
        return final
    
    # def getMaterialIdx(self, idx):
        

    def __init__(self, filepath):
        self.combinedMesh = trimesh.Trimesh()
        self.materials = []
        self.filenames = []
        self.materialIdx = []
        l = 0
        for filename in os.listdir(filepath):
            self.materials.append(self.getMaterial(filename))
            self.filenames.append(filename)
            p = os.path.join(filepath, filename)
            # print(p)
            newgeometry = trimesh.load(p, force="mesh")
            # print(type(newgeometry))
            # print(f"Material of filename {filename} is {self.getMaterial(filename)}")
            self.combinedMesh = self.combinedMesh + newgeometry
            self.materialIdx.append(self.combinedMesh.faces.shape[0])
            l += 1
        self.faceIdxToMaterial = np.zeros(len(self.combinedMesh.faces), dtype=int) - 1
        print(self.faceIdxToMaterial[10])
        prev = 0
        for (i, idx) in enumerate(self.materialIdx):
            self.faceIdxToMaterial[prev:idx] = i
            prev = idx
        # for i in range(len(self.materials)):
        #     filename = self.filenames[i]
        #     p = os.path.join(filepath, filename)
        #     newgeometry = trimesh.load(p, force="mesh")
        #     print("Trying material", i)
        #     for j in range(len(self.combinedMesh.faces)):
        #         if (self.faceIdxToMaterial[j] == -1):
        #             for k in range(len(newgeometry.faces)):
        #                 if (np.array_equal(self.combinedMesh.face_normals[j], newgeometry.face_normals[k])):
        #                     self.faceIdxToMaterial[j] = i
        #                     # print("Setting index", j, "to", i)
        #                     break
        #             if (self.faceIdxToMaterial[j] == -1):
        #                 print("Failed to find face for index", j, "with material", i)


        self.rays = self.combinedMesh.ray

    def RayTrace(self, origins, directions, other=False):
        ''' Traces rays on the Connected House geometry
        origins: list or array of origins of rays (N x 3)
        directions: list or array of directions of rays (N x 3)        
        Returns: (didHit, locations, normals)
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
        if other:
            return intersections_out, ray_idx, face_idx
        print("Intersections: ", intersections_out)
        print("Ray Index: ", ray_idx)
        print("Face Index: ", face_idx)
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