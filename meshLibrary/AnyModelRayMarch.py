import numpy as np
import trimesh
import os

# Each dictionary maps from material name to a list of tuples (angle, magnitude, phase)
class AttenuationCoefficients:
    def __init__(self, folderpath, nameToMaterialDict=None):
        self.nameToMaterialDict = nameToMaterialDict
        self.parallel_dict = {}
        self.perpendicular_dict = {}
        parallels = folderpath + "/parallel_polarity"
        perpendiculars = folderpath + "/perpendicular_polarity"
        for filename in os.listdir(parallels):
            self.parallel_dict[filename[:-4]] = np.genfromtxt(parallels + "/" + filename, delimiter=",")[1:]
        for filename in os.listdir(perpendiculars):
            self.perpendicular_dict[filename[:-4]] = np.genfromtxt(perpendiculars + "/" +filename, delimiter=",")[1:]


    
    def getAttenuationCoeff(self, name, angle, parallelPortion=0.5, degrees=True, decibels=False):
        """
        Returns a pair of [exitStrength/entryStrength, exitPhase - entryPhase] (in the specified units)
        or `None` if the material is not found
        """
        perpendicularPortion = 1 - parallelPortion
        if not degrees:
            angle = np.rad2deg(angle)
        if self.nameToMaterialDict is not None and name in self.nameToMaterialDict:
            name = self.nameToMaterialDict[name]
        if name not in self.parallel_dict:
            return (None, None)
        
        attenInfo = self.parallel_dict[name] * parallelPortion + self.perpendicular_dict[name] * perpendicularPortion
        #binary search for the nearest angle
        idx = np.searchsorted(attenInfo[:, 0], angle)
        # print("Index: ", idx, "Angle: ", angle, "Found: ", attenInfo[idx, 0])
        nums = attenInfo[idx]
        if not decibels:
            nums[1] = 10**(nums[1]/10)
        return nums[1:]
        


class ModelRayTrace:
    def getFileMaterial(self, filename):
        parts = filename.split(" ")
        final = parts[-1].split(".")[0]
        if final.endswith("(2)"):
          return final[:-3]
        return final
    
    def getMaterial(self, idx):
        return self.faceIdxToMaterial[idx]
        

    def __init__(self, filepath):
        self.combinedMesh = trimesh.Trimesh()
        self.materials = []
        self.filenames = []
        self.materialIdx = []
        l = 0
        for filename in os.listdir(filepath):
            self.materials.append(self.getFileMaterial(filename))
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
        self.rays = self.combinedMesh.ray

    def RayTrace(self, origins, directions, other=False):
        ''' Traces rays on the Connected House geometry
        origins: list or array of origins of rays (N x 3)
        directions: list or array of directions of rays (N x 3)        
        Returns: (didHit, locations, normals, face_indices)
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
        intersections = np.zeros_like(origins, dtype=float)
        normals = np.zeros_like(origins, dtype=float)
        normals[ray_idx] = normals_out
        a = np.sum(np.multiply(normals, directions), axis=1)
        normals[a>0] *= -1
        # return intersections_out, ray_idx, face_idx
        print("Intersections: ", intersections_out)
        print("Ray Index: ", ray_idx)
        print("Face Index: ", face_idx)
        face_indices = np.zeros(origins.shape[0], dtype=int) - 1
        didHit = np.zeros(origins.shape[0])
        if (intersections_out.shape[0] != 0):
          intersections[ray_idx] = intersections_out
          print("Setting stuff")
          didHit[ray_idx] = 1
          face_indices[ray_idx] = face_idx
        return (didHit, intersections, normals, face_indices)
        
def rayReflect(direction, normal):
    """Returns the reflected direction of a ray off a surface with the given outward normal"""
    # The normal is not necessarily normalized
    normal = normal / np.linalg.norm(normal)
    direction = direction / np.linalg.norm(direction)
    return direction - 2 * np.dot(direction, normal) * normal

def simulateRays(environment: ModelRayTrace, coeffs: AttenuationCoefficients, origins, directions):
    """
    Calculated the attenuation coefficients for a set of rays traveling in the environment.
    Returns a tuple (intersection points, new directions, attenuation coefficients)

    An entry in the coefficients is nan if the ray didn't intersect anything. Otherwise, a complex number.
    An entry in the coefficients is np.inf if the a material is not found (THIS SHOULD NEVER HAPPEN)
    Ex: simulateRays(env, coeffs, [[0, 0, 0], [2, 0, -1]], [[1, 0, 0], [0, 1, 1]]) could return
    ([None, nan+nanj], [normal1, None]) if the first ray hit nothing but the second did
    """
    didHit, intersections, normals, face_indices = environment.RayTrace(origins, directions)
    output = np.zeros(didHit.shape[0], dtype=complex)
    reflectDirections = np.zeros_like(np.array(directions), dtype=float)
    output[didHit<1] = None
    for idx in range(output.shape[0]):
      if didHit[idx] == 0:
          output[idx] = None
          continue
      fname = environment.filenames[environment.getMaterial(face_indices[idx])]
      material = environment.getFileMaterial(fname)
      direction = np.array(directions[idx])
      normal = -normals[idx]
      print(f"Direction is {direction}; normal is {normal}")
      cosTheta = np.dot(direction, normal) / (np.linalg.norm(direction) * np.linalg.norm(normal))
      # print(f"Scaled dot product is {cosTheta}")
      angle = np.arccos(cosTheta)
      reflectDirections[idx] = rayReflect(direction, normal)
      magnitude, phase = coeffs.getAttenuationCoeff(material, angle, 0.5, False, False)
      if magnitude is None:
          output[idx] = np.inf
      else:
        output[idx] = magnitude * np.exp(phase)
    # print("Intersections: ", intersections) 
    return intersections, reflectDirections, output


# Example usage
if __name__ == "__main__":
  # Folder where the model is stored
  mrt = ModelRayTrace("COLLADA dae geometry files/Connected Home Dae - Trimesh")
  # The dictionary translates the material in the filename to the material name in the coefficient folder
  attenuationCoeffs = AttenuationCoefficients("Reflection Coefficients", {"Terrain": "Wood", "Driveway" : "Concrete"})
  print("Plastic at 10.01 degrees:", attenuationCoeffs.getAttenuationCoeff("Plastic", 10.01, decibels=False, degrees=True, parallelPortion=0.8))
  origins = [[0, 0, 1], [0, 3, 0]]
  directions = [[0, 1, -2], [1, 0, 0]]
  intersections, newDirs, factors = simulateRays(mrt, attenuationCoeffs, origins, directions)
  for origin, factor, direction, intersectionPt, newDirection in zip(origins, factors, directions, intersections, newDirs):
      if np.isnan(factor):
          print(f"No intersection for ray from {origin} in {direction}")
      elif factor == np.inf:
          print(f"Could not find material for fay from {origin} in {direction}")
      else:
          print(f"Ray from {origin} in {direction} has factor: {factor}")
          print(f"Intersected at point {intersectionPt} and will reflect in direction {newDirection}")
