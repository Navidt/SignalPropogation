import csi_utils.synthetic_channel as synthetic_channel
import csi_utils.constants as constants
import numpy as np
from scipy.linalg import orth
from scipy.linalg import subspace_angles


def principal_angle(q1, q2):
  # S = q1.T.dot(q2)[None, :]
  # [u, s, v] = np.linalg.svd(S, full_matrices=False)

  # compute principle angles
  # theta = np.arccos(s)
  # print("Angles:", theta)
  print("HELLO")
  angles = subspace_angles(q1, q2)
  print("Angles: ", angles)
  return angles

def subspace_intersection(q1, q2):
  """
  Calculate the intersection of two subspaces
  :param q1: Subspace 1
  :param q2: Subspace 2
  :return: Intersection of the two subspaces
  """
  return orth(q1 @ q1.conj().T @ q2)



class IndexIterator():
  def __init__(self, bounds):
    # bounds is a list of the number of elements in each dimension
    self.bounds = bounds
    self.indices = np.zeros(len(bounds), dtype=int)
  
  def __iter__(self):
    # reset indices
    self.indices = np.zeros(len(self.bounds), dtype=int)
    self.indices[-1] -= 1
    return self
  
  def __next__(self):
    for i in range(len(self.indices) - 1, -1, -1):
      self.indices[i] += 1
      if self.indices[i] < self.bounds[i]:
        return self.indices
      self.indices[i] = 0
    raise StopIteration


def point_on_hypersphere(angles: np.ndarray):
  """
  Generate a point on the hypersphere
  :param angles: Angles to generate the point
  :return: Point on the hypersphere
  """
  sineCoefficient = 1
  point = np.ones(len(angles) + 1)
  for i, angle in enumerate(angles):
    point[i] *= sineCoefficient * np.cos(angle)
    sineCoefficient *= np.sin(angle)
  point[-1] = sineCoefficient
  return point

def unit_magnitude_vectors_in_subspace_new(subspace: np.ndarray, granularity: int = 100):
  """
  Find unit magnitude vectors in a subspace
  :param subspace: Subspace to search
  :param granularity: Number of points to search per dimension
  """

  vector_dimension = subspace.shape[0]
  subspace_dimension = subspace.shape[1]

  sphere_angles_search = [np.linspace(0, np.pi, granularity) for _ in range(subspace_dimension - 1)]
  sphere_angles_search[-1] = np.linspace(0, 2 * np.pi, granularity)
  sphere_angles_search[0] = np.linspace(0, np.pi / 2, granularity)

  complex_angles_search = [np.linspace(0, 2 * np.pi, granularity) for _ in range(subspace_dimension - 1)]

  result_length = 2 * subspace_dimension - 2
  results = np.zeros(tuple([granularity] * result_length))
  vectors = np.zeros(tuple(([granularity] * result_length + [vector_dimension])), dtype=np.complex128)
  sphere_index_iterator = IndexIterator([len(sphere_angles_search[i]) for i in range(subspace_dimension - 1)])
  complex_index_iterator = IndexIterator([len(complex_angles_search[i]) for i in range(subspace_dimension - 1)])
  for sphere_indices in sphere_index_iterator:
    # print("Sphere indices:", sphere_indices)
    sphere_angles = [sphere_angles_search[i][sphere_indices[i]] for i in range(subspace_dimension - 1)]
    magnitudes = point_on_hypersphere(sphere_angles)
    for complex_indices in complex_index_iterator:
      complex_angles = np.array([0] + [complex_angles_search[i][complex_indices[i]] for i in range(subspace_dimension - 1)])
      # print("Complex angles:", complex_angles)
      basis = magnitudes * np.exp(1j * complex_angles)
      # print("Basis magnitude:", np.abs(basis))
      vector = subspace @ basis
      # print("Vector:", vector)

      magnidutes = np.abs(vector)
      avg = np.average(magnidutes)
      # print("result shape:", results.shape)
      results[tuple(sphere_indices)][tuple(complex_indices)] = 1 - subspace.shape[0] * avg * avg
      # print("Sohere indices:", tuple(sphere_indices))
      # print("Vectors shape:", vectors[tuple(sphere_indices)].shape)
      vectors[tuple(sphere_indices)][tuple(complex_indices)] = vector

  return 1 / results, vectors


class AdjacentIterator:
  def __init__(self, dimension, radius):
    self.indices = np.zeros(dimension, dtype=int) - radius
    self.radius = radius
    self.dimension = dimension

  def __iter__(self):
    # reset values
    self.indices = np.zeros(self.dimension, dtype=int) - self.radius
    return self
  
  def __next__(self):
    for i in range(len(self.indices) - 1, -1, -1):
      self.indices[i] += 1
      if self.indices[i] <= self.radius:
        return self.indices
      self.indices[i] = -self.radius
    raise StopIteration

def local_maxima(matrix: np.ndarray, threshold: float = 0.0, radius: int = 1, search_indices: np.ndarray = None):
  """
  Detect local maxima in a matrix. It's a good idea to use a threshold to avoid detecting noise as local maxima.
  To efficiently narrow down on local maxima, it's a good idea to repeatedly apply this function to its output with increasing radius.

  :param matrix: Matrix to search
  :param threshold: Threshold for local maxima
  :param radius: Radius to search for local maxima
  :param search_space: Array of indices to search for local maxima. If none, search the whole matrix

  :return: List of indices of local maxima
  """
  # A local maximum is defined as a point in the matrix that is greater than all of its neighbors
  # A neighbor is defined as a point that is adjacent to the point in question

  # for example, in 2D, the set would be {(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)}


  dimension = matrix.ndim
  indexIterator = AdjacentIterator(dimension, radius)

  results = []
  if search_indices is None:
    it = np.nditer(matrix, flags=['multi_index'])
    for x in it:
      if x < threshold:
        continue
      localMax = True
      for indices in indexIterator:
        if all([0 <= i < matrix.shape[j] for j, i in enumerate(np.add(it.multi_index, indices))]):
          if matrix[tuple(np.add(it.multi_index, indices))] > x:
            localMax = False
            break
          if matrix[tuple(np.add(it.multi_index, indices))] == x and tuple(np.add(it.multi_index, indices)) < it.multi_index:
            localMax = False
            break
      if localMax:  
        results.append(it.multi_index)
  else:
    for index in search_indices:
      x = matrix[tuple(index)]
      if x < threshold:
        continue
      localMax = True
      for indices in indexIterator:
        if all([0 <= i < matrix.shape[j] for j, i in enumerate(np.add(index, indices))]):
          if matrix[tuple(np.add(index, indices))] > x:
            localMax = False
            break
      if localMax:
        results.append(index)
  return results

def steering_vector_multi_correlation(transformedVectors: list[np.ndarray], steeringVectors: list[np.ndarray]):
  correlations = []
  for i in range(1, len(transformedVectors)):
    correlations.append(steering_vector_correlation([transformedVectors[0], transformedVectors[i]], steeringVectors))
  return correlations

def steering_vector_correlation(transformedVectors: list[np.ndarray], steeringVectors: list[np.ndarray]):
  #For now, only works with 2 transformed vectors
  vectorRatio = (transformedVectors[0] / transformedVectors[0][0]) / (transformedVectors[1] / transformedVectors[1][0])
  profile = np.zeros((len(steeringVectors), len(steeringVectors)))
  for i in range(len(steeringVectors)):
    for j in range(len(steeringVectors)):
      steeringVectorRatio = (steeringVectors[i] / steeringVectors[i][0]) / (steeringVectors[j] / steeringVectors[j][0])
      # profile[i, j] = np.abs((steeringVectorRatio.conj()[np.newaxis, :] @ vectorRatio)**2)
      profile[i, j] = ((steeringVectorRatio.conj()[np.newaxis, :] @ vectorRatio)).real**3

      # relation = steeringVectorRatio / vectorRatio
      # deviation = np.abs(relation.imag)
      # profile[i, j] = 1 / (np.linalg.norm(deviation))
  return profile


def calculate_compensation_correlation(covarianceMatrix: np.ndarray, steeringVectors: np.ndarray, numPaths = 2):
  """
  Calculate the compensation correlation matrix
  :param covariance_matrix: Covariance matrix of the channel
  :param steering_vectors: Steering vectors of the channel
  :return: Compensation correlation matrix
  """
  eigenvalues, eigenvecs = np.linalg.eigh(covarianceMatrix)

  eigenvalues = eigenvalues[::-1]
  eigenvecs = eigenvecs[:,::-1]

  signalSpace = eigenvecs[:, :numPaths]


  signal_dimension = len(steeringVectors[0])

  compensation_correlation = np.zeros((len(steeringVectors), len(steeringVectors)))
  unit_correlation = np.zeros((len(steeringVectors), len(steeringVectors)))
  distance_correlation = np.zeros((len(steeringVectors), len(steeringVectors)))

  matrixTemplate = np.zeros((signal_dimension, 2 * numPaths - 1), dtype=complex)

  matrixTemplate[:, :numPaths - 1] = -signalSpace[:, 1:numPaths]
  matrixTemplate[:, numPaths - 1:] = signalSpace

  # print(matrixTemplate)

  for i in range(len(steeringVectors)):
    for j in range(len(steeringVectors)):
      if abs(i -j) < 5:
        compensation_correlation[i, j] = 0.1
        continue
      steeringVectorRatio = (steeringVectors[i] / steeringVectors[i][0]) / (steeringVectors[j] / steeringVectors[j][0])
      
      A = matrixTemplate.copy()
      A[:, numPaths - 1:] *= steeringVectorRatio[:, np.newaxis]
      solution = np.linalg.lstsq(A, signalSpace[:, 0], rcond=None)

      vector = A @ solution[0]

      compensation_correlation[i, j] = (np.linalg.norm(vector - signalSpace[:, 0]))
      distance_correlation[i, j] = 1 / compensation_correlation[i, j]

      candidateVec1 = signalSpace[:, 1:] @ solution[0][:numPaths - 1] + signalSpace[:, 0]
      candidateVec2 = signalSpace @ solution[0][numPaths - 1:]

      candidateVec1 /= candidateVec1[0]
      candidateVec2 /= candidateVec2[0]
      standardDeviation = np.std(np.abs(candidateVec1)) + np.std(np.abs(candidateVec2))
      unit_correlation[i, j] = 1 / standardDeviation
      compensation_correlation[i, j] += standardDeviation
      compensation_correlation[i, j] = 1 / compensation_correlation[i, j]


  # print("Distance correlation:", distance_correlation.shape)
  return compensation_correlation, unit_correlation, distance_correlation
