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

  

def unit_magnitude_vectors_in_subspace(subspace: np.ndarray, a_search_space: np.ndarray = None, theta_search_space: np.ndarray = None):
  # subspace_dimension = subspace.shape[0]
  num_vectors = subspace.shape[1]
  print("Num vectors:", num_vectors)
  if num_vectors != 2:
    raise ValueError("Only works for 2 vectors right now")
  # search the projection of the subspace onto the unit circle
  if a_search_space is None:
    a_search_space = np.linspace(0, np.pi / 2, 200)
  if theta_search_space is None:
    theta_search_space = np.linspace(0, 2 * np.pi, 200)
  results = np.zeros((len(a_search_space), len(theta_search_space)))
  vectors = np.zeros((len(a_search_space), len(theta_search_space), subspace.shape[0]), dtype=np.complex128)
  for i, a in enumerate(a_search_space):
    for j, theta in enumerate(theta_search_space):
      # r = np.sqrt(1 - a * a)
      c1 = np.cos(a)
      c2 = np.sin(a) * np.exp(1j * theta)
      vector = c1 * subspace[:, 0] + c2 * subspace[:, 1]
      # print("Vector norm (should be 1):", np.linalg.norm(vector))
      magnidutes = np.abs(vector)
      avg = np.average(magnidutes)
      results[i, j] = 1 - subspace.shape[0] * avg * avg
      vectors[i, j] = vector
      # results[i, j] = np.std(magnidutes)
  return 1 / results, vectors

class PartitionIterator:
  def __init__(self, n, k):
    self.partition = [k] + [0] * (n - 1)
    self.k = k
    self.transferIndex = 0

  def __iter__(self):
    return self
  
  def __next__(self):
    if self.partition[-1] == self.k:
      raise StopIteration
    self.partition[self.transferIndex] -= 1
    endValue = self.partition[-1]
    self.partition[-1] = 0
    self.partition[self.transferIndex + 1] = endValue + 1
    self.transferIndex += 1
    if self.transferIndex == len(self.partition) - 1:
      for i in range(len(self.partition) - 2, -1, -1):
        if self.partition[i] != 0:
          self.transferIndex = i
          break
    return self.partition


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

def next_partition(partition: list[int], n: int):
  """
  Generate the next partition of n
  :param partition: Current partition
  :param n: Number to partition
  :return: Next partition
  """
  if partition[-1] == n:
    return None
  for i in range(len(partition) - 2, -1, -1):
    if partition[i] == 0:
      continue
    partition[i] -= 1
    endValue = partition[-1]
    partition[-1] = 0
    partition[i + 1] = endValue + 1
    break
  return partition

def generate_partitions_recursive(n, k, partition=None):
  partition = [k] + [0] * (n - 1)
  partitions = [partition]
  while True:
    partition = next_partition(partition, k)
    if partition is None:
      break
    partitions.append(partition.copy())
  return partitions

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


# THIS DOESN'T WORK; THERE IS NO REASON THAT THE COMPENSATION SUBSPACES HAVE TO BE THE SAME -- THEY MUST ONLY
# AGREE ON THE SINGLE VECTOR
def calculate_compensation_correlation(covariance_matrix: np.ndarray, steering_vectors: np.ndarray, numPaths = 2):
  """
  Calculate the compensation correlation matrix
  :param covariance_matrix: Covariance matrix of the channel
  :param steering_vectors: Steering vectors of the channel
  :return: Compensation correlation matrix
  """
  eigenvalues, eigenvecs = np.linalg.eigh(covariance_matrix)

  eigenvalues = eigenvalues[::-1]
  eigenvecs = eigenvecs[:,::-1]

  signal_space = eigenvecs[:, :numPaths]
  product_signal_space = signal_space / signal_space[:, 0][:, np.newaxis]
  # product_signal_basis = np.linalg.orth(product_signal_space)

  signal_dimension = len(steering_vectors[0])

  # steering_subspaces = np.zeros((len(steering_vectors), signal_dimension, numPaths), dtype=np.complex128)

  # a subspace for the possible compensation vectors of each steering vector. The first numPaths columns are the basis of the subspace
  # the rest of the columns are the basis of the orthogonal complement of the subspace. All orthonormal.
  steering_subspaces = np.zeros((len(steering_vectors), signal_dimension, signal_dimension), dtype=np.complex128)


  # matrix_basis = np.identity(numPaths * signal_dimension).reshape(-1, numPaths, signal_dimension)

  sum_squares = signal_dimension * signal_dimension - 2 * numPaths * (signal_dimension - numPaths)

  matrix_basis = np.zeros((sum_squares, signal_dimension, signal_dimension))
  k = 0
  for i in range(numPaths):
    for j in range(numPaths):
      matrix_basis[k][i, j] = 1
      k += 1
  
  for i in range(numPaths, signal_dimension):
    for j in range(numPaths, signal_dimension):
      matrix_basis[k][i, j] = 1
      k += 1


  # stores an orthonormal basis for the (flattened) matrices that preserve the subspace of a steering vector
  transformation_matrix_subspaces = np.zeros((len(steering_vectors), signal_dimension * signal_dimension, sum_squares), dtype=np.complex128)

  for i in range(len(steering_vectors)):
    # just the vectors spanning the subspace of the compensation vectors
    print("signal space shape:", signal_space.shape)
    print("steering vector shape:", steering_vectors[i].shape)
    steering_subspace_sparse = (signal_space.T / steering_vectors[i]).T

    eigenvalues, eigenvecs = np.linalg.eigh(steering_subspace_sparse @ steering_subspace_sparse.conj().T)
    eigenvecs = eigenvecs[:, ::-1]
    # print("Eigenvalues:", eigenvalues)

    # steering_subspaces[i] = orth((signal_space.T / steering_vectors[i]).T)
    steering_subspaces[i] = eigenvecs


    # matrix_subspace = (steering_subspaces[i] @ matrix_basis @ steering_subspaces[i].conj().T).reshape(numPaths * numPaths, -1).T

    # print("Matrix basis shape:", matrix_basis.shape)
    # print("Steering subspace shape:", steering_subspaces[i].shape)

    matrix_subspace = (steering_subspaces[i] @ matrix_basis @ steering_subspaces[i].conj().T).reshape(-1, signal_dimension * signal_dimension).T

    # print("Matrix subspace shape:", matrix_subspace.shape)

    transformation_matrix_subspaces[i] = orth(matrix_subspace)

  output_distances = np.zeros((len(steering_vectors), len(steering_vectors)))

  for i in range(len(steering_vectors)):
    for j in range(len(steering_vectors)):
      # if abs(i - j) == 0:
      #   output_distances[i, j] = 0
      #   continue

      intersection = subspace_intersection(steering_subspaces[i][:, :numPaths], steering_subspaces[j][:, :numPaths])
      # print("Intersection:", intersection)
      for i in range(intersection.shape[1]):
        vector = intersection[:, i]
        # project the vector onto both subspaces
        # print("Vector:", vector)
        print("Magnitude:", np.linalg.norm(vector))
        projection_i = steering_subspaces[i] @ steering_subspaces[i].conj().T @ vector
        projection_j = steering_subspaces[j] @ steering_subspaces[j].conj().T @ vector
        difference_i = vector - projection_i
        difference_j = vector - projection_j
        print("Difference i:", np.linalg.norm(difference_i))
        print("Difference j:", np.linalg.norm(difference_j))
      print("Dot product:", intersection.conj().T @ intersection)
      # this doesn't work because they only have to agree on one compensation value, not the whole subspace
      # angles = principal_angle(steering_subspaces[i][:, :numPaths], steering_subspaces[j][:, :numPaths])
      # output_distances[i, j] = 1 / np.linalg.norm(angles)
      continue
      difference = np.diag(steering_vectors[i] / steering_vectors[j]).flatten()
      print("Difference:", difference)

      # print("Difference:", difference.shape)
      # print("Transformation:", transformation_matrix_subspaces[j].shape)

      # project the difference and turn it back into our coordinates

      # tm = transformation_matrix_subspaces[j]
      # middleMatrix = np.linalg.inv(tm.conj().T @ tm)
      # print("Middle matrix:", middleMatrix)
      # difference_projection = tm @ middleMatrix @ tm.conj().T @ difference
      difference_projection = transformation_matrix_subspaces[j] @ transformation_matrix_subspaces[j].conj().T @ difference

      if i == j:
        print("Difference:", difference)
        print("Difference projection:", difference_projection)
      # get the magnitude of the difference between the two matrices
      output_distances[i, j] = np.linalg.norm(difference - difference_projection)
      print("Distance:", output_distances[i, j])

  return output_distances
