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


def unit_magnitude_vectors_in_subspace(subspace: np.ndarray, a_search_space: np.ndarray = None, theta_search_space: np.ndarray = None):
  # subspace_dimension = subspace.shape[0]
  num_vectors = subspace.shape[1]
  print("Num vectors:", num_vectors)
  if num_vectors != 2:
    raise ValueError("Only works for 2 vectors right now")
  # search the projection of the subspace onto the unit circle
  if a_search_space is None:
    a_search_space = np.linspace(0, 1, 200)
  if theta_search_space is None:
    theta_search_space = np.linspace(0, 2 * np.pi, 200)
  results = np.zeros((len(a_search_space), len(theta_search_space)))
  for i, a in enumerate(a_search_space):
    for j, theta in enumerate(theta_search_space):
      r = np.sqrt(1 - a * a)
      c1 = a
      c2 = r * np.exp(1j * theta)
      vector = c1 * subspace[:, 0] + c2 * subspace[:, 1]
      # print("Vector norm (should be 1):", np.linalg.norm(vector))
      magnidutes = np.abs(vector)
      # projection = np.dot(np.ones(len(magnidutes)), magnidutes) * np.ones(len(magnidutes))
      # difference = magnidutes - projection
      # results[i, j] = np.linalg.norm(difference)
      results[i, j] = np.std(magnidutes)
  return 1 / results

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
