import csi_utils.synthetic_channel as synthetic_channel
import csi_utils.constants as constants
import numpy as np


def generic_music(covarianceMatrix, steeringVectors, numPaths=2, calculateStrength=False):
  """
  Perform the MUSIC algorithm on the given covariance matrix and steering vectors
  :param covarianceMatrix: Covariance matrix
  :param steeringVectors: Steering vectors
  :return: Spectrum
  """
  # Compute the noise subspace
  covarianceMatrix = np.array(covarianceMatrix)
  steeringVectors = np.array(steeringVectors)
  eigenvalues, eigenvecs = np.linalg.eigh(covarianceMatrix)

  eigenvalues = eigenvalues[::-1]
  eigenvecs = eigenvecs[:,::-1]

  # print("Eigenvalues:", eigenvalues)

  signalSpace = eigenvecs[:,:numPaths]
  noiseSpace = eigenvecs[:,numPaths:]
  # Compute the MUSIC spectrum

  eHU  = steeringVectors @ noiseSpace.conj()

  
  music_spectrum = np.linalg.norm(eHU, axis=1)

  if calculateStrength:
    weighted_signal_space = signalSpace * eigenvalues[:numPaths]
    eSU  = steeringVectors @ weighted_signal_space.conj()
    music_spectrum /= np.linalg.norm(eSU, axis=1)

  # print("Spectrum shape:", np.shape(music_spectrum))
  return 1 / music_spectrum

class MusicResults:
  def __init__(self, matrix, thetaRange, phiRange):
    self.matrix = matrix
    self.thetaRange = thetaRange
    self.phiRange = phiRange
    self.maxima = {} # dictionary from indices to the finer results at that index
    self.depth = 0
  def add_maximum(self, i, j, finerResults):
    self.maxima[(i, j)] = finerResults
    self.depth = max(self.depth, finerResults.depth + 1)

  def levelOneIdx(self, i, j):
    coarseI = i // np.shape(self.matrix)[0]
    coarseJ = j // np.shape(self.matrix)[1]
    if (coarseI, coarseJ) in self.maxima:
      return self.maxima[(coarseI, coarseJ)].matrix[i % np.shape(self.matrix)[0], j % np.shape(self.matrix)[1]]
    else:
      return self.matrix[coarseI, coarseJ]

  def renderIntoImg(self):
    image = np.zeros((np.shape(self.matrix)[0]**2, np.shape(self.matrix)[1]**2))
    print("Image shape:", np.shape(image))
    for i in range(np.shape(image)[0]):
      for j in range(np.shape(image)[1]):
        image[i, j] = self.levelOneIdx(i, j)
    return image




def generic_music_fast(covarianceMatrix, steeringVectorFunction, numPaths=2, granularity=10, thetaRange=[0, 180], phiRange=[0, 180], calculateStrength=False, depth=2, showChosen=False):
  """
  steeringVectorFunction should take in two angles and return a steering vector
  thetaRange and phiRange are the ranges of angles to consider
  granularity is the number of points to consider in each range
  """
  steeringVectors = np.zeros((granularity * granularity, np.shape(covarianceMatrix)[0]), dtype=np.complex128)
  for i in range(granularity):
    for j in range(granularity):
      theta = thetaRange[0] + (thetaRange[1] - thetaRange[0]) * i / granularity
      phi = phiRange[0] + (phiRange[1] - phiRange[0]) * j / granularity
      steeringVector = steeringVectorFunction(theta, phi)
      steeringVectors[i * granularity + j] = steeringVector
  matrix = generic_music(covarianceMatrix, steeringVectors, numPaths, calculateStrength).reshape(granularity, granularity)
  # Find the local maxima in the results and try again
  # get indices of top 3 maxima in results
  top = 5
  maxima = np.argpartition(matrix.flatten(), -top)[-top:]
  # maxima = [np.argmax(matrix)]
  
  results = MusicResults(matrix, thetaRange, phiRange)
  # return results
  if (depth == 0):
    return results
  for maximum in maxima:
    maxi = maximum // granularity
    maxj = maximum % granularity
    for offsets in [(0, 0), (1, 0), (0, 1), (0, -1), (-1, 0)]:
      i = maxi + offsets[0]
      j = maxj + offsets[1]
      # get the indices of the maximum in the 2D array
      
      # get the angles corresponding to the maximum
      startTheta = thetaRange[0] + (thetaRange[1] - thetaRange[0]) * i / granularity
      startPhi = phiRange[0] + (phiRange[1] - phiRange[0]) * j / granularity
      endTheta = thetaRange[0] + (thetaRange[1] - thetaRange[0]) * (i + 1) / granularity
      endPhi = phiRange[0] + (phiRange[1] - phiRange[0]) * (j + 1) / granularity
      newResults = generic_music_fast(covarianceMatrix, steeringVectorFunction, numPaths, granularity, [startTheta, endTheta], [startPhi, endPhi], calculateStrength, depth - 1)
      if showChosen:
        newResults.matrix += 0.2
      results.add_maximum(i, j, newResults)
    

    # get the steering vector at the maximum
    # steeringVector = steeringVectorFunction(theta, phi)
    # steeringVectors[maximum] = steeringVector
  print("Maxima:", maxima)
  return results

# def maximum_sampler()

class full_music_aoa_aod_sensor():
  def __init__(self, rx_pos, tx_pos, theta_space, phi_space, pkt_window=40):
    self.rx_pos = rx_pos
    self.tx_pos = tx_pos
    self.theta_space = theta_space
    self.phi_space = phi_space
    self.pkt_window = pkt_window
    self.steering_matrices = {}
    self.svd_window = {}
    self.svd_roll = {}
    self.chanspec_seen = set()
  
  def run(self, H, chanspec, subcarriers, txs=[0, 1, 2, 3], rxs=[0, 1, 2, 3], numPaths=2, normSq=False, calculateStrength=False):
    # only generate the steering matrices once since they'll be used for every packet on this channel
    if chanspec not in self.chanspec_seen:
      rx_positions = []
      for rx in rxs:
        rx_positions.append(self.rx_pos[rx])
      rx_positions = np.array(rx_positions)
      
      tx_positions = []
      for tx in txs:
        tx_positions.append(self.tx_pos[tx])
      tx_positions = np.array(tx_positions)

      self.steering_matrices[chanspec] = np.zeros((len(self.theta_space), len(self.phi_space), len(rx_positions), len(tx_positions)), dtype=np.complex128)
      freqs = constants.get_channel_frequencies(chanspec[0],chanspec[1])
      for i, theta in enumerate(self.theta_space):
        for j, phi in enumerate(self.phi_space):
          self.steering_matrices[chanspec][i, j] = synthetic_channel.aoa_aod_steering_matrix(theta, phi, rx_positions, tx_positions, np.array([np.mean(freqs)]))[0]
      self.steering_matrices[chanspec] = self.steering_matrices[chanspec].reshape(-1, len(rx_positions) * len(tx_positions))

      self.chanspec_seen.add(chanspec)
      
      self.svd_window[chanspec] = np.zeros((self.pkt_window * len(subcarriers),rx_positions.shape[0]*tx_positions.shape[0]),dtype=np.complex128)
      self.svd_roll[chanspec] = 0
    c_roll = self.svd_roll[chanspec]
    H = H[subcarriers, :, :]
    H = H[:,rxs,:]
    self.svd_window[chanspec][c_roll * len(subcarriers):(c_roll + 1) * len(subcarriers), :] = H[:, :, txs].reshape(len(subcarriers), -1)

    c_roll += 1
    if c_roll >= self.pkt_window:
      c_roll = 0
    self.svd_roll[chanspec] = c_roll

    covariance = self.svd_window[chanspec].T @ self.svd_window[chanspec].conj()
    return generic_music(covariance, self.steering_matrices[chanspec], numPaths, calculateStrength=calculateStrength).reshape(len(self.theta_space), len(self.phi_space))