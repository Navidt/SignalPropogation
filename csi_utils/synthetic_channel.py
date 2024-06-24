import numpy as np
from csi_utils import constants

def angle_steering_vector(angle: float, rx_positions: list[tuple[float, float]], frequencies: np.ndarray):
  steering_vector = np.zeros((len(rx_positions), len(frequencies)), dtype=complex)
  for i, (x, y) in enumerate(rx_positions):
    distance = np.cos(angle) * x + np.sin(angle) * y
    steering_vector[i] = np.exp(-1j * 2 * np.pi * distance * frequencies / constants.c)
  # print("steering_vector", steering_vector.shape)
  return steering_vector.transpose(1, 0)

# IGNORE THIS FUNCTION
def test_steering_matrix(aoa, aod, rx_positions: list[tuple[float, float]], tx_positions: list[tuple[float, float]], frequencies: np.ndarray):
  matrix = np.zeros((len(frequencies), len(rx_positions), len(tx_positions)), dtype=complex)
  for i, rx in enumerate(rx_positions):
    for j, tx in enumerate(tx_positions):
      matrix[:, i, j] = np.exp(-1j * (i * aoa - j * aod))
  return matrix


def aoa_aod_steering_matrix(aoa: float, aod: float, rx_positions: list[tuple[float, float]], tx_positions: list[tuple[float, float]], frequencies: np.ndarray):
    steering_vector_rx = angle_steering_vector(aoa, rx_positions, frequencies)
    steering_vector_tx = angle_steering_vector(aod, tx_positions, frequencies)

    # add axes to make the outer product work
    steering_vector_rx = steering_vector_rx[:, :, np.newaxis]
    steering_vector_tx = steering_vector_tx[:, np.newaxis, :]

    return steering_vector_rx @ steering_vector_tx

def create_channel(num_rx: int, num_tx: int, paths: list[tuple[float, float, float]], frequencies: list[float], rx_positions=None, tx_positions=None, tof=None, noiseStrength=0, degrees=False):
  """
  Create a synthetic channel matrix based on the given AoAs and AoDs
  :param num_rx: Number of receive antennas
  :param num_tx: Number of transmit antennas
  :param paths: List of tuples of (AoA, Aod, strength) in radians
  :param frequencies: List of frequencies for each subcarrier
  :param rx_positions: List of receiver antenna positions; Default: lambda/2 spacing
  :param tx_positions: List of transmit antenna positions; Default: lambda/2 spacing
  :param tof: Add time of flight to the channel matrix; Default: None
  :param noiseStrength: Noise strength; Default: 0
  :param degrees: If True, convert angles to radians
  :return: Channel matrix
  """
  channelMatrix = np.zeros((len(frequencies), num_rx, num_tx), dtype=complex)
  frequencies = np.array(frequencies)
  wavelength = constants.c / frequencies[len(frequencies) // 2]
  if rx_positions is None:
    rx_positions = np.array([(i * wavelength / 2, 0) for i in range(num_rx)])
  if tx_positions is None:
    tx_positions = np.array([(i * wavelength / 2, 0) for i in range(num_tx)])

  for aoa, aod, strength in paths:
    if degrees:
      aoa = np.deg2rad(aoa)
      aod = np.deg2rad(aod)
    steering_matrix = aoa_aod_steering_matrix(aoa, aod, rx_positions, tx_positions, frequencies)
    # steering_matrix = test_steering_matrix(aoa, aod, rx_positions, tx_positions, frequencies)
    if tof is not None:
      channelMatrix += strength * steering_matrix * np.exp(1j * 2 * np.pi * tof * frequencies)
    else:
      channelMatrix += strength * steering_matrix * (2 + 5 * np.random.random((len(frequencies), 1, 1)))

  if noiseStrength > 0:
    channelMatrix += np.random.normal(0, noiseStrength, channelMatrix.shape) + 1j * np.random.normal(0, noiseStrength, channelMatrix.shape)
  return channelMatrix