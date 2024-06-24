import csi_utils.synthetic_channel as synthetic_channel
import csi_utils.constants as constants
import numpy as np

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
    # self.svd_window[chanspec][c_roll * len(subcarriers):(c_roll + 1) * len(subcarriers), :] = H.reshape(len(subcarriers), -1)

    c_roll += 1
    if c_roll >= self.pkt_window:
      c_roll = 0
    self.svd_roll[chanspec] = c_roll

    # print("SVD window shape:", np.shape(self.svd_window[chanspec]))

    covariance = self.svd_window[chanspec].T @ self.svd_window[chanspec].conj()
    eigenvalues, eigenvecs = np.linalg.eigh(covariance)

    eigenvalues = eigenvalues[::-1]
    eigenvecs = eigenvecs[:,::-1]

    print("Eigenvalues:", eigenvalues)

    signal_space = eigenvecs[:,:numPaths]
    noise_space = eigenvecs[:,numPaths:]

    # print("Noise shape:", np.shape(noise_space))
    # print("Steering shape:", np.shape(self.steering_matrices[chanspec]))

    eHU  = self.steering_matrices[chanspec] @ noise_space.conj()

    if normSq:
      music_spectrum = (eHU[:, np.newaxis, :] @ eHU[:, :, np.newaxis].conj()).real
    else:
      music_spectrum = np.linalg.norm(eHU, axis=1)

    if calculateStrength:
      weighted_signal_space = signal_space * eigenvalues[:numPaths]
      eSU  = self.steering_matrices[chanspec] @ weighted_signal_space.conj()
      music_spectrum /= np.linalg.norm(eSU, axis=1)

    # print("Spectrum shape:", np.shape(music_spectrum))
    return 1 / music_spectrum.reshape(len(self.theta_space), len(self.phi_space))