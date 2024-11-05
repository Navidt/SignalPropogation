import numpy as np
import matplotlib.pyplot as plt
import quaternion
import plottingScripts.position as position
import csi_utils.music_algorithms as music_algorithms

def calculate_profiles(data: list, searchSpace: np.ndarray, rx_positions, tx_positions, smoothingSize=1, batchSize=1):
  """
  data: list of tuples (time, x, y, z, qx, qy, qz, qw, csi, rssi)

  returns: list of tuples (time, x, y, z, qx, qy, qz, qw, profile, rssi)
  """

  sensor = music_algorithms.full_music_aoa_aod_sensor(rx_positions, tx_positions, searchSpace, searchSpace, smoothingSize)

  profiles = []
  for d in data[::batchSize]:
    csi = d[8]
    profile = sensor.run(csi, (157, 20e6), numPaths=6, subcarriers=range(0, 50), calculateStrength=False)
    profiles.append(d[:8] + (profile,d[-1]))

  return profiles



def plot_angle_profile(angles: np.ndarray, profile: np.ndarray, center=(0, 0), ax=None, degrees=True, maxSize=15, zeroVector=None, color="blue"):
  newPlot = False
  if ax == None:
    fig, ax = plt.subplots()
    newPlot = True
  if degrees:
    angles = np.deg2rad(angles)
  profile *= maxSize / np.max(profile)

  # zero vector is the vector pointing in the direction of 0 degrees on the profile
  angleOffset = 0
  if zeroVector is not None:
    angleOffset = np.arctan2(zeroVector[1], zeroVector[0])

  xs = []
  ys = []
  for i in range(len(angles)):
    angle = angles[i] + angleOffset
    radius = profile[i]
    x = np.cos(angle) * radius + center[0]
    y = np.sin(angle) * radius + center[1]
    xs.append(x)
    ys.append(y)
  ax.plot(xs, ys, color=color)
  # ax.set_xlim(-15, 15)
  # ax.set_ylim(-15, 15)
  if newPlot:
    plt.show()

def plot_angle_vector(angle: float, center=(0, 0), ax=None, degrees=True, length=0.5, zeroVector=None, color="blue"):
  newPlot = False
  if ax == None:
    fig, ax = plt.subplots()
    newPlot = True
  if degrees:
    angle = np.deg2rad(angle)
  if zeroVector is not None:
    angle += np.arctan2(zeroVector[1], zeroVector[0])
  dx = np.cos(angle) * length
  dy = np.sin(angle) * length
  # print(dx, dy)
  # print(center[0], center[1])
  ax.quiver(center[0], center[1], dx, dy, color=color)



  if newPlot:
    plt.show()

def calculate_angle_to_position(devicePosition, deviceOrientation, targetPosition):
  """
  devicePosition: tuple (x, y)
  deviceOrientation: quaternion (x, y)
  targetPosition: tuple (x, y)

  returns: float angle in radians
  """

  targetVector = (targetPosition[0] - devicePosition[0], targetPosition[1] - devicePosition[1])
  targetVectorComplex = targetVector[0] + targetVector[1] * 1j
  orientationComplex = deviceOrientation[0] + deviceOrientation[1] * 1j
  angle = np.angle(targetVectorComplex / orientationComplex)
  if (angle > np.pi / 2):
    angle = np.pi - angle
  elif (angle < -np.pi / 2):
    angle = -np.pi + angle
  return angle


def get_aoa_aod_intersections(angles, profile, txPos, txOr, xRange, yRange):
  # angles is a list of angles in radians
  # profile is a list of tuples (time, x, y, z, qx, qy, qz, qw, profile, rssi)
  # txPos is the position of the transmitter
  # txOr is the orientation of the transmitter, given as a vector
  # xRange and yRange are the ranges of the x and y coordinates to plot
  num_steps = 50
  results = np.zeros((num_steps, num_steps))
  x_coords = np.linspace(xRange[0], xRange[1], num_steps)
  y_coords = np.linspace(yRange[0], yRange[1], num_steps)

  rxPos = (profile[1], profile[2])
  rxOr = position.get_orientation_vector(profile)

  for (i, x) in enumerate(x_coords):
    for (j, y) in enumerate(y_coords):
      # print(f"rxPos: {rxPos}, rxOr: {rxOr}, txPos: {txPos}, txOr: {txOr}")
      rxAngle = calculate_angle_to_position(rxPos, (rxOr.x, rxOr.y), (x, y))
      txAngle = calculate_angle_to_position(txPos, txOr, (x, y))
      rxIdx = np.argmin(np.abs(angles - rxAngle))
      txIdx = np.argmin(np.abs(angles - txAngle))
      results[i, j] = profile[8][rxIdx, txIdx]
      # print(f"x: {x}, y: {y}, rxAngle: {np.rad2deg(rxAngle)}, txAngle: {np.rad2deg(txAngle)}")
  return results

  # for x in np.arange(xRange[0], xRange[1], 0.1):
  #   for y in np.arange(yRange[0], yRange[1], 0.1):
  #     for p in profile:
  #       if (p[1] == x and p[2] == y):
  #         plot_angle_profile(angles, p[8], center=(x, y), degrees=False, maxSize=15)
  #         break
