import numpy as np
import csi_utils.pipeline_utils as pipeline_utils

def interpolate_timestamps(start, end, t):
  lowerTime = start[0]
  upperTime = end[0]
  frac = (t - lowerTime) / (upperTime - lowerTime)
  toReturn = [t]
  for a, b in zip(start[1:], end[1:]):
    toReturn.append(a + (b - a) * frac)
  return toReturn

def interpolate_data(bag, timestamps, compensation=None):
  # list of tuples (time, x, y, z, qx, qy, qz, qw, csi, rssi)
  data = []
  upperIndex = 1
  for (topic, msg, t), k in zip(bag.read_messages('/csi'), bag):
    t = t.to_sec()
    if (t < timestamps[0][0]):
      continue
    if (t > timestamps[-1][0]):
      break
    if (t >= timestamps[upperIndex][0]):
      while (t >= timestamps[upperIndex][0] and upperIndex < len(timestamps) - 1):
        upperIndex += 1
    lowerIndex = upperIndex - 1
    positionData = interpolate_timestamps(timestamps[lowerIndex], timestamps[upperIndex], t)

    csi = pipeline_utils.extract_csi(msg)
    if compensation is not None:
      csi *= compensation
    positionData.append(csi)

    positionData.append(k[1].rssi)
    data.append(tuple(positionData))

  return data

def consolidate(data, n=10):
  newData = []
  for i in range(0, len(data), n):
    sum = 0
    for j in range(0, n):
      if (i + j >= len(data)):
        n = j
        break
      sum += data[i+j]
    newData.append(sum / n)
  return newData

def consolidateSOA(data, n=10):
  newData = []
  for d in data:
    newData.append(consolidate(d, n))
  return newData


def get_median_magnitude(csi, rx):
  sum = 0
  allMagnitudes = [csi[subcarrier, rx, 2] for subcarrier in range(0, 50)]
  allMagnitudes.sort()
  return 20*np.log10(np.abs(allMagnitudes[25]))

def get_mean_magnitude(csi, rx):
  sum = 0
  allMagnitudes = [csi[subcarrier, rx, 2] for subcarrier in range(0, 50)]
  for data in allMagnitudes:
    sum += 20*np.log10(np.abs(data))
    # sum += np.abs(data)
  return sum / 50

def get_plot_data(combinedData):
  plotData = [[], [], [], []]
  for time, x, y, z, qx, qy, qz, qw, csi, rssi in combinedData:
    # magnitude = (get_mean_magnitude(csi, 0) + get_mean_magnitude(csi, 1) + get_mean_magnitude(csi, 2) + get_mean_magnitude(csi, 3))/4
    magnitude = rssi
    plotData[0].append(x)
    plotData[1].append(y)
    plotData[2].append(z)
    plotData[3].append(magnitude)
  return plotData
