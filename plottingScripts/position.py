import numpy as np
import quaternion

def get_timestamps(file):
  """
  Read the timestamps from the file and return them as a list of tuples.
  Format: (timestamp, x, y, z, qx, qy, qz, qw)
  """
  timestamps = []
  with open(file) as f:
    for line in f.readlines():
      if line[0] == "#":
        continue
      parts = list(map(lambda x: float(x), line.split(" ")))
      # if (parts[0]):
      #   continue
      timestamps.append((parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6], parts[7]))
  return timestamps

def get_orientation_vector(timestamp):
  """
  Return the orientation vector for the given timestamp. Given as the x, y, and z components of the quaternion.
  """
  quat = quaternion.quaternion(timestamp[7], timestamp[4], timestamp[5], timestamp[6])
  vector = quat * quaternion.quaternion(0, 1, 0, 0) * quat.conjugate()
  return vector


import matplotlib.pyplot as plt

def plot_3d_data(plotData, i=None, ax=None, title=None, savename=None, color=None, bounds=None):
  fig = None
  newPlot = False
  if ax == None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    newPlot = True
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  if bounds != None:
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_zlim(bounds[4], bounds[5])
  if title == None:
    title = f"RSSI data"
  ax.set_title(title)
  if color is None:
    sc = ax.scatter(plotData[0], plotData[1], plotData[2], c=plotData[3], cmap="viridis", alpha=0.5)
  else:
    sc = ax.scatter(plotData[0], plotData[1], plotData[2], c=color, alpha=0.5)
  if i != None:
    ax.scatter(plotData[0][i], plotData[1][i], plotData[2][i], color="red", s=100)
  if fig != None and color is None:
    fig.colorbar(sc)

  if (savename == None and newPlot):
    plt.show()
  elif savename != None:
    plt.savefig(savename)


def plot_2d_data(plotData, i=None, ax=None, title=None, savename=None, color=None, bounds=None):
  fig = None
  newPlot = False
  if ax == None:
    fig = plt.figure()
    ax = fig.add_subplot()
    newPlot = True
  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  if bounds != None:
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
  # ax.set_xlim(-1.5, 1.5)
  # ax.set_ylim(-3, 2)
  if title == None:
    # title = f"Magnitude of CSI data from Tx2 (Mean over subcarriers)"
    title = f"RSSI data"
  ax.set_title(title)
  if color is None:
    sc = ax.scatter(plotData[0], plotData[1], c=plotData[3], cmap="viridis", alpha=1)
  else:
    sc = ax.scatter(plotData[0], plotData[1], c=color, alpha=1)
  if i != None:
    ax.scatter(plotData[0][i], plotData[1][i], color="red", s=100)
  if fig != None:
    fig.colorbar(sc)

  if (savename == None and newPlot):
    plt.show()
  elif savename != None:
    plt.savefig(savename)
  return sc

def plot_2d_data_aos(plotData, i=None, ax=None, title=None, savename=None, color=None, bounds=None):
  xs = [t[1] for t in plotData]
  ys = [t[2] for t in plotData]
  zs = [t[3] for t in plotData]
  ts = [t[0] for t in plotData]
  plot_2d_data([xs, ys, zs, ts], i, ax, title, savename, color, bounds)
