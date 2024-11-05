import dataTools, position
import numpy as np

def plot_csi_data_3d(bag, timestamps, ax=None, title=None, savename=None):
  combinedData = dataTools.interpolate_data(bag, timestamps)
  plotData = dataTools.get_plot_data(combinedData)
  plotData = np.array(dataTools.consolidateSOA(plotData, 100))
  position.plot_3d_data(plotData, ax, title, savename)

def plot_csi_data_2d(bag, timestamps, ax=None, title=None, savename=None):
  combinedData = dataTools.interpolate_data(bag, timestamps)
  plotData = dataTools.get_plot_data(combinedData)
  plotData = np.array(dataTools.consolidateSOA(plotData, 3))
  position.plot_2d_data(plotData, ax, title, savename)