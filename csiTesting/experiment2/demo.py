import rosbag
import numpy as np
import tqdm
#these needs to be copied over into hte same folder as this file from this github repo: https://github.com/ucsdwcsng/wiros_processing_node/tree/main/src/csi_utils
import csi_utils.constants as constants
import csi_utils.transform_utils as transform_utils
import csi_utils.pipeline_utils as pipeline_utils
import csi_utils.comp_utils as comp_utils

#compensation data
comp = np.load("tx2-192.168.43.1-157.npy")

#opening all the files
zeroAOAdata = rosbag.Bag("AOA0.bag")
negAOAdata = rosbag.Bag("AOA-2.bag")
posAOAdata = rosbag.Bag("AOA+.bag")

#go through all the CSI packets of a bag
for (topic, msg, t) in zeroAOAdata:
    csi = pipeline_utils.extract_csi(msg, comp)
    # csi is an array with 3 dimensions: csi[subcarrier, rx, tx] gives the complex CSI data for a given subcarrier and rx-tx pair

