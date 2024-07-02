from parse_csi import read_bf_file
from parse_csi import db
from parse_csi import get_scale_csi

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
csi_trace = read_bf_file('test1.dat')
csi_entry = csi_trace[0]
csi = get_scale_csi(csi_entry)

plt.figure(figsize=(15, 10))

for tx in range(csi.shape[0]):
    # Create a subplot for each transmitter
    plt.subplot(3, 1, tx + 1)
    # Extract data for each transmitter across all subcarriers
    plt.plot(db(np.abs(np.squeeze(csi[tx, :, :]).T)))
    plt.legend(['RX Antenna A', 'RX Antenna B', 'RX Antenna C'], loc='lower right')
    plt.xlabel('Subcarrier index')
    plt.ylabel(f'SNR [dB] - {tx + 1}')

plt.subplots_adjust(hspace=0.5)  # Adjust horizontal space between plots
plt.savefig('timestamp.jpg')
plt.show()

for subcarrier_index in range(30):
    plt.figure(figsize=(15, 10))

    for tx in range(3):
        csi_values = []
        # Iterate over all entries (timestamps) in the CSI trace
        for csi_entry in csi_trace:
            csi = get_scale_csi(csi_entry)
            csi_values.append(np.abs(csi[tx, :, subcarrier_index]))

        csi_values = np.array(csi_values)

        # Create a subplot for each transmitter
        plt.subplot(3, 1, tx + 1)
        # Plot the CSI values over time
        plt.plot(db(csi_values))
        plt.legend(['RX Antenna A', 'RX Antenna B', 'RX Antenna C'], loc='lower right')
        plt.xlabel('Time index')
        plt.ylabel(f'SNR [dB] -{tx + 1} - {subcarrier_index + 1}')

    plt.subplots_adjust(hspace=0.5)  # Adjust horizontal space between plots
    plt.savefig(f'subcarrrier{subcarrier_index + 1}.jpg')
    plt.show()
