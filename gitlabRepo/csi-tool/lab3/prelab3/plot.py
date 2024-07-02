import csv
import numpy as np
import matplotlib.pyplot as plt


def peak_detection_with_thresh(t, sig):
    peaks = []
    for i in range(10, len(sig)-10):
        if sig[i] == max(sig[i-10: i+10]):
            peaks.append((t[i], sig[i]))
    return np.array(peaks)

with open('rssi.csv', 'r') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    next(data, None)  # skip the headers

    x = []
    y = []

    for row in data:
        x.append(float(row[0]))
        y.append(int(float(row[3])))

    min_x = min(x)
    x = [value - min_x for value in x]



    plt.plot(x, y)
    plt.title('RSSI vs Time')  # Title
    plt.xlabel('Time')  # X-axis label
    plt.ylabel('RSSI (dBm)')  # Y-axis label

    peaks = peak_detection_with_thresh(x, y)
    for peak in peaks:
        plt.scatter(peak[0], peak[1], color='red')
    plt.legend()
    plt.show()
