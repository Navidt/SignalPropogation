#  Python version of the MATLAB tool in linux-80211n-csitool-supplementary
#  https://github.com/dhalperi/linux-80211n-csitool-supplementary/tree/master/matlab
#
#  <ytchi2@illinois.edu>
#

import numpy as np
from math import ceil


def read_bf_file(filename):
    """
    Reads in a file of beamforming feedback logs
    """
    # Open file
    with open(filename, 'rb') as f:
        inBytes = f.read()

    # Initialize variables
    ret = [{}] * ceil(len(inBytes) / 95)  # Holds the return values - 1x1 CSI is 95 bytes big, so this should be upper bound
    cur = 0  # Current offset into file
    count = 0  # Number of records output
    broken_perm = 0  # Flag marking whether we've encountered a broken CSI yet
    triangle = [1, 3, 6]  # What perm should sum to for 1,2,3 antennas

    # Process all entries in file
    # Need 3 bytes -- 2 byte size field and 1 byte code
    while cur < (len(inBytes) - 3):
        # Read size and code
        field_len = int.from_bytes(inBytes[cur:cur + 2], byteorder='big', signed=False)
        code = inBytes[cur + 2]
        cur = cur + 3

        # If unhandled code, skip (seek over) the record and continue
        if code == 187:  # get beamforming or phy data
            pass
        else:  # skip all other info
            cur = cur + field_len - 1
            continue

        if code == 187:  # hex2dec('bb')) Beamforming matrix -- output a record
            count = count + 1

            read_bfee_timestamp_low = int.from_bytes(inBytes[cur:cur + 4], byteorder='little', signed=False)
            read_bfee_bfee_count = int.from_bytes(inBytes[cur + 4:cur + 6], byteorder='little', signed=False)
            read_bfee_Nrx = inBytes[cur + 8]
            read_bfee_Ntx = inBytes[cur + 9]
            read_bfee_rssi_a = inBytes[cur + 10]
            read_bfee_rssi_b = inBytes[cur + 11]
            read_bfee_rssi_c = inBytes[cur + 12]
            read_bfee_noise = inBytes[cur + 13] - 256
            read_bfee_agc = inBytes[cur + 14]
            read_bfee_antenna_sel = inBytes[cur + 15]
            read_bfee_len = int.from_bytes(inBytes[cur + 16:cur + 18], byteorder='little', signed=False)
            read_bfee_fake_rate_n_flags = int.from_bytes(inBytes[cur + 18:cur + 20], byteorder='little', signed=False)
            read_bfee_calc_len = (30 * (read_bfee_Nrx * read_bfee_Ntx * 8 * 2 + 3) + 7) // 8
            index = 0
            payload = inBytes[cur + 20:cur + 20 + read_bfee_len]
            read_bfee_csi = np.zeros(shape=(read_bfee_Ntx * read_bfee_Nrx, 30), dtype=complex)
            read_bfee_perm = np.zeros(shape=3, dtype=int)

            # Check that length matches what it should
            if read_bfee_len != read_bfee_calc_len:
                print("MIMOToolbox:read_bfee_new:size", "Wrong beamforming matrix size.")

            # Compute CSI from all this crap :)
            for i in range(30):
                index += 3
                remainder = index % 8

                for j in range(read_bfee_Nrx * read_bfee_Ntx):
                    tmp = bytes([(payload[index // 8] >> remainder) |
                                 (payload[index // 8 + 1] << (8 - remainder)) & 0xFF])
                    R = float(int.from_bytes(tmp, byteorder='little', signed=True))

                    tmp = bytes([(payload[index // 8 + 1] >> remainder) |
                                 (payload[index // 8 + 2] << (8 - remainder)) & 0xFF])
                    I = float(int.from_bytes(tmp, byteorder='little', signed=True))

                    read_bfee_csi[j, i] = complex(R, I)
                    index += 16

            read_bfee_csi = read_bfee_csi.reshape((read_bfee_Ntx, read_bfee_Nrx, 30), order='F')

            # Compute the permutation array
            read_bfee_perm[0] = ((read_bfee_antenna_sel) & 0x3) + 1
            read_bfee_perm[1] = ((read_bfee_antenna_sel >> 2) & 0x3) + 1
            read_bfee_perm[2] = ((read_bfee_antenna_sel >> 4) & 0x3) + 1

            cur = cur + field_len - 1

            # ret{count} = read_bfee(bytes);
            ret[count] = {'timestamp_low': read_bfee_timestamp_low,
                          'bfee_count': read_bfee_bfee_count,
                          'Nrx': read_bfee_Nrx,
                          'Ntx': read_bfee_Ntx,
                          'rssi_a': read_bfee_rssi_a,
                          'rssi_b': read_bfee_rssi_b,
                          'rssi_c': read_bfee_rssi_c,
                          'noise': read_bfee_noise,
                          'agc': read_bfee_agc,
                          'perm': read_bfee_perm,
                          'rate': read_bfee_fake_rate_n_flags,
                          'csi': read_bfee_csi}

            perm = ret[count]['perm']
            Nrx = ret[count]['Nrx']
            if Nrx == 1:  # No permuting needed for only 1 antenna
                continue

            if sum(perm) != triangle[Nrx - 1]:  # matrix does not contain default values
                if broken_perm == 0:
                    broken_perm = 1
                    print(f'WARN ONCE: Found CSI ({filename}) with Nrx={Nrx} and invalid perm=[{perm}]\n')
            else:
                ret[count]['csi'][:, perm[:Nrx] - 1, :] = ret[count]['csi'][:, :Nrx, :].copy()

    ret = ret[1:count]
    return ret


def db(x, Signal_Type='voltage'):
    """
    Convert energy or power measurements to decibels
    """
    if Signal_Type == 'power' or Signal_Type == 'pow':
        if np.any(np.less(x, 0)):
            raise ValueError("If you specify SignalType as 'power', then all elements of x must be nonnegative.")
        return 10 * np.log10(x)
    elif Signal_Type == 'voltage':
        return 10 * np.log10(np.square(x))
    else:
        raise ValueError("Signal type must be either 'voltage' or 'power'.")


def dbinv(x):
    """
    Convert from decibels.
    """
    return np.power(10, x / 10)


def get_total_rss(csi_st):
    """
    Calculates the Received Signal Strength (RSS) in dBm from
    a CSI struct.
    """
    # Careful here: rssis could be zero
    rssi_mag = 0
    if csi_st['rssi_a'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_a'])
    if csi_st['rssi_b'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_b'])
    if csi_st['rssi_c'] != 0:
        rssi_mag = rssi_mag + dbinv(csi_st['rssi_c'])
    return db(rssi_mag, 'pow') - 44 - csi_st['agc']


def get_scale_csi(csi_st):
    """
    Converts a CSI struct to a channel matrix H.
    """
    # Pull out CSI
    csi = csi_st['csi']

    # Calculate the scale factor between normalized CSI and RSSI (mW)
    csi_sq = np.multiply(csi, np.conj(csi)).real
    csi_pwr = np.sum(csi_sq)
    rssi_pwr = dbinv(get_total_rss(csi_st))

    # Scale CSI -> Signal power : rssi_pwr / (mean of csi_pwr)
    scale = rssi_pwr / (csi_pwr / 30)

    # Thermal noise might be undefined if the trace was
    # captured in monitor mode.
    # ... If so, set it to -92
    if csi_st['noise'] == -127:
        noise_db = -92
    else:
        noise_db = csi_st['noise']
    thermal_noise_pwr = dbinv(noise_db)

    # Quantization error: the coefficients in the matrices are
    # 8-bit signed numbers, max 127/-128 to min 0/1. Given that Intel
    # only uses a 6-bit ADC, I expect every entry to be off by about
    # +/- 1 (total across real & complex parts) per entry.
    #
    # The total power is then 1^2 = 1 per entry, and there are
    # Nrx*Ntx entries per carrier. We only want one carrier's worth of
    # error, since we only computed one carrier's worth of signal above.
    quant_error_pwr = scale * (csi_st['Nrx'] * csi_st['Ntx'])

    # Total noise and error power
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    # Ret now has units of sqrt(SNR) just like H in textbooks
    ret = csi * np.sqrt(scale / total_noise_pwr)
    if csi_st['Ntx'] == 2:
        ret = ret * np.sqrt(2)
    elif csi_st['Ntx'] == 3:
        # Note: this should be sqrt(3)~ 4.77 dB. But, 4.5 dB is how
        # Intel (and some other chip makers) approximate a factor of 3
        #
        # You may need to change this if your card does the right thing.
        ret = ret * np.sqrt(dbinv(4.5))
    return ret
