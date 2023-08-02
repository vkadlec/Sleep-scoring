import numpy as np
from scipy.signal import lfilter
from montage import bipolar_montage


def channel_sort_list(channels):
    """
    Function to ad 0s to channels and sort them. Meant for FNUSA

    Parameters:
    -----------
    channels - list\n

    Returns:
    --------
    mod_df - modified and sorted dataframe\n
    """

    # Rename channels add 0s - for good ordering
    modified_channels = []
    for channel in channels:
        digits = [x for x in channel if x.isdigit()]
        if len(digits) == 1:
            dig_idx = channel.index(digits[0])
            mod_chan = channel[:dig_idx] + '0' + channel[dig_idx:]
            modified_channels.append(mod_chan)
        else:
            modified_channels.append(channel)

    modified_channels.sort()

    for ci, channel in enumerate(modified_channels):
        digits = [x for x in channel if x.isdigit()]
        if not len(digits):
            continue
        if digits[0] == '0':
            dig_idx = channel.index(digits[0])
            modified_channels[ci] = modified_channels[ci][0:dig_idx] + modified_channels[ci][dig_idx + 1:]

    return modified_channels


# Bipolar
def define_pairs(channels):
    # =============================================================================
    #     channels = ["B'1", "B'2", "B1", "B2", "Bo'1", "B'12"]
    #     channels = ch_names
    # =============================================================================

    channels = channel_sort_list(channels)

    channel_bases = []
    for channel in channels:
        channel_bases.append(''.join([x.strip() for x in channel if x.isalpha() or x == "'"]))

    channel_nums = []
    not_use_num = []
    for i, channel in enumerate(channels):
        num = ''.join([x.strip() for x in channel if x.isnumeric()])
        if num != '':
            channel_nums.append(int(''.join([x for x in channel if x.isnumeric()])))
        else:
            not_use_num.append(i)

    if len(not_use_num) > 0:
        ch_numbers = np.arange(len(channels))
        ch_numbers = [x for x in ch_numbers if x not in not_use_num]
        channel_bases = [channel_bases[k] for k in ch_numbers]
        channels = [channels[k] for k in ch_numbers]

    bipolar_pairs = []
    bipolar_names = []

    for i, ch in enumerate(channels[:-1]):
        channel_base = channel_bases[i]
        ch_num = channel_nums[i]
        if channel_bases[i + 1] == channel_base and channel_nums[i + 1] == ch_num + 1:
            bipolar_pairs.append([ch, channels[i + 1]])
            bipolar_names.append(ch + '_' + str(channel_nums[i + 1]))

    # We have bipolar pairs - we can create monatges
    return bipolar_pairs, bipolar_names


def bipolar_montage(data, channels):
    if len(data.T) != len(channels):
        assert ('Number of channel names must be equal to number of signals.')

    ch_dict = dict(zip(channels, np.arange(len(channels))))
    pairs, names = define_pairs(channels)

    bi_data = np.zeros([len(data), len(pairs)], dtype='float64')

    for i, pair in enumerate(pairs):
        bi_data[:, i] = data[:, ch_dict[pair[0]]] - data[:, ch_dict[pair[1]]]

    return bi_data, names


def read_signal(ms, time_start, window, overlap, channel_names):    # práce s definovaným start_time
    overlap_size = overlap * 1e6
    window = window * 1e6

    time_start = time_start - overlap_size
    time_stop = time_start + window + 2 * overlap_size

    data = np.c_[ms.read_ts_channels_uutc(channel_names, [int(time_start), int(time_stop)])]

    # create bipolar
    bipolar_data, bipolar_channels = bipolar_montage(data.T, channel_names)

    return bipolar_data, bipolar_channels


def change_sampling_rate(x, fs):
    od5, od2, ou2 = 54, 26, 34

    hd5 = np.array([-0.000413312132792,   0.000384910656353,   0.000895384486596,   0.001426584098180,   0.001572675788393,
                     0.000956099017099,  -0.000559378457343,  -0.002678217568221,  -0.004629975982837,  -0.005358589238386,
                    -0.003933117464092,  -0.000059710059922,   0.005521319363883,   0.010983495478404,   0.013840996082966,
                     0.011817315106321,   0.003905283425021,  -0.008768844009700,  -0.022682212400564,  -0.032498023687148,
                    -0.032456772047175,  -0.018225658085891,   0.011386634156651,   0.053456542440034,   0.101168250947271,
                     0.145263694388270,   0.176384224234024,   0.187607302744229,   0.176384224234024,   0.145263694388270,
                     0.101168250947271,   0.053456542440034,   0.011386634156651,  -0.018225658085891,  -0.032456772047175,
                    -0.032498023687148,  -0.022682212400564,  -0.008768844009700,   0.003905283425021,   0.011817315106321,
                     0.013840996082966,   0.010983495478404,   0.005521319363883,  -0.000059710059922,  -0.003933117464092,
                    -0.005358589238386,  -0.004629975982837,  -0.002678217568221,  -0.000559378457343,   0.000956099017099,
                     0.001572675788393,   0.001426584098180,   0.000895384486596,   0.000384910656353,  -0.000413312132792])

    if fs == 2000:
        z = np.zeros((od5 // 2, x.shape[1]))
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

        x = 4 * np.reshape(np.stack([x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]), (4 * x.shape[0], x.shape[1]), order='F')
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

        x = 4 * np.reshape(np.stack([x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]), (4 * x.shape[0], x.shape[1]), order='F')
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

    elif fs == 5000:
        z = np.zeros((od5 // 2, x.shape[1]))
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

        z = np.zeros((od5 // 2, x.shape[1]))
        x = 2 * np.reshape(np.stack([x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]), (4 * x.shape[0], x.shape[1]), order='F')
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

        x = 4 * np.reshape(np.stack([x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]), (4 * x.shape[0], x.shape[1]), order='F')
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

        x = 4 * np.reshape(np.stack([x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]), (4 * x.shape[0], x.shape[1]), order='F')
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

    else:
        raise ValueError

    return x


def compute_features(x):
    J = 8
    nd = 5
    l = np.array([.22641898, .85394354, 1.02432694, .19576696, -.34265671, -.04560113,
                  .10970265, -.0088268, -.01779187, .00471742793])

    h = np.flipud(l) * (-1) ** np.arange(2 * nd)

    x = x[:2 ** J * int(np.floor(len(x) * 2 ** -J))]

    laa = np.zeros((round(x.shape[0] / 2)))
    C = np.zeros((3, J))
    b = np.ones((J, 1))
    mwc = np.zeros((3 * J, 1))

    for jj in range(J):
        # compute the wavelet leaders
        d = lfilter(h, 1, x, axis=0)
        lea = d[::2]
        x = lfilter(l, 1, x, axis=0)
        x = x[::2]

        mm = np.abs(lea[nd - 1 + int(np.ceil(256 / 2 ** (jj + 1))):-1 - int(max(np.ceil(256 / 2 ** (jj + 1)) - nd - 1, 0))])

        mwc[jj] = np.log(np.mean(mm))
        mwc[jj + J] = np.sum(mm ** 2)
        mwc[jj + 2 * J] = np.log(np.std(mm))

        lea = np.abs(np.hstack([0, lea.flatten(), 0]))
        lea = np.maximum(np.maximum(lea[:-2], lea[1:-1]), lea[2:])
        lea = np.maximum(lea, laa)
        laa = np.maximum(lea[::2], lea[1::2])

        # get cumulants of ln leaders
        lea = lea[nd - 1 + int(np.ceil(256 / 2 ** (jj + 1))):-1 - int(max(np.ceil(256 / 2 ** (jj + 1)) - nd - 1, 0))] # transients discarding

        le = np.log(lea)
        u1 = np.mean(le)
        u2 = np.mean(le ** 2)
        u3 = np.mean(le ** 3)

        C[:, jj] = np.array([u1, u2 - u1 ** 2, u3 - 3 * u1 * u2 + 2 * u1 ** 3])
        nj = lea.shape[0]
        b[jj] = nj

    sc = np.arange(1, 6, 1)
    C = C[:, sc]
    b = b[sc]
    v0 = np.sum(b)
    v1 = np.dot(sc + 1, b)
    v2 = np.dot((sc + 1) ** 2, b)
    w = b.T * ((v0 * (sc + 1) - v1) / (v0 * v2 - v1 ** 2))

    f = np.sum(np.log2(np.exp(1)) * w * C, axis=1)
    f = np.concatenate([f, mwc.flatten()])
    f = np.delete(f, np.array((4, 4 + J, 4 + 2 * J), dtype=int), axis=0)  # exclude the 64-128 Hz scale

    return f
