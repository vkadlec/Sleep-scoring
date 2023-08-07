import numpy as np
from scipy.signal import lfilter
from montage import bipolar_montage


def read_signal(ms, time_start, window, overlap, channels, pairs):
    """
    Function to read signal from MEF session and compute bipolar data

    :param ms: MEF session
    :param time_start: Signal segment start time
    :param window: Thirty-second window
    :param overlap: Overlapping part size
    :param channels: Unipolar channels
    :param pairs: Bipolar pairs
    :return: Computed bipolar data
    """

    overlap_size = overlap * 1e6
    window = window * 1e6

    time_start = time_start - overlap_size
    time_stop = time_start + window + 2 * overlap_size

    uni_data = np.c_[ms.read_ts_channels_uutc(channels, [int(time_start), int(time_stop)])]

    # create bipolar
    bi_data = bipolar_montage(uni_data.T, channels, pairs)

    return bi_data


def change_sampling_rate(x, fs):
    """
    Function to resample signal to 256 samples per second

    :param x: Signal segment to resample
    :param fs: Sampling rate
    :return: Resampled signal segment
    """

    od5, od2, ou2 = 54, 26, 34

    hd5 = np.array([-0.000413312132792,   0.000384910656353,   0.000895384486596,   0.001426584098180,
                     0.001572675788393,   0.000956099017099,  -0.000559378457343,  -0.002678217568221,
                    -0.004629975982837,  -0.005358589238386,  -0.003933117464092,  -0.000059710059922,
                     0.005521319363883,   0.010983495478404,   0.013840996082966,   0.011817315106321,
                     0.003905283425021,  -0.008768844009700,  -0.022682212400564,  -0.032498023687148,
                    -0.032456772047175,  -0.018225658085891,   0.011386634156651,   0.053456542440034,
                     0.101168250947271,   0.145263694388270,   0.176384224234024,   0.187607302744229,
                     0.176384224234024,   0.145263694388270,   0.101168250947271,   0.053456542440034,
                     0.011386634156651,  -0.018225658085891,  -0.032456772047175,  -0.032498023687148,
                    -0.022682212400564,  -0.008768844009700,   0.003905283425021,   0.011817315106321,
                     0.013840996082966,   0.010983495478404,   0.005521319363883,  -0.000059710059922,
                    -0.003933117464092,  -0.005358589238386,  -0.004629975982837,  -0.002678217568221,
                    -0.000559378457343,   0.000956099017099,   0.001572675788393,   0.001426584098180,
                     0.000895384486596,   0.000384910656353,  -0.000413312132792])

    if fs == 2000:
        z = np.zeros((od5 // 2, x.shape[1]))
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

        x = 4 * np.reshape(np.stack([x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]),
                           (4 * x.shape[0], x.shape[1]), order='F')
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

        x = 4 * np.reshape(np.stack([x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]),
                           (4 * x.shape[0], x.shape[1]), order='F')
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

    elif fs == 5000:
        z = np.zeros((od5 // 2, x.shape[1]))
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

        z = np.zeros((od5 // 2, x.shape[1]))
        x = 2 * np.reshape(np.stack([x, np.zeros_like(x)]), (2 * x.shape[0], x.shape[1]), order='F')
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

        x = 4 * np.reshape(np.stack([x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]),
                           (4 * x.shape[0], x.shape[1]), order='F')
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

        x = 4 * np.reshape(np.stack([x, np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]),
                           (4 * x.shape[0], x.shape[1]), order='F')
        x = lfilter(hd5, 1, np.concatenate([x, z]), axis=0)[od5 // 2:]
        x = x[::5, :]

    else:
        raise ValueError

    return x


def compute_features(x):
    """
    Function to compute features for sleep stages classification

    :param x: Thirty-second signal segment
    :return: Computed features for signal segment
    """

    j = 8
    nd = 5
    lx = np.array([.22641898, .85394354, 1.02432694, .19576696, -.34265671, -.04560113,
                   .10970265, -.0088268, -.01779187, .00471742793])

    h = np.flipud(lx) * (-1) ** np.arange(2 * nd)

    x = x[:2 ** j * int(np.floor(len(x) * 2 ** -j))]

    laa = np.zeros((round(x.shape[0] / 2)))
    c = np.zeros((3, j))
    b = np.ones((j, 1))
    mwc = np.zeros((3 * j, 1))

    for jj in range(j):
        # compute the wavelet leaders
        d = lfilter(h, 1, x, axis=0)
        lea = d[::2]
        x = lfilter(lx, 1, x, axis=0)
        x = x[::2]

        mm = np.abs(
            lea[nd - 1 + int(np.ceil(256 / 2 ** (jj + 1))):-1 - int(max(np.ceil(256 / 2 ** (jj + 1)) - nd - 1, 0))])

        mwc[jj] = np.log(np.mean(mm))
        mwc[jj + j] = np.sum(mm ** 2)
        mwc[jj + 2 * j] = np.log(np.std(mm))

        lea = np.abs(np.hstack([0, lea.flatten(), 0]))
        lea = np.maximum(np.maximum(lea[:-2], lea[1:-1]), lea[2:])
        lea = np.maximum(lea, laa)
        laa = np.maximum(lea[::2], lea[1::2])

        # get cumulants of ln leaders
        # transients discarding
        lea = lea[nd - 1 + int(np.ceil(256 / 2 ** (jj + 1))):-1 - int(max(np.ceil(256 / 2 ** (jj + 1)) - nd - 1, 0))]

        le = np.log(lea)
        u1 = np.mean(le)
        u2 = np.mean(le ** 2)
        u3 = np.mean(le ** 3)

        c[:, jj] = np.array([u1, u2 - u1 ** 2, u3 - 3 * u1 * u2 + 2 * u1 ** 3])
        nj = lea.shape[0]
        b[jj] = nj

    sc = np.arange(1, 6, 1)
    c = c[:, sc]
    b = b[sc]
    v0 = np.sum(b)
    v1 = np.dot(sc + 1, b)
    v2 = np.dot((sc + 1) ** 2, b)
    w = b.T * ((v0 * (sc + 1) - v1) / (v0 * v2 - v1 ** 2))

    f = np.sum(np.log2(np.exp(1)) * w * c, axis=1)
    f = np.concatenate([f, mwc.flatten()])
    f = np.delete(f, np.array((4, 4 + j, 4 + 2 * j), dtype=int), axis=0)  # exclude the 64-128 Hz scale

    return f
