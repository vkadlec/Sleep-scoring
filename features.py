import numpy as np
import pickle as pkl
from datetime import datetime
from pymef.mef_session import MefSession

from utils import *
from montage import define_pairs

# mef_file = 'C:/Users/vojta/PycharmProjects/ICRC/HFO/data/mni_00583_01.mefd'
# password = 'mnimef'

# mef_file = 'D:/sleep_scoring/mefs/seeg_060_sleep_sliced.mefd'
# password = 'bemena'

mef_file = 'D:/sleep/seeg_066_sleep_sliced.mefd'
password = 'bemena'

window = 30
overlap = 1.25

# loading mef session
ms = MefSession(mef_file, password)

hdr = ms.read_ts_channel_basic_info()
fs = int(hdr[0]['fsamp'][0])
nsamp = hdr[0]['nsamp'][0]
start_time = hdr[0]['start_time'][0] * 1e-6

uni_names = [x['name'] for x in hdr]

date = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
sec = float(date.split(':')[-1])

if sec < 27.75:
    sta = fs * (29 - sec)
    sec = '29'
elif sec < 57.75:
    sta = fs * (59 - sec)
    sec = '59'
else:
    sta = fs * (60 - sec + 29)
    sec = '29'

sta = round(sta)
start_time = (start_time + sta / fs) * 1e6

# compute features init
epochs = int((nsamp - sta) / fs / 30)
nfeat = 24
ne = 0
nf = 1
nnf = 1
sleep_stage = np.zeros((1500, 5))
night = np.ones((1500, 1), dtype=bool)
sleep_stage[ne:epochs, 0:2] = np.array((np.ones(epochs) * nf, start_time + np.arange(0, epochs) / 2880), dtype=int).T

bi_pairs, bi_names = define_pairs(uni_names)
num_channels = len(bi_names)
feature = np.zeros((nfeat, len(bi_names), 1500))

print('Computing features...')

for ii in range(0, epochs):
    bi_data = read_signal(ms, start_time + ii * window * 1e6, window, overlap, uni_names, bi_pairs)
    bi_data = change_sampling_rate(bi_data, fs)

    bi_data = bi_data[63:, :]
    bi_data = bi_data[:8192, :]
    bi_data = bi_data - np.tile(np.mean(bi_data), (8192, 1))

    for i in range(bi_data.shape[1]):
        feature[:, i, ne] = compute_features(bi_data[:, i])

    night[ne] = nf <= nnf
    sleep_stage[ne, 4] = sta + 30 * fs * ii

    ne += 1

    print('Epoch: ', ii + 1)

print('Computing features done!')

ne += 1
num_feats = 24

feature = feature[:, :, 0:ne]
night = night[0:ne]
feature = feature.transpose((1, 2, 0))
sleep_stage = sleep_stage[0:ne, :]

# preprocess features
print('Preprocessing features...')

x, y = np.where(np.sum(np.isnan(feature) | np.isinf(feature), axis=2) > 0)

for ii in range(len(x)):
    feature[x[ii], y[ii], :] = np.nan

featfeat = np.zeros((num_feats, num_channels))

for nch in range(0, num_channels):
    for nf in range(0, num_feats):
        # outlier detection
        f = feature[nch, :, nf]
        in_nan = np.isnan(f)
        f_nan_removed = f[~in_nan]
        f_nan_removed = f_nan_removed - (np.convolve(f_nan_removed, np.ones(10), mode='same') / 10)

        fsort = np.sort(f_nan_removed)
        m = fsort[[round(0.25 * len(fsort)), round(0.75 * len(fsort))]]
        del_nan = np.isnan(feature[nch, :, nf])
        del_nan[~in_nan] = (f_nan_removed < m[0] - 2.5 * (m[1] - m[0])) | (
                f_nan_removed > m[1] + 2.5 * (m[1] - m[0])) | np.isnan(f_nan_removed)

        feature[nch, del_nan, nf] = np.nan

        # smoothing
        f = feature[nch, :, nf]
        ff = f.copy()
        in_nan = np.isnan(ff)
        ff = ff[~in_nan]
        ff = np.convolve(ff, np.ones(3), mode='same') / 3
        f[~in_nan] = ff

        # normalizing
        if np.any(f):
            night = night.astype(bool)
            f = (f - np.nanmean(f[night[:, 0]])) / np.nanstd(f[night[:, 0]])

        # get features coordinates
        feature[nch, :, nf] = f
        in_nan = np.isnan(f)
        f = f[~in_nan]
        fm = np.convolve(f, np.ones(10), 'same') / 10
        f = np.linalg.norm(f)

        if f:
            featfeat[nf, nch] = np.linalg.norm(fm) / np.linalg.norm(f)

featfeat = featfeat.T

print('Saving features to pickle...')

with open('sleep_features.pkl', 'wb') as f:
    pkl.dump(feature, f)
