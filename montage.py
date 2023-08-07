import numpy as np


def __indices(lst, element):
    """
    Function to find all occurrences of element in list

    :param lst: Channel names
    :param element: Utility channel
    :return:
    """

    result = []
    offset = -1
    while True:
        try:
            offset = lst.index(element, offset + 1)
        except ValueError:
            return result
        result.append(offset)


def get_utility_channels():
    return ['EAS', 'fd1', 'fd2', 'mic', 'Fz', 'Cz', 'Cz_1', 'Pz', 'EEG2', 'Ms1', 'Ms2', 'A12_1',
            'EOG1', 'EOG2', 'EOG3', 'EOG4', 'MS1', 'MS2', 'eA12', 'EKG2', 'EKG1', 'EKG4',
            'CAL', 'SDT', 'EVT', 'BmRf', 'EKG3', 'ECG', 'Oz', 'EKG', 'FPz', 'FP1', 'FP2',
            'FP1-Cz_f', 'FP2-Cz_f', 'Fz-Cz_f', 'Cz-Oz_f', 'REF_wm', 'cz', 'ecg', 'REF_orig',
            'foto', 'e', 'MM1', 'MM2', 'A12A', 'A12B', 'A12C', 'A12D', 'f3', 'c3', 'p3', 'f4',
            'f4', 'c4', 'p4', 'fz', 'pz', 'x', 'y', 'z', 'v', 'w', 'ekg', 'f3-ref', 'c3-ref',
            'p3-ref', 'f4-ref', 'f4-ref', 'c4-ref', 'p4-ref', 'fz-ref', 'pz-ref', 'EEG F3-Ref', 'EEG C3-Ref',
            'EEG P3-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG Pz-Ref', 'EEG F4-Ref', 'EEG C4-Ref', 'EEG P4-Ref',
            'POL EKG']


def remove_utility_channels(channel_names):
    """
    Function to remove utility channels

    :param channel_names: Channels names
    :return: Channel names without utility channels
    """

    not_needed = get_utility_channels()

    if isinstance(channel_names[0], str):
        index = []
        for i in np.arange(len(not_needed)):
            try:
                ind = __indices(channel_names, not_needed[i])
                index = index + ind
            except:
                pass
        ch_numbers = np.arange(len(channel_names))
        ch_numbers = [x for x in ch_numbers if x not in index]
        channel_names = [channel_names[k] for k in ch_numbers]

        return channel_names

    elif isinstance(channel_names[0], dict):

        ch_names = [channel_names[x]['name'] for x in np.arange(len(channel_names))]
        index = []
        for i in np.arange(len(not_needed)):
            try:
                ind = __indices(ch_names, not_needed[i])
                index = index + ind
            except:
                pass

        ch_numbers = np.arange(len(channel_names))
        ch_numbers = [x for x in ch_numbers if x not in index]
        channel_names = [channel_names[k] for k in ch_numbers]

        return channel_names


def channel_sort_list(channels):
    """
    Function to add 0s to unipolar channels and sort them

    :param channels: Unipolar channels
    :return: Modified and sorted unipolar channels
    """

    # rename unipolar channels and add 0s - for good ordering
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


def define_pairs(channels):
    """
    Function to define bipolar channels and bipolar pairs

    :param channels: Unipolar channels
    :return: Defined bipolar channels and bipolar pairs
    """

    channels = remove_utility_channels(channels)
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

    return bipolar_pairs, bipolar_names


def bipolar_montage(data, channels, pairs):
    """
    Function to define bipolar channels and compute signal of bipolar measurement

    :param data: Thirty-second signal segment
    :param channels: Unipolar channels
    :param pairs: Bipolar pairs
    :return: Computed bipolar data
    """

    if len(data.T) != len(channels):
        assert 'Number of channel names must be equal to number of signals.'

    ch_dict = dict(zip(channels, np.arange(len(channels))))

    bi_data = np.zeros([len(data), len(pairs)], dtype='float64')

    for i, pair in enumerate(pairs):
        bi_data[:, i] = data[:, ch_dict[pair[0]]] - data[:, ch_dict[pair[1]]]

    return bi_data
