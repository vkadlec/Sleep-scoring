import pandas as pd
import numpy as np


df = pd.read_csv('C:/Users/vojta/PycharmProjects/ICRC/SleepSEEG/sleep_score.csv')
df_stage = pd.DataFrame()

for patient in df.pat.unique():
    df_patient = df[df.pat == patient]

    start = df_patient.start_time.min()
    stop = df_patient.end_time.max()

    # create thirty-second intervals for patient
    intervals = np.arange(start, stop, 30 * 1e6)
    intervals = pd.DataFrame(data={'patient': (len(intervals) - 1) * [str(patient)], 'starts': intervals[:-1],
                                   'stops': intervals[1:]})

    # assign sleep stage for each interval
    for i, row in df_patient.iterrows():
        intervals.loc[
            (intervals.starts >= row.start_time) & (intervals.stops <= row.end_time), 'sleep_stage'] = row.sleep_stage

    df_stage = pd.concat([df_stage, intervals])

df_final = df_stage[df_stage.sleep_stage.notna()]
df_final.to_csv('patient_stages.csv', index=False)
