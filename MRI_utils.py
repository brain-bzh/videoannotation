
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nistats import hemodynamic_models

def experimental_matrix(trials_length, inter_trials_interval, stimuli_numbers, stimuli_duration, inter_stimuli_intervals, tr = 1, amplitude_stimuli = 1):
    
    total_length = sum(trials_length)+(len(trials_length)+1)*inter_trials_interval
    frame_times = np.arange(start = 0, stop = total_length, step = tr)

    onsets = []
    for i, trial in enumerate(trials_length) : 
        onsets_trial = np.linspace(0, trial, stimuli_numbers)+(inter_trials_interval*(i+1)+sum(trials_length[:i]))
        onsets = np.concatenate((onsets, onsets_trial))
    onsets = np.round(onsets.reshape(onsets.shape[0], 1), decimals=2)

    durations = []
    for i in stimuli_duration :
        dur = np.ones(stimuli_numbers)
        dur = dur * i
        durations = np.concatenate((durations, dur))
    durations = np.round(durations.reshape(durations.shape[0], 1), decimals=2)

    amplitudes = []
    if isinstance(amplitude_stimuli, (int, float)) :
        amplitudes = np.ones_like(onsets)*amplitude_stimuli
    elif isinstance(amplitude_stimuli, (list, np.ndarray)):
        amplitudes = np.array(amplitude_stimuli)
    amplitudes = amplitudes.reshape(amplitudes.shape[0], -1)

    if onsets.shape[1] != amplitudes.shape[1]:
        onsets = np.repeat(onsets, amplitudes.shape[1], axis = 1)
    if durations.shape[1] != amplitudes.shape[1]:
        durations = np.repeat(durations, amplitudes.shape[1], axis = 1)

    #stimuli = np.zeros_like(frame_times)
    #for start, stimuli_duration in zip(onsets[:,0], durations[:,0]):
    #    stimuli[int(start):int(start+stimuli_duration)] = 1
    #exp_stimuli.append(stimuli)

    return (onsets, durations, amplitudes), frame_times#, exp_stimuli

def design_matrix_for_Cneuromod(audio_length, signal_or_FV, audiopad = 0, tr = 1.49):
    trials_length = [audio_length-(tr*2*audiopad)]
    inter_trials_interval = 0

    stimuli_amplitude = signal_or_FV
    stimuli_numbers = len(stimuli_amplitude)
    stimuli_duration = [tr]
    inter_stimuli_intervals = np.zeros_like(stimuli_amplitude)

    (onsets, durations, amplitudes), frame_times = experimental_matrix(trials_length, 
                            inter_trials_interval, stimuli_numbers, stimuli_duration, 
                            inter_stimuli_intervals, tr, stimuli_amplitude)

    return (onsets, durations, amplitudes), frame_times