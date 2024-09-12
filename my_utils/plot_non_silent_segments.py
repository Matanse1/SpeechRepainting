from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import itertools
import math

def calc_rms(audio_segment):
    """
    Calculate the root mean square of the audio segment.
    """
    signal = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    rms = np.sqrt(np.mean(np.square(signal)))
    return rms


def detect_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, seek_step=1):
    """
    Returns a list of all silent sections [start, end] in milliseconds of audio_segment.
    Inverse of detect_nonsilent()

    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    """
    seg_len = len(audio_segment)

    # you can't have a silent portion of a sound that is longer than the sound
    if seg_len < min_silence_len:
        return []

    # convert silence threshold to a float value (so we can compare it to rms)
    silence_thresh = db_to_float(silence_thresh) * audio_segment.max_possible_amplitude
    # find silence and add start and end indicies to the to_cut list
    silence_starts = []

    # check successive (1 sec by default) chunk of sound for silence
    # try a chunk at every "seek step" (or every chunk for a seek step == 1)
    last_slice_start = seg_len - min_silence_len
    slice_starts = range(0, last_slice_start + 1, seek_step)

    # guarantee last_slice_start is included in the range
    # to make sure the last portion of the audio is searched
    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])

    for i in slice_starts:
        audio_slice = audio_segment[i:i + min_silence_len]
        # print(f"rms: {audio_slice.rms} \t i = {i}, \t silence_thresh = {silence_thresh}")
        if audio_slice.rms <= silence_thresh:
            silence_starts.append(i)
            if i > 3*1000  and i < 4*1000:
                my_rms = calc_rms(audio_slice)
                print(f"Getting TRUE rms: {audio_slice.rms} \t my_rms = {my_rms} \t  i = {i}, \t silence_thresh = {silence_thresh}")
        else:
            if i > 3*1000  and i < 4*1000:
                my_rms = calc_rms(audio_slice)
                print(f"Getting FALSE rms: {audio_slice.rms} \t my_rms = {my_rms} \t i = {i}, \t silence_thresh = {silence_thresh}")

    # short circuit when there is no silence
    if not silence_starts:
        return []

    # combine the silence we detected into ranges (start ms - end ms)
    silent_ranges = []

    prev_i = silence_starts.pop(0)
    current_range_start = prev_i

    for silence_start_i in silence_starts:
        continuous = (silence_start_i == prev_i + seek_step)

        # sometimes two small blips are enough for one particular slice to be
        # non-silent, despite the silence all running together. Just combine
        # the two overlapping silent ranges.
        silence_has_gap = silence_start_i > (prev_i + min_silence_len)

        if not continuous and silence_has_gap:
            silent_ranges.append([current_range_start,
                                  prev_i + min_silence_len])
            current_range_start = silence_start_i
        prev_i = silence_start_i

    silent_ranges.append([current_range_start,
                          prev_i + min_silence_len])

    return silent_ranges


def db_to_float(db, using_amplitude=True):
    """
    Converts the input db to a float, which represents the equivalent
    ratio in power.
    """
    db = float(db)
    if using_amplitude:
        return 10 ** (db / 20)
    else:  # using power
        return 10 ** (db / 10)
    
    
def numpy_to_audiosegment(audio: np.ndarray, sample_rate: int) -> AudioSegment:
    """
    Convert a NumPy array to an AudioSegment object.
    
    :param audio: NumPy array containing the audio data.
    :param sample_rate: Sample rate of the audio (samples per second).
    :return: AudioSegment object.
    """
    audio = (audio * 32767).astype(np.int16)  # Convert to 16-bit PCM format
    return AudioSegment(audio.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1)

def plot_nonsilent_segments_from_numpy(audio: np.ndarray, sample_rate: int, silence_thresh: int = -20, min_silence_len: int = 500):
    """
    Plot the non-silent segments of a NumPy audio array.

    :param audio: NumPy array containing the audio data.
    :param sample_rate: Sample rate of the audio (samples per second).
    :param silence_thresh: Silence threshold in dBFS. Audio quieter than this will be considered silence.
    :param min_silence_len: Minimum length of silence (in ms) to detect.
    """
    # Convert NumPy array to AudioSegment
    audio_segment = numpy_to_audiosegment(audio, sample_rate)

    # Detect the non-silent chunks of the audio
    nonsilent_ranges = detect_nonsilent(audio_segment, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    
    # Convert the audio to raw data for plotting
    time = np.linspace(0, len(audio) / sample_rate, num=len(audio))  # Convert to seconds

    # Plot the audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio, label='Audio Signal', alpha=0.6)

    # Highlight non-silent segments
    # start, end = nonsilent_ranges[0]
    # plt.axvspan(start / 1000, end / 1000, color='green', alpha=0.3, label='Non-Silent Segment')
    for i, (start, end) in enumerate(nonsilent_ranges):
        # print(f"The rms is:{audio_segment[start:end].rms}, \t the start is: {start}, \t the end is: {end}")
        plt.axvspan(start / 1000, end / 1000, color='green', alpha=0.3, label='Non-Silent Segment')
        # if i ==1:
        #     break

    plt.title('Non-Silent Segments in Audio')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('/home/dsi/moradim/SpeechRepainting/temp_dir/non_silent_segments.png')

# Example usage:
silence_thresh = -20 #20*np.log10(0.1)
print(f"Silence threshold: {silence_thresh:.2f} dB")
explosion_path = '/dsi/gannot-lab1/datasets/FSD50K/FSD50K.dev_audio_16k/40971.wav'
rate1, explosion = wavfile.read(explosion_path)
audio = explosion / max(abs(explosion))
output_file = '/home/dsi/moradim/SpeechRepainting/temp_dir/example_silent_checking.wav'
wavfile.write(output_file, rate1, audio)
plot_nonsilent_segments_from_numpy(audio, sample_rate=rate1, silence_thresh=silence_thresh, min_silence_len=50)

# 178180,
# 155654,
# ,184418
# 40971,
# ,
# ,
# 32799,
# 350264,
# 32841,
# 32842,
# ,
# ,
