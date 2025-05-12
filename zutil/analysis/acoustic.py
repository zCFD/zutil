"""
Copyright (c) 2012-2025, Zenotech Ltd
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Zenotech Ltd nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL ZENOTECH LTD BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

acoustic post processing functions
"""

import numpy as np
from collections.abc import Iterable
from typing import Union
from scipy import signal


def calculate_PSD(
    p: list,
    time: Union[Iterable, float],
    sampling_frequency: float = 1.0,
) -> tuple:
    """
    Calculate Power Spectral Density (PSD) using Welch's method.

    Args:
        T (Arraylike): Time data array
        p (Arraylike): Pressure data array
        sampling_frequency (float = 1.0): Sampling frequency in Hz.
    Returns:
        (frequency, psd values): Frequencies and corresponding PSD values.
    """
    if isinstance(time, Iterable):
        # If time is a list, calculate the time step (dT)
        dT = time[1] - time[0]
    elif isinstance(time, float):
        # If time is a float, use it as the time step (dT)
        dT = time
    else:
        raise ValueError("Time must be a list or a float")

    # Parameters for Welch's method in frequency analysis
    fs = 1.0 / dT  # Sampling frequency
    nps = (1.0 / sampling_frequency) * fs  # Number of points per segment

    # Compute Power Spectral Density (PSD) using Welch's method
    f, Pxx_den = signal.welch(p, fs, nperseg=nps, scaling="density")  # Welch's method

    return f, Pxx_den


def convert_to_dB(Pxx_den: list, inv_pref: float = 1.0 / 2e-5) -> list:
    """
    Convert Power Spectral Density (PSD) to decibels (dB).

    Args:
        Pxx_den (Arraylike): Power Spectral Density values.
        inv_pref (float = 1.0/2e-5): Inverse reference pressure for SPL in dB.
    Returns:
        Arraylike: Converted PSD values in dB.
    """
    return [10 * np.log10(xx * inv_pref**2) for xx in Pxx_den]


def calculate_thirdoctave_bands(
    f: list,
    Pxx_den: list,
) -> tuple[list, list]:
    """
    Calculate third-octave bands from Power Spectral Density (PSD).

    Args:
        f (Arraylike): Frequency data array.
        Pxx_den (Arraylike): Power Spectral Density values.
    Returns:
        (band_frequencies, band_power): Third-octave band data.
    """
    # Define third-octave frequency bands
    third_octave_bands = [
        [11.2, 12.5, 14.1],
        [14.1, 16, 17.8],
        [17.8, 20, 22.4],
        [22.4, 25, 28.2],
        [28.2, 31.5, 35.5],
        [35.5, 40, 44.7],
        [44.7, 50, 56.2],
        [56.2, 63, 70.8],
        [70.8, 80, 89.1],
        [89.1, 100, 112],
        [112, 125, 141],
        [141, 160, 178],
        [178, 200, 224],
        [224, 250, 282],
        [282, 315, 355],
        [355, 400, 447],
        [447, 500, 562],
        [562, 630, 708],
        [708, 800, 891],
        [891, 1000, 1122],
        [1122, 1250, 1413],
        [1413, 1600, 1778],
        [1778, 2000, 2239],
        [2239, 2500, 2818],
        [2818, 3150, 3548],
        [3548, 4000, 4467],
        [4467, 5000, 5623],
        [5623, 6300, 7079],
        [7079, 8000, 8913],
        [8913, 10000, 11220],
        [11220, 12500, 14130],
        [14130, 16000, 17780],
        [17780, 20000, 22390],
    ]

    # Initialize data structures for third-octave band processing
    third_octave_data = [0.0 for i in third_octave_bands]
    third_octave_freq = [band[1] for band in third_octave_bands]
    df = f[1] - f[0]

    # Populate third-octave band data by summing within each band's frequency range
    for freq, psd in zip(f, Pxx_den):
        bin_min = freq - 0.5 * df
        bin_max = freq + 0.5 * df
        if 11.2 <= freq <= 22390.0:
            for j, band in enumerate(third_octave_bands):
                band_min, band_center, band_max = band
                if bin_min >= band_min and bin_max < band_max:
                    third_octave_data[j] += psd * df
                elif bin_min < band_min and bin_max >= band_max:
                    third_octave_data[j] += psd * (band_max - band_min)
                elif bin_min < band_min and bin_max >= band_min:
                    third_octave_data[j] += psd * (bin_max - band_min)
                elif bin_min < band_max and bin_max >= band_max:
                    third_octave_data[j] += psd * (band_max - bin_min)

    # clip values to avoid undefined behaviour
    non_zero_indices = np.nonzero(third_octave_data)
    if len(non_zero_indices[0]) == 0:
        raise ValueError("No non-zero elements found in third-octave data")
    else:
        start, end = non_zero_indices[0][0], non_zero_indices[0][-1] + 1
        filtered_freq = third_octave_freq[start:end]
        filtered_data = third_octave_data[start:end]

    return filtered_freq, filtered_data
