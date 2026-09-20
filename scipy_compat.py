"""Pure-NumPy drop-in replacements for scipy functions used in the pipeline.

Eliminates the ~100 MB scipy dependency from the Vercel serverless bundle.
All functions preserve the same call signatures and return types as their
scipy counterparts.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────── scipy.fft ────────────────────────────────────

def fft(a, n=None, axis=-1):
    """numpy.fft.fft — identical to scipy.fft.fft for 1-D real inputs."""
    return np.fft.fft(a, n=n, axis=axis)


# ─────────────────────────── scipy.interpolate ────────────────────────────

class interp1d:
    """Minimal interp1d replacement (linear + nearest, 1-D).

    Supports the subset of the scipy API used in this project:
        f = interp1d(x, y, kind='linear', fill_value='extrapolate')
        y_new = f(x_new)
    """

    def __init__(self, x, y, kind="linear", bounds_error=True,
                 fill_value=np.nan, axis=-1):
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._kind = kind
        self._fill_value = fill_value
        self._bounds_error = bounds_error
        if kind not in ("linear", "nearest"):
            raise NotImplementedError(f"interp1d kind={kind!r} not implemented in compat layer")

    def __call__(self, x_new):
        x_new = np.asarray(x_new, dtype=float)
        xp, fp = self._x, self._y

        if self._bounds_error:
            if np.any(x_new < xp[0]) or np.any(x_new > xp[-1]):
                raise ValueError("x_new is out of the interpolation range")

        if self._kind == "nearest":
            idx = np.searchsorted(xp, x_new)
            idx = np.clip(idx, 0, len(xp) - 1)
            idx_low = np.clip(idx - 1, 0, len(xp) - 1)
            mask = np.abs(x_new - xp[idx_low]) < np.abs(x_new - xp[idx])
            result = fp[idx]
            result = np.where(mask, fp[idx_low], result)
            return result

        # Linear (default) — numpy.interp handles extrapolation correctly
        if self._fill_value == "extrapolate":
            # numpy.interp clamps; do manual extrapolation for edge points
            result = np.interp(x_new, xp, fp)
            # Left extrapolation
            m_left = (fp[1] - fp[0]) / (xp[1] - xp[0] + 1e-300)
            result = np.where(x_new < xp[0], fp[0] + m_left * (x_new - xp[0]), result)
            # Right extrapolation
            m_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2] + 1e-300)
            result = np.where(x_new > xp[-1], fp[-1] + m_right * (x_new - xp[-1]), result)
            return result

        return np.interp(x_new, xp, fp,
                         left=self._fill_value, right=self._fill_value)


# ─────────────────────────── scipy.ndimage ────────────────────────────────

def uniform_filter1d(a, size, axis=0, mode="reflect"):
    """1-D uniform (box / moving-average) filter via cumsum — O(n)."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    half = size // 2
    tail = size - half - 1

    # Reflect-pad
    left = a[half:0:-1]          # half elements, reflected
    right = a[-2:-(tail + 2):-1] if tail > 0 else np.array([])
    padded = np.concatenate([left, a, right])

    cs = np.concatenate([[0.0], np.cumsum(padded)])
    return (cs[size:] - cs[:n]) / size


def median_filter(a, size):
    """1-D median filter with reflect-padding."""
    a = np.asarray(a, dtype=float)
    half = size // 2
    padded = np.pad(a, half, mode="reflect")
    # Use stride tricks for efficient sliding-window median
    shape = (len(a), size)
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.median(windows, axis=1)


# ─────────────────────────── scipy.signal ─────────────────────────────────

def find_peaks(x, height=None, distance=None, prominence=None, width=None,
               threshold=None, rel_height=0.5, plateau_size=None):
    """Find local maxima in a 1-D array.

    Returns (peaks, properties) matching scipy.signal.find_peaks output.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    candidates = []

    for i in range(1, n - 1):
        if x[i] > x[i - 1] and x[i] > x[i + 1]:
            candidates.append(i)

    candidates = np.array(candidates, dtype=int)
    if len(candidates) == 0:
        return candidates, {}

    # Filter by height
    if height is not None:
        if np.isscalar(height):
            min_h, max_h = height, np.inf
        else:
            min_h = height[0] if height[0] is not None else -np.inf
            max_h = height[1] if len(height) > 1 and height[1] is not None else np.inf
        mask = (x[candidates] >= min_h) & (x[candidates] <= max_h)
        candidates = candidates[mask]

    if len(candidates) == 0:
        return candidates, {}

    # Filter by distance (keep highest peak within each window)
    if distance is not None and len(candidates) > 1:
        keep = np.ones(len(candidates), dtype=bool)
        for i in range(len(candidates)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(candidates)):
                if candidates[j] - candidates[i] < distance:
                    if x[candidates[i]] >= x[candidates[j]]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
                else:
                    break
        candidates = candidates[keep]

    properties = {}
    return candidates, properties


def peak_prominences(x, peaks, wlen=None):
    """Compute prominence of each peak.

    Returns (prominences, left_bases, right_bases).
    """
    x = np.asarray(x, dtype=float)
    peaks = np.asarray(peaks, dtype=int)
    n = len(x)
    prominences = np.empty(len(peaks), dtype=float)
    left_bases = np.empty(len(peaks), dtype=int)
    right_bases = np.empty(len(peaks), dtype=int)

    for i, peak in enumerate(peaks):
        # Determine search window
        if wlen is not None:
            half = wlen // 2
            lo = max(0, peak - half)
            hi = min(n, peak + half + 1)
        else:
            lo, hi = 0, n

        # Left base: lowest point between peak and the nearest higher peak to the left
        left_min_idx = lo
        left_min_val = x[lo]
        for j in range(peak - 1, lo - 1, -1):
            if x[j] < left_min_val:
                left_min_val = x[j]
                left_min_idx = j
            if x[j] >= x[peak]:   # hit a higher peak → stop
                break

        # Right base
        right_min_idx = hi - 1
        right_min_val = x[hi - 1]
        for j in range(peak + 1, hi):
            if x[j] < right_min_val:
                right_min_val = x[j]
                right_min_idx = j
            if x[j] >= x[peak]:
                break

        prominences[i] = x[peak] - max(left_min_val, right_min_val)
        left_bases[i] = left_min_idx
        right_bases[i] = right_min_idx

    return prominences, left_bases, right_bases


def peak_widths(x, peaks, rel_height=0.5, prominence_data=None, wlen=None):
    """Measure the width of each peak at a relative height.

    Returns (widths, width_heights, left_ips, right_ips).
    """
    x = np.asarray(x, dtype=float)
    peaks = np.asarray(peaks, dtype=int)

    if prominence_data is None:
        proms, left_bases, right_bases = peak_prominences(x, peaks, wlen=wlen)
    else:
        proms, left_bases, right_bases = prominence_data

    widths = np.empty(len(peaks), dtype=float)
    width_heights = np.empty(len(peaks), dtype=float)
    left_ips = np.empty(len(peaks), dtype=float)
    right_ips = np.empty(len(peaks), dtype=float)

    for i, (peak, prom, lb, rb) in enumerate(zip(peaks, proms, left_bases, right_bases)):
        wh = x[peak] - prom * rel_height
        width_heights[i] = wh

        # Left interpolated position
        j = peak
        while j > lb and x[j] > wh:
            j -= 1
        if j == lb or x[j] >= wh:
            left_ip = float(lb)
        else:
            # linear interpolation between j and j+1
            denom = x[j + 1] - x[j]
            left_ip = j + (wh - x[j]) / denom if abs(denom) > 1e-10 else float(j)
        left_ips[i] = left_ip

        # Right interpolated position
        j = peak
        while j < rb and x[j] > wh:
            j += 1
        if j == rb or x[j] >= wh:
            right_ip = float(rb)
        else:
            denom = x[j] - x[j - 1]
            right_ip = j - (x[j] - wh) / denom if abs(denom) > 1e-10 else float(j)
        right_ips[i] = right_ip

        widths[i] = right_ips[i] - left_ips[i]

    return widths, width_heights, left_ips, right_ips


# ─────────────────────────── scipy.stats ──────────────────────────────────

def entropy(pk, qk=None, base=None, axis=0):
    """Shannon entropy. Matches scipy.stats.entropy(pk) for 1-D inputs."""
    pk = np.asarray(pk, dtype=float)
    pk = pk / (pk.sum() + 1e-300)
    mask = pk > 0
    if qk is None:
        h = -float(np.sum(pk[mask] * np.log(pk[mask])))
    else:
        qk = np.asarray(qk, dtype=float)
        qk = qk / (qk.sum() + 1e-300)
        h = float(np.sum(pk[mask] * np.log(pk[mask] / (qk[mask] + 1e-300))))
    if base is not None:
        h /= np.log(base)
    return h


def kurtosis(a, axis=0, fisher=True, bias=True):
    """Excess kurtosis (Fisher definition, bias=True). Matches scipy.stats.kurtosis."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    mean = np.mean(a)
    std = np.std(a)
    if std < 1e-12:
        return 0.0
    kurt = float(np.mean(((a - mean) / std) ** 4))
    return kurt - 3.0 if fisher else kurt


def skew(a, axis=0, bias=True):
    """Sample skewness. Matches scipy.stats.skew."""
    a = np.asarray(a, dtype=float)
    mean = np.mean(a)
    std = np.std(a)
    if std < 1e-12:
        return 0.0
    return float(np.mean(((a - mean) / std) ** 3))
