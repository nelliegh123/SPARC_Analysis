import numpy as np

def otsu_threshold(
    tots: np.ndarray,
    n_bins: int = 256,
    valley_ratio_max: float = 0.1,
    min_kept_fraction: float = 0.05,
    fallback: float = 1e-3,
) -> float:
    """Split a bimodal ToT distribution at the between-class-variance maximum

    The ToT spectrum has two peaks: injection chargeup (lower) and discharge
    (upper). This returns the threshold separating the two, which tracks the
    settings instead of relying on a fixed value.

    Otsu always returns a split, even when there is nothing to separate. Two
    guards reject such spurious splits and fall back to a fixed threshold:
    the split must sit in a clear valley (density at the threshold below
    valley_ratio_max times the highest peak), and it must keep a meaningful
    fraction of events above it. The latter rejects splits that only carve off
    a handful of isolated noise hits.

    The fixed threshold is fallback [µs]. The default 0.0 keeps every hit when
    there is no chargeup peak, since on sparc7 the real ToT is below 1 µs at
    low Vh. Hits from a merged chargeup + real pulse (~32 µs) have to be
    removed before calling this, or Otsu splits them off instead.
    """
    if len(tots) < 3:
        return fallback

    bin_counts, bin_edges = np.histogram(tots, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    weight_below = np.cumsum(bin_counts)
    weight_above = weight_below[-1] - weight_below
    mean_below = np.cumsum(bin_counts * bin_centers)
    mean_total = mean_below[-1]

    # Only bins with entries on both sides give a defined split
    valid = (weight_below > 0) & (weight_above > 0)
    mean_below_valid = np.divide(
        mean_below, weight_below, out=np.zeros_like(mean_below), where=valid
    )
    mean_above_valid = np.divide(
        mean_total - mean_below,
        weight_above,
        out=np.zeros_like(mean_below),
        where=valid,
    )

    variance_between = (
        weight_below * weight_above * (mean_below_valid - mean_above_valid) ** 2
    )
    variance_between[~valid] = 0.0
    threshold = bin_centers[np.argmax(variance_between)]

    # Reject the split if it does not sit in a clear valley (i.e. the spectrum
    # is unimodal and there is nothing to separate)
    smoothed = np.convolve(bin_counts, np.ones(5) / 5, mode="same")
    if smoothed.max() > 0:
        i_threshold = int(np.argmin(np.abs(bin_centers - threshold)))
        if smoothed[i_threshold] > valley_ratio_max * smoothed.max():
            return fallback

    # Reject the split if it leaves almost nothing above the threshold (i.e.
    # it only separated a few isolated noise hits from the main peak)
    if np.count_nonzero(tots > threshold) < min_kept_fraction * len(tots):
        return fallback

    return threshold