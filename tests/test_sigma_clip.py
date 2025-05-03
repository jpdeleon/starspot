import numpy as np
import matplotlib.pyplot as plt
from starspot.rotation_tools import filter_sigma_clip, sigma_clip


def test_sigma_clip():
    """Test the sigma_clip function for outlier detection."""
    np.random.seed(42)
    N, Nout = 1000, 20
    t0 = np.linspace(0, 100, N)
    p = 10
    w = 2*np.pi/p
    y0 = np.sin(w*t0) + np.random.randn(N)*.1

    # Add some outliers
    inds = np.random.choice(np.arange(len(y0)), Nout)
    y0[inds] += np.random.randn(Nout)*10.

    # Initial removal of extreme outliers
    m = sigma_clip(y0, nsigma=7)

    # Check that the mask has the right shape
    assert len(m) == N

    # Check that the mask is mostly True (most points kept)
    assert np.sum(m) >= N - Nout

    # Check that the outliers were removed
    t, y = t0[m], y0[m]
    assert len(t) < N  # Some points should be removed
    assert np.std(y) < np.std(y0)  # Standard deviation should be smaller after clipping

    # The standard deviation after clipping will be higher than the noise level
    # because we still have the sine wave component
    assert 0.1 < np.std(y) < 1.0


def test_filter_sigma_clip():
    """Test the filter_sigma_clip function for smoothing and outlier detection."""
    np.random.seed(42)
    N, Nout = 1000, 20
    t0 = np.linspace(0, 100, N)
    p = 10
    w = 2*np.pi/p
    y0 = np.sin(w*t0) + np.random.randn(N)*.1

    # Add some outliers
    inds = np.random.choice(np.arange(len(y0)), Nout)
    y0[inds] += np.random.randn(Nout)*10.

    # Initial removal of extreme outliers
    m = sigma_clip(y0, nsigma=7)
    t, y = t0[m], y0[m]

    # Apply filter_sigma_clip
    smooth, mask = filter_sigma_clip(t, y, nsigma=3, window_length=49, polyorder=2)

    # Check that the smooth array has the right shape
    assert len(smooth) == len(t)

    # Check that the mask has the right shape
    assert len(mask) == len(t)

    # Calculate residuals
    resids = y - smooth

    # Check that the residuals of non-outlier points are small
    assert np.std(resids[mask]) < 0.2

    # Check that the residuals of outlier points are large
    if np.sum(~mask) > 0:  # If there are any outliers detected
        assert np.std(resids[~mask]) > 0.2


if __name__ == "__main__":
    test_sigma_clip()
    test_filter_sigma_clip()
