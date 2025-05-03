import numpy as np
import pytest
from starspot import phase_dispersion_minimization as pdm
import starspot as ss


def test_sj2():
    """Test the sj2 function for variance calculation."""
    np.random.seed(42)
    N = 10000
    x = np.random.randn(N)  # Normal distribution with mean 0, std 1
    sj2 = pdm.sj2(x, 0, N)
    # Variance of a standard normal distribution should be close to 1
    assert np.isclose(sj2, 1, atol=.01)


def test_s2():
    """Test the s2 function for overall variance calculation."""
    np.random.seed(42)
    N = 10000
    M = 10
    nj = np.ones(M) * N
    sj2 = np.zeros(M)
    for j in range(M):
        x = np.random.randn(N)
        sj2[j] = pdm.sj2(x, 0, nj[j])
    s2 = pdm.s2(nj, sj2, M)
    # Overall variance should be close to 1
    assert np.isclose(s2, 1, atol=.01)


def test_calc_phase():
    """Test the calc_phase function."""
    # Create time array
    t = np.array([0, 5, 10, 15, 20, 25])

    # Test with period = 10
    phase = pdm.calc_phase(10, t)
    expected_phase = np.array([0, 0.5, 0, 0.5, 0, 0.5])
    assert np.allclose(phase, expected_phase)

    # Test with period = 5
    phase = pdm.calc_phase(5, t)
    expected_phase = np.array([0, 0, 0, 0, 0, 0])
    assert np.allclose(phase, expected_phase)


def test_phase_bins(synthetic_data):
    """Test the phase_bins function with synthetic data."""
    t, x, _, p = synthetic_data
    nbins = 10

    # Test with the correct period (10)
    phase = pdm.calc_phase(p, t)
    x_means, phase_bins, Ns, sj2s, x_binned, phase_binned = \
        pdm.phase_bins(nbins, phase, x)

    # Check that the output arrays have the correct shapes
    assert len(x_means) == nbins
    assert len(phase_bins) == nbins + 1
    assert len(Ns) == nbins
    assert len(sj2s) == nbins
    assert len(x_binned) == nbins
    assert len(phase_binned) == nbins

    # Check that the bins cover the full phase range [0, 1]
    assert phase_bins[0] == 0
    assert phase_bins[-1] == 1

    # Check that the number of points in each bin is approximately equal
    assert np.isclose(np.std(Ns) / np.mean(Ns), 0, atol=0.2)

    # Try different periods and check that the correct period has the lowest variance
    periods = [2.5, 5, p]
    s2_values = []

    for period in periods:
        phase = pdm.calc_phase(period, t)
        x_means, phase_bins, Ns, sj2s, _, _ = pdm.phase_bins(nbins, phase, x)
        s2_values.append(pdm.s2(Ns, sj2s, nbins))

    # The correct period should have the lowest s2 value
    assert s2_values[2] < s2_values[0]
    assert s2_values[2] < s2_values[1]


def test_phi(synthetic_data):
    """Test the phi function with synthetic data."""
    t, x, _, p = synthetic_data
    nbins = 10

    # Calculate phi for different periods
    periods = [5, 7.5, p, 12.5, 15]
    phis = [pdm.phi(nbins, period, t, x) for period in periods]

    # The correct period should have the lowest phi value
    assert phis[2] == min(phis)

    # Test with a range of periods
    nperiods = 100
    period_grid = np.linspace(5, 15, nperiods)
    phi_values = np.array([pdm.phi(nbins, period, t, x) for period in period_grid])

    # Find the period with the minimum phi
    best_period = period_grid[np.argmin(phi_values)]

    # The best period should be close to the true period
    assert np.isclose(best_period, p, atol=0.1)


def test_uncertainty(synthetic_data):
    """Test the uncertainty estimation in PDM."""
    t, x, xerr, p = synthetic_data

    # Create RotationModel instance
    rm = ss.RotationModel(t, x, xerr)

    # Test with period grid around the true period
    nperiods = 100
    period_grid = np.linspace(5, 15, nperiods)
    pdm_period, period_err = rm.pdm_rotation(period_grid)

    # The PDM period should be close to the true period
    assert np.isclose(pdm_period, p, atol=0.5)

    # The uncertainty should be positive and reasonable
    assert period_err > 0
    assert period_err < p / 2  # Uncertainty should be less than half the period


@pytest.mark.parametrize("test_period", [2, 5, 10, 20, 50])
def test_pdm_different_periods(test_period):
    """Test PDM with different input periods."""
    # Generate data with the specified period
    np.random.seed(42)
    t = np.linspace(0, 100, 1000)
    w = 2*np.pi/test_period
    x = np.sin(w*t) + np.random.randn(len(t))*1e-2
    xerr = np.ones_like(x)*1e-2

    # Create RotationModel instance
    rm = ss.RotationModel(t, x, xerr)

    # Create period grid centered around the test period
    nperiods = 100
    period_grid = np.linspace(max(0.1, test_period/2), test_period*1.5, nperiods)

    # Calculate PDM period
    pdm_period, period_err = rm.pdm_rotation(period_grid)

    # The PDM period should be close to the true period
    assert np.isclose(pdm_period, test_period, rtol=0.1)


def test_gaussian_fit():
    """Test the Gaussian fitting function used for uncertainty estimation."""
    # Create a Gaussian curve with known parameters
    x = np.linspace(-5, 5, 100)
    A, b, mu, sigma = -1.0, 0.5, 0.0, 1.0
    y = pdm.gaussian([A, b, mu, sigma], x)

    # Add some noise
    np.random.seed(42)
    y_noisy = y + np.random.randn(len(x))*0.05

    # Fit a Gaussian to the noisy data
    from scipy.optimize import minimize
    result = minimize(pdm.nll, [A, b, mu, sigma], args=(x, y_noisy))

    # The fitted parameters should be close to the true parameters
    fitted_A, fitted_b, fitted_mu, fitted_sigma = result.x
    assert np.isclose(fitted_A, A, atol=0.2)
    assert np.isclose(fitted_b, b, atol=0.2)
    assert np.isclose(fitted_mu, mu, atol=0.2)
    assert np.isclose(fitted_sigma, sigma, atol=0.2)


if __name__ == "__main__":
    test_sj2()
    test_s2()
    test_calc_phase()
    # The following tests require pytest fixtures
    # test_phase_bins()
    # test_phi()
    # test_uncertainty()
    # test_pdm_different_periods()
    test_gaussian_fit()
