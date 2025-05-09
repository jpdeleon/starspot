"""
A script for measuring the rotation periods of a set of stars.
"""

import numpy as np
from .rotation_tools import simple_acf, get_peak_statistics
import exoplanet as xo
import pymc3 as pm
import theano.tensor as tt

# Import the GP modules directly from celerite2
try:
    import celerite2.theano as ce
    from celerite2.theano import terms as ceterms
except ImportError:
    # Fallback to older exoplanet structure if available
    try:
        from exoplanet.gp import terms
        from exoplanet.gp.celerite import GP
    except ImportError:
        print("Error: Could not import required GP modules from either celerite2 or exoplanet.")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import astropy.timeseries as apt
from .phase_dispersion_minimization import phi, calc_phase, phase_bins, \
    estimate_uncertainty, gaussian
from tqdm import tqdm, trange

plotpar = {'axes.labelsize': 25,
           'xtick.labelsize': 20,
           'ytick.labelsize': 20,
           'text.usetex': True}
plt.rcParams.update(plotpar)


class RotationModel(object):
    """
    Code for measuring stellar rotation periods.

    Args:
        time (array): The time array in days.
        flux (array): The flux array.
        flux_err (array): The array of flux uncertainties.
    """

    def __init__(self, time, flux, flux_err):
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.Rvar = np.percentile(flux, 95) - np.percentile(flux, 5)

    # def calc_Rvar(self):
    #     Rvar = np.percentile(self.flux, 95) - np.percentile(self.flux, 5)
    #     self.Rvar = Rvar
    #     return Rvar

    def lc_plot(self):
        """
        Plot the light curve.
        """
        plt.figure(figsize=(20, 5))
        plt.plot(self.time, self.flux, "k.", ms=.5);
        plt.xlabel("Time [days]")
        plt.ylabel("Relative Flux");
        plt.subplots_adjust(bottom=.2)

    def ls_rotation(self, high_pass=False, min_period=.5, max_period=50.,
                    samples_per_peak=50, input_freq=None, input_power=None,
                    input_ls_period=None):
        # filter_period=None, order=3,
        """
        Measure a rotation period using a Lomb-Scargle periodogram.

        Args:
            min_period (Optional[float]): The minimum rotation period you'd
                like to search for. The default is 0.5 days since most stars
                rotate more slowly than this.
            max_period (Optional[float]): The maximum rotation period you'd
                like to search for. For Kepler this could be as high as 70
                days but for K2 it should probably be more like 20-25 days
                and 10-15 days for TESS. Default is 50.
            samples_per_peak (Optional[int]): The number of samples per peak.

        Returns:
            ls_period (float): The Lomb-Scargle rotation period.

        """
        if input_freq is not None and input_power is not None:
            self.freq = input_freq
            self.power = input_power
        else:
            self.freq = np.linspace(1./max_period, 1./min_period, 100000)

        if input_ls_period is not None:
            self.ls_period = input_ls_period
            return input_ls_period

        assert len(self.flux) == sum(np.isfinite(self.flux)), "Remove NaNs" \
            " from your flux array before trying to compute a periodogram."

        self.power = apt.LombScargle(
            self.time, self.flux).power(self.freq)

        ps = 1./self.freq
        peaks = np.array([i for i in range(1, len(ps)-1) if self.power[i-1] <
                          self.power[i] and self.power[i+1] < self.power[i]])

        if len(peaks) == 0:
            self.ls_period = 0
        else:
            self.ls_period = ps[self.power == max(self.power[peaks])][0]
        return self.ls_period


    def ls_plot(self):
        """
        Make a plot of the periodogram.

        """

        fig = plt.figure(figsize=(16, 9))
        plt.plot(-np.log10(self.freq), self.power, "k", zorder=0)
        plt.axvline(np.log10(self.ls_period), color="C1", lw=4, alpha=0.5,
                    zorder=1)
        plt.xlim((-np.log10(self.freq)).min(), (-np.log10(self.freq)).max())
        plt.yticks([])
        plt.xlabel("log10(Period [days])")
        plt.ylabel("Power");
        plt.subplots_adjust(left=.15, bottom=.15)
        return fig

    def acf_rotation(self, interval, smooth=9, cutoff=0, window_length=99,
                     polyorder=3):
        """
        Calculate a rotation period based on an autocorrelation function.

        Args:
            interval (float): The time in days between observations. For
                Kepler/K2 long cadence this is 0.02043365, for Tess its about
                0.00138889 days. Use interval = "TESS" or "Kepler" for these.
            smooth (Optional[float]): The smoothing window in days.
            cutoff (Optional[float]): The number of days to cut off at the
                beginning.
            window_length (Optional[float]): The filter window length.
            polyorder (Optional[float]): The polynomial order of the filter.

        Returns:
            acf_period (float): The ACF rotation period in days.

        """
        if interval == "TESS":
            interval = 0.00138889
        if interval == "Kepler":
            interval = 0.02043365

        lags, acf, _x, _y = simple_acf(self.time, self.flux, interval,
                                       smooth=smooth,
                                       window_length=window_length,
                                       polyorder=polyorder)

        self.acf_x = _x
        self.acf_y = _y

        # find all the peaks
        m = lags > cutoff
        xpeaks, ypeaks = get_peak_statistics(lags[m], acf[m],
                                             sort_by="height")

        self.lags = lags[m]
        self.acf = acf[m]
        self.acf_period = xpeaks[0]
        return xpeaks[0]

    def acf_plot(self):
        """
        Make a plot of the autocorrelation function.

        """
        fig = plt.figure(figsize=(16, 9))
        plt.plot(self.lags, self.acf, "k")
        plt.axvline(self.acf_period, color="C1")
        plt.xlabel("Period [days]")
        plt.ylabel("Correlation")
        plt.xlim(0, max(self.lags))
        plt.subplots_adjust(left=.15, bottom=.15)
        return fig

    def pdm_rotation(self, period_grid, pdm_nbins=10):
        """
        Calculate the optimum period from phase dispersion minimization.

        Args:
            period_grid (array): The period grid.
            pdm_nbins (array): The number of bins to use when calculating phase
                dispersion.

        Returns:
            phis (array): The array of phi statistics

        """
        self.pdm_nbins = pdm_nbins
        self.period_grid = period_grid

        nperiods = len(period_grid)
        phis = np.zeros(nperiods)
        # for i, p in enumerate(period_grid):
        for i in trange(len(period_grid)):
            phis[i] = phi(pdm_nbins, period_grid[i], self.time, self.flux)

        self.phis = phis

        # Find period with the lowest Phi
        ind = np.argmin(phis)
        self.pdm_period = period_grid[ind]
        if hasattr(self.pdm_period, 'len'):
            self.pdm_period = self.pdm_period[0]

        # Estimate the uncertainty
        err, mu, a, b = estimate_uncertainty(period_grid, phis,
                                               period_grid[ind])
        self.sigma = err
        self.mu = mu
        self.a = a
        self.b = b
        self.period_err = err

        # Calculate phase (for plotting)
        phase = calc_phase(self.pdm_period, self.time)
        self.phase = phase

        return self.pdm_period, err

    def pdm_plot(self):
        """
        Make a plot of the phase dispersion function.

        """
        # Calculate phase, etc.
        x_means, phase_bs, Ns, sj2s, xb, pb = \
            phase_bins(self.pdm_nbins, self.phase, self.flux)
        mid_phase_bins = np.diff(phase_bs)*.5 + phase_bs[:-1]

        fig = plt.figure(figsize=(16, 9), dpi=200)
        ax1 = fig.add_subplot(311)
        ax1.plot(self.time, self.flux, "k.", ms=1, alpha=.5)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Flux")

        ax2 = fig.add_subplot(312)
        ax2.plot(self.phase, self.flux, "k.", alpha=.1)
        # ax2.errorbar(mid_phase_bins, x_means, yerr=sj2s, fmt=".")
        ax2.set_xlabel("Phase")
        ax2.set_ylabel("Flux")

        ax3 = fig.add_subplot(313)
        ax3.plot(self.period_grid, gaussian([self.a, self.b, self.mu,
                                             self.sigma], self.period_grid))
        ax3.plot(self.period_grid, self.phis, "k")
        ax3.set_xlabel("Period [days]")
        ax3.set_ylabel("Dispersion")
        ax3.axvline(self.pdm_period, color=".5", ls="--")
        ax3.axvline(self.mu, color="C0", ls="--")
        ax3.axvline(self.mu - self.sigma, ls="--", lw=.5)
        ax3.axvline(self.mu + self.sigma, ls="--", lw=.5)
        plt.tight_layout()
        return fig

    def big_plot(self, methods, xlim=None, method_xlim=(0, 50)):
        """
        Make a plot of LS periodogram, ACF and PDM, combined. These things
        must be precomputed.

        Args:
            methods (list): A list of period measurement methods to plot. For
                example, ["pdm", "ls", "acf"], or ["ls", "pdm"].
            xlim (Optional[tuple]): The xlim for the light curve panel.
            method_xlim (Optional[tuple]): The xlim for the methods panel.
                Default is 0-50 days.

        Returns:
            The figure object.

        """
        methods = np.array(methods)
        nmethods = len(methods)

        # Assemble indices selecting methods to plot.
        inds, i_s, names = np.arange(3), [], np.array(["pdm", "ls", "acf"])
        for i in range(nmethods):
            mask = methods[i] == names
            i_s.append(int(inds[mask]))
        i_s = np.array(i_s)

        outer = gridspec.GridSpec(3, nmethods,
                                  height_ratios=[1, 1, nmethods])

        # The light curve panel
        # --------------------------------------------------------------------
        gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer[0, :])

        fig = plt.figure(figsize=(16, 16), dpi=200)
        ax1 = fig.add_subplot(gs0[0, :])
        # ax1.plot(self.time, self.flux, "k", lw=.5, rasterized=True)
        # ax1.errorbar(self.time, self.flux, yerr=self.flux_err,
                     # fmt="k.", alpha=.1, rasterized=True)
        ax1.plot(self.time, self.flux, "k.", alpha=.3, mec="none",
                 rasterized=True)
        ax1.set_xlabel("$\mathrm{Time~[days]}$")
        ax1.set_ylabel("$\mathrm{Normalized~Flux}$")
        if xlim is not None:
            ax1.set_xlim(xlim)

        pdm_tit, ls_tit, acf_tit = "", "", ""
        if np.any(i_s == 0):
            pdm_tit = "PDM = {0:.2f} +/- {1:.2f} days.".format(self.pdm_period,
                                                              self.period_err)
            pdm_x, pdm_y, pdm_p = self.period_grid, self.phis, self.pdm_period
            err, pdm_phase =  self.period_err, self.phase
        else:
            pdm_x, pdm_y, pdm_p, err, pdm_phase = None, None, None, None, None
        if np.any(i_s == 1):
            ls_tit = " LS = {0:.2f} days.".format(self.ls_period)
            ls_x, ls_y, ls_p = 1./self.freq, self.power, self.ls_period
            ls_phase = calc_phase(self.ls_period, self.time)
        else:
            ls_x, ls_y, ls_p, ls_phase = None, None, None, None
        if np.any(i_s == 2):
            acf_tit = " ACF = {0:.2f} days.".format(self.acf_period)
            acf_x, acf_y, acf_p = self.lags, self.acf, self.acf_period
            acf_phase = calc_phase(self.acf_period, self.time)
        else:
            acf_x, acf_y, acf_p, acf_phase = None, None, None, None

        plt.title("{0}{1}{2}".format(pdm_tit, ls_tit, acf_tit), fontsize=20)

        # The phase-fold panel
        # --------------------------------------------------------------------
        gs1 = gridspec.GridSpecFromSubplotSpec(1, nmethods,
                                               subplot_spec=outer[1, :],
                                               wspace=0)

        xs = [pdm_phase, ls_phase, acf_phase]
        xlabels = ["$\mathrm{PDM~Phase}$", "$\mathrm{LS~Phase}$",
                   "$\mathrm{ACF~Phase}$"]

        colors = ["k", "C0", "C1"]
        def phase_subplot(x, y, i, xlabel):
            ax = fig.add_subplot(gs1[0, i])
            ax.plot(x, y, ".", color=colors[i], alpha=.1, rasterized=True)
            ax.set_xlabel(xlabel)
            ax.set_xlim(0, 1)
            return ax

        # Plot the panels
        axs = []
        for j in range(nmethods):
            ax = phase_subplot(xs[i_s[j]], self.flux, j, xlabels[i_s[j]])
            axs.append(ax)
            if j > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
        ax0 = axs[0]
        ax0.set_ylabel("$\mathrm{Normalized~Flux}$")

        # The method panel
        # --------------------------------------------------------------------
        gs2 = gridspec.GridSpecFromSubplotSpec(nmethods, 1,
                                               subplot_spec=outer[2, :],
                                               hspace=0)

        mxs = [pdm_x, ls_x, acf_x]
        mys = [pdm_y, ls_y, acf_y]
        ps = [pdm_p, ls_p, acf_p]
        ylabels = ["$\mathrm{Relative~Dispersion}$", "$\mathrm{LS-Power}$",
                   "$\mathrm{Autocorrelation}$"]

        def method_subplot(i, x, y, p, ylabel, sharex):
            if i == 0:
                ax = fig.add_subplot(gs2[i, :])
            else:
                ax = fig.add_subplot(gs2[i, :], sharex=sharex)
            ax.plot(x, y, "k", rasterized=True)
            ax.axvline(p)
            ax.axvline(p/2., ls="--")
            ax.axvline(p*2., ls="--")
            ax.set_ylabel(ylabel)
            if method_xlim is not None:
                ax.set_xlim(method_xlim)
            return ax

        maxs = []
        for j in range(nmethods):
            sharex = ax
            ax = method_subplot(j, mxs[i_s[j]], mys[i_s[j]], ps[i_s[j]],
                                ylabels[i_s[j]], sharex)
            maxs.append(ax)
            if nmethods > 1 and j < nmethods-1:
                plt.setp(ax.get_xticklabels(), visible=False)


        # Plot a Gaussian on top of PDM plot
        axloc_ind = np.arange(len(maxs))[np.array(i_s) == 0]
        if np.any(i_s == 0):
            ax3 = maxs[int(axloc_ind)]
            ax3.plot(self.period_grid, gaussian([self.a, self.b, self.mu,
                                             self.sigma], self.period_grid),
                     rasterized=True)

        ax5 = maxs[-1]
        ax5.set_xlabel("$\mathrm{Time~[days]}$")
        plt.subplots_adjust(hspace=.1)
        plt.tight_layout()
        return fig

    def gp_rotation(self, init_period=None, tune=2000, draws=2000,
                    prediction=True, cores=None):
        """
        Calculate a rotation period using a Gaussian process method.

        Args:
            init_period (Optional[float]): Your initial guess for the rotation
                period. The default is the Lomb-Scargle period.
            tune (Optional[int]): The number of tuning samples. Default is
                2000.
            draws (Optional[int]): The number of samples. Default is 2000.
            prediction (Optional[Bool]): If true, a prediction will be
                calculated for each sample. This is useful for plotting the
                prediction but will slow down the whole calculation.
            cores (Optional[int]): The number of cores to use. Default is
                None (for running one process).

        Returns:
            gp_period (float): The GP rotation period in days.
            errp (float): The upper uncertainty on the rotation period.
            errm (float): The lower uncertainty on the rotation period.
            logQ (float): The Q factor.
            Qerrp (float): The upper uncertainty on the Q factor.
            Qerrm (float): The lower uncertainty on the Q factor.
        """
        self.prediction = prediction

        x = np.array(self.time, dtype=float)
        # Median of data must be zero
        y = np.array(self.flux, dtype=float) - np.median(self.flux)
        yerr = np.array(self.flux_err, dtype=float)

        if init_period is None:
            # Calculate ls period
            init_period = self.ls_rotation()

        with pm.Model() as model:

            # The mean flux of the time series
            mean = pm.Normal("mean", mu=0.0, sd=10.0)

            # A jitter term describing excess white noise
            logs2 = pm.Normal("logs2", mu=2*np.log(np.min(yerr)), sd=5.0)

            # The parameters of the RotationTerm kernel
            logamp = pm.Normal("logamp", mu=np.log(np.var(y)), sd=5.0)
            logperiod = pm.Normal("logperiod", mu=np.log(init_period),
                                  sd=5.0)
            logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
            logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
            mix = pm.Uniform("mix", lower=0, upper=1.0)

            # Track the period as a deterministic
            period = pm.Deterministic("period", tt.exp(logperiod))

            # Set up the Gaussian Process model
            # Try to use the appropriate GP implementation based on what's available
            try:
                # Try using celerite2 first (newer versions of exoplanet)
                kernel = ceterms.RotationTerm(
                    sigma=tt.exp(0.5*logamp),
                    period=period,
                    Q0=tt.exp(logQ0),
                    dQ=tt.exp(logdeltaQ),
                    f=mix
                )
                # In celerite2, J is a parameter of the kernel, not the GP
                kernel.J = 4
                # Initialize the GP without the J parameter
                gp = ce.GaussianProcess(kernel, t=x, diag=yerr**2 + tt.exp(logs2))
                # No need to call compute again as it's already done in the constructor
            except (NameError, AttributeError):
                try:
                    # Fall back to older exoplanet structure
                    kernel = terms.RotationTerm(
                        log_amp=logamp,
                        period=period,
                        log_Q0=logQ0,
                        log_deltaQ=logdeltaQ,
                        mix=mix
                    )
                    gp = GP(kernel, x, yerr**2 + tt.exp(logs2), J=4)
                except (NameError, AttributeError):
                    raise ImportError("Could not initialize GP model. Please check your exoplanet installation.")

            # Compute the Gaussian Process likelihood and add it into the
            # the PyMC3 model as a "potential"
            try:
                # For celerite2
                gp.condition(y - mean)
                pm.Potential("loglike", gp.log_likelihood())
            except (TypeError, AttributeError):
                try:
                    # Alternative celerite2 API
                    pm.Potential("loglike", gp.log_likelihood(y - mean))
                except (TypeError, AttributeError):
                    # For older exoplanet
                    pm.Potential("loglike", gp.log_likelihood(y - mean))

            # Compute the mean model prediction for plotting purposes
            if prediction:
                try:
                    # For celerite2
                    # We've already conditioned the GP in the log_likelihood step
                    pm.Deterministic("pred", gp.predict())
                except (TypeError, AttributeError):
                    try:
                        # Alternative celerite2 API
                        pm.Deterministic("pred", gp.predict(y - mean))
                    except (TypeError, AttributeError):
                        # For older exoplanet
                        pm.Deterministic("pred", gp.predict())

            # Optimize to find the maximum a posteriori parameters
            try:
                # Try direct PyMC3 find_MAP first (most compatible)
                self.map_soln = pm.find_MAP(model=model)
            except Exception as e:
                print(f"find_MAP failed: {e}")
                try:
                    # For newer exoplanet versions
                    import pymc3_ext
                    self.map_soln = pymc3_ext.optimize(start=model.test_point)
                except ImportError:
                    try:
                        # For older exoplanet versions
                        self.map_soln = xo.optimize(start=model.test_point)
                    except Exception as e:
                        print(f"Optimization failed: {e}")
                        # If all else fails, just use the test point
                        self.map_soln = model.test_point

            # Sample from the posterior
            np.random.seed(42)
            with model:
                print("sampling...")
                try:
                    # Try with newer PyMC3 versions that support return_inferencedata
                    trace = pm.sample(
                        tune=tune,
                        draws=draws,
                        start=self.map_soln,
                        cores=cores,
                        return_inferencedata=False,
                        target_accept=0.9  # Pass directly to sample
                    )
                except TypeError:
                    # For older PyMC3 versions
                    try:
                        # Try with step object explicitly created
                        step = pm.NUTS(target_accept=0.9)
                        trace = pm.sample(
                            tune=tune,
                            draws=draws,
                            start=self.map_soln,
                            cores=cores,
                            step=step
                        )
                    except Exception as e:
                        print(f"Sampling error: {e}")
                        # Last resort - simplest possible sampling call
                        trace = pm.sample(
                            tune=tune,
                            draws=draws,
                            start=self.map_soln,
                            cores=cores
                        )

            # Save samples
            samples = pm.trace_to_dataframe(trace)
            self.samples = samples

            self.period_samples = trace["period"]
            self.gp_period = np.median(self.period_samples)
            lower = np.percentile(self.period_samples, 16)
            upper = np.percentile(self.period_samples, 84)
            self.errm = self.gp_period - lower
            self.errp = upper - self.gp_period
            self.logQ = np.median(trace["logQ0"])
            upperQ = np.percentile(trace["logQ0"], 84)
            lowerQ = np.percentile(trace["logQ0"], 16)
            self.Qerrp = upperQ - self.logQ
            self.Qerrm = self.logQ - lowerQ

        self.trace = trace

        return self.gp_period, self.errp, self.errm, self.logQ, self.Qerrp, \
            self.Qerrm

    def plot_prediction(self):
        """
        Plot the GP prediction, fit to the data.

        """
        if not self.prediction:
            print("You must run GP_rotate with prediction=True in order" \
                    " to plot the prediction.")
            return

        plt.figure(figsize=(20, 5))
        plt.plot(self.time, self.flux-np.median(self.flux), "k.", ms=2,
                 label="data")
        plt.plot(self.time, np.median(self.trace["pred"], axis=0),
                    color="C1", lw=2, label="model")
        plt.xlabel("Time [days]")
        plt.ylabel("Relative flux")
        plt.legend(fontsize=20)
        self.prediction = np.median(self.trace["pred"], axis=0)

    def plot_posterior(self, nbins=30):
        """
        Plot the posterior probability density function for rotation period.

        Args:
            nbins (Optional[int]): The number of histogram bins. Default is 30
            cutoff (Optional[float]): The maximum sample value to plot.
        """
        plt.hist(self.period_samples, nbins, histtype="step", color="k")
        plt.axvline(self.gp_period)
        plt.yticks([])
        plt.xlabel("Rotation period [days]")
        plt.ylabel("Posterior density");
        plt.axvline(self.gp_period - self.errm, ls="--", color="C1");
        plt.axvline(self.gp_period + self.errp, ls="--", color="C1");
