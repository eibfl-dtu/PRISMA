# © Copyright 2021, PRISMA’s Authors
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional
from scipy.optimize import curve_fit
import prisma.util.lineshapes

"""Templates of Spectrum object"""


class Spectrum:
    object_identifiers = {"Class": "Spectrum", "BattInfo ID": "00X0X0"}

    def __init__(
        self,
        indexes: np.ndarray,
        counts: np.ndarray,
        baseline: Optional[np.ndarray] = None,
        profiles: Optional[dict] = None,
        peaks: Optional[dict] = None,
        **kwargs
    ):
        self.indexes = indexes
        self.RESOLVABLE_WIDTH_FACTOR = 3  # minimum resolvable width = factor *
        # minimum difference between datapoint indexes
        self.counts = counts
        self.baseline = baseline
        self.profiles = profiles
        self.peaks = peaks
        self.__load_metadata(**kwargs)

    def __load_metadata(self, **kwargs):

        self.metadata = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self.metadata.update(value)
            else:
                self.metadata.update({key: value})

    def trimming(self, spectrum, within):
        """Trim raw spectrum
        * within [float,float]: lower and upper limits
        of the range to be studied
        """
        new_metadata = {
            "Process": "Trimming",
            "Process ID": self.object_identifiers["BattInfo ID"],
        }

        idxs_within = np.where(
            (spectrum.indexes > within[0]) &
            (spectrum.indexes < within[1]),
            True,
            False
        )  # nparray of booleans

        # trimming interval outside spectrum.indexes
        if np.all(~idxs_within):
            new_indexes = spectrum.indexes
            new_counts = spectrum.counts

        else:
            new_indexes = spectrum.indexes[idxs_within]
            new_counts = spectrum.counts[idxs_within]
            new_metadata["Trim interval"] = within

        new_metadata["Trim interval"] = [
            min(within[0], np.amin(spectrum.indexes)),
            max(within[1], np.amax(spectrum.indexes)),
        ]

        return Spectrum(
            indexes=new_indexes,
            counts=new_counts,
            metadata=new_metadata
        )

    def downsample(self, spectrum, downsampling_factor: int):

        if downsampling_factor > 1:

            samples_decimated = int(len(spectrum.counts) / downsampling_factor)
            min_index: float = min(spectrum.indexes)
            max_index: float = max(spectrum.indexes)
            new_counts = sp.signal.decimate(
                spectrum.counts, downsampling_factor
            )
            new_indexes = np.linspace(
                min_index, max_index, samples_decimated, endpoint=False
            )

            return Spectrum(indexes=new_indexes, counts=new_counts)

        else:
            return spectrum

    def reject_outliers(self, spectrum, outliers_threshold=0.0):

        if outliers_threshold > 0.0:

            differential_counts = np.abs(
                np.diff(
                    spectrum.counts,
                    n=2,
                    prepend=spectrum.counts[0],
                    append=spectrum.counts[-1],
                )
            )
            q1, q3 = np.percentile(differential_counts, [25, 75])
            iqr = q3 - q1

            outliers_idxs = np.where(
                (differential_counts < q1 - outliers_threshold * iqr)
                | (differential_counts > q3 + outliers_threshold * iqr)
            )[0]

            if outliers_idxs.size == 0:  # if there are no outliers
                return spectrum
            else:
                outlier_groups = np.split(
                    outliers_idxs, np.where(np.diff(outliers_idxs) > 1)[0] + 1
                )  # neighboring points are also classified as outliers.
                # This groups an outlier and its neighbors
                outliers_idxs_no_neighbors = [
                    group[np.argmax(differential_counts[group])]
                    for group in outlier_groups
                ]  # This select the outlier as the maximum value
                # among its neighbors

                new_counts = spectrum.counts.copy()
                new_counts[outliers_idxs_no_neighbors] = np.nan
                new_indexes = spectrum.indexes

            return Spectrum(indexes=new_indexes, counts=new_counts)

        else:
            return spectrum

    def asymmetric_least_squares(self, spectrum, log_p=-1.5, log_lambda=7):
        PROCESS_TYPE = "Asymmetric Least Squares"
        BATTINFO_ID = self.object_identifiers["BattInfo ID"]
        new_metadata = {
            "Process": PROCESS_TYPE,
            "Process ID": BATTINFO_ID,
            "Method": "Assymetric Least Squares",
            "Log10(p)": log_p,
            "Log10(lambda)": log_lambda,
        }

        param_p, param_lambda = 10**log_p, 10**log_lambda

        nan_counts_mask = np.isnan(spectrum.counts)
        if np.any(nan_counts_mask):  # interpolation of nans

            non_nan_counts = np.interp(
                np.arange(len(spectrum.counts)),
                np.arange(len(spectrum.counts))[~nan_counts_mask],
                spectrum.counts[~nan_counts_mask],
            )
        else:
            non_nan_counts = spectrum.counts

        m = len(non_nan_counts)
        D = sparse.diags(
            [1, -2, 1], [0, -1, -2], shape=(m, m - 2)
        )  # sparse representation of difference_2 matrix
        w = np.ones(m)  # initial -symmetric- weights
        W = sparse.spdiags(w, 0, m, m)  # weight matrix with initial weights
        iterations = 20

        for _ in range(iterations):

            W.setdiag(w)  # wiegth matrix is updated with newest weights
            C = W + param_lambda * D.dot(
                D.transpose()
            )  # matrix summarizing the fit and smooth penalties
            z = spsolve(C, w * non_nan_counts)
            updated_w = param_p * (non_nan_counts > z) + (1 - param_p) * (
                non_nan_counts < z
            )

            if np.linalg.norm(w) == np.linalg.norm(updated_w):
                break
            else:
                w = updated_w

        new_counts = non_nan_counts - z
        new_counts[nan_counts_mask] = np.nan
        new_indexes = spectrum.indexes

        return Spectrum(
            indexes=new_indexes,
            counts=new_counts,
            baseline=z,
            metadata=new_metadata
        )
    # *****************************HELPER FUNCTIONS************************

    def prisma_peak_defaults(self, peak_bounds, max_widths, spectrum):
        # Format peak_bounds and peak_widhts to parameter bounds for the curve
        # fit function
        # init_guess   --> initial guesses for the fitting parameters:
        # [y0, h1, p1, w1, h2, p2, w2, h3, p3, w3, ...]
        # initial guesses for the fitting parameters
        # param_bounds --> 2-tuple of lists with lower and upper bounds for
        # the fitting parameters: ([y0,h1,p1,w1,...],[y0,h1,p1,w1,...])
        overall_max_counts = np.amax(spectrum.counts)

        limit_resolvable_width = self.RESOLVABLE_WIDTH_FACTOR * np.abs(
            spectrum.indexes[1] - spectrum.indexes[0]
        )

        # Bounds for y0
        init_guess = [0]
        param_bounds_low = [-0.1 * overall_max_counts]
        param_bounds_high = [0.1 * overall_max_counts]

        # Bounds for all other parameters
        for width, bound in zip(max_widths, peak_bounds):

            max_counts_within_bounds = np.amax(
                spectrum.counts[
                    (spectrum.indexes > bound[0]) &
                    (spectrum.indexes < bound[1])
                ]
            )

            # guess height = 30% maximum height
            # guess position: halfway between bounds
            # guess width: half the maximum width provided or 5% more of
            # min_resolvable_width, whoever is greater
            init_guess += [
                0.3 * max_counts_within_bounds,
                0.5 * (bound[1] - bound[0]) + bound[0],
                max(1.05 * limit_resolvable_width, width / 2),
            ]

            # lower bound height = 0 | lower bound position: the one provided |
            # lower bound width: minimum resolvable width
            param_bounds_low += [0, bound[0], limit_resolvable_width]

            # upper bound height = 110% max height | upper bound position:
            # the one provided | upper bound width: the one provided or 10%
            # more of min_resolvable_width, whoever is greater
            param_bounds_high += [
                1.1 * max_counts_within_bounds,
                bound[1],
                max(1.1 * limit_resolvable_width, width),
            ]

        return init_guess, (param_bounds_low, param_bounds_high)

    def get_fitting_functions(self, lineshape, number_of_peaks):
        if lineshape == 'Lorentzian':
            fitting_function = prisma.util.lineshapes.lorentzians(
                number_of_peaks
            )
            single_peak_function = prisma.util.lineshapes.lorentzians(1)

        elif lineshape == 'Gaussian':
            fitting_function = prisma.util.lineshapes.gaussians(
                number_of_peaks
            )
            single_peak_function = prisma.util.lineshapes.gaussians(1)

        elif lineshape == 'Pseudo-Voight 50% Lorentzian':
            fitting_function = prisma.util.lineshapes.pseudo_voight_50(
                number_of_peaks
            )
            single_peak_function = prisma.util.lineshapes.pseudo_voight_50(1)

        return fitting_function, single_peak_function

    # ***************************FITTING FUNCTION***************************

    def fit_peaks(
        self, spectrum, peak_bounds, guess_widths, lineshape_peak="Lorentzian"
    ):
        """Fits peaks in the spectrum with a lorentzian profile. Parameters:
        * peak_bounds: [(low1,high1),(low2,high2),(low3,high3),...]
          list of 2-tuples with lower and upper bounds for the peak positions
        * guess_widths: [w1,w2,w3,...] initial guesses for the peak widths
        """

        new_metadata = {
            "Process": "Peak fitting",
            "Process ID": self.object_identifiers["BattInfo ID"],
            "Peak lineshapes": lineshape_peak,
            "Number of peaks": len(guess_widths),
            "Initial widths": guess_widths,
            "Position bounds": peak_bounds,
            "Fitting success": False,
        }
        new_indexes = spectrum.indexes

        # formatting bounds and define fitting functions with helper functions
        init_guess, param_bounds = self.prisma_peak_defaults(
            peak_bounds, guess_widths, spectrum
        )
        fitting_function, single_peak_function = self.get_fitting_functions(
            lineshape_peak,
            new_metadata["Number of peaks"]
        )

        # fitting
        try:
            fitted_coeffs, _ = curve_fit(
                fitting_function,
                spectrum.indexes,
                spectrum.counts,
                p0=init_guess,
                bounds=param_bounds,
                ftol=1e-6,
                xtol=1e-6,
            )

            # store peaks and peak sum
            new_profiles = {
                peak_n: single_peak_function(
                    spectrum.indexes,
                    *(
                        np.append(
                            fitted_coeffs[0],
                            fitted_coeffs[3 * peak_n + 1: 3 * peak_n + 4]
                        )
                    )
                )
                for peak_n in range(new_metadata["Number of peaks"])
            }
            new_counts = fitting_function(
                spectrum.indexes, *fitted_coeffs
            )  # evaluate wavenumbers with the fitted coefficients

            new_metadata["Fitting success"] = True

        except RuntimeError:
            nan_vector = np.full(len(new_indexes), np.nan)
            fitted_coeffs = np.full(
                3 * new_metadata["Number of peaks"] + 1, np.nan
            )
            new_profiles = {
                peak_n: nan_vector
                for peak_n in range(new_metadata["Number of peaks"])
            }
            new_counts = nan_vector
            new_metadata["Fitting success"] = False

        # store fitting parameters
        new_metadata["Fitted parameters"] = {"y_0": fitted_coeffs[0]}
        for peak_n in range(new_metadata["Number of peaks"]):
            new_metadata["Fitted parameters"].update(
                {
                    "h_{}".format(peak_n + 1): fitted_coeffs[3 * peak_n + 1],
                    "p_{}".format(peak_n + 1): fitted_coeffs[3 * peak_n + 2],
                    "w_{}".format(peak_n + 1): fitted_coeffs[3 * peak_n + 3],
                }
            )

        return Spectrum(
            indexes=new_indexes,
            counts=new_counts,
            parent=spectrum,
            profiles=new_profiles,
            metadata=new_metadata,
        )

    @property
    def class_id(self):
        return (
            self.__class__.object_identifiers
        )  # return the class variable of the class instantiating the object


if __name__ == "__main__":
    raw_spectrum = Spectrum(np.ndarray(46), np.ndarray(12), wavelnght=45)
    # print(dir(test_object))
    # print(type(test_object.__str__()))
