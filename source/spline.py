'''
Copied and modified from: 
[1] https://github.com/futakw/DensityAwareCalibration - spline.py
[2] https://github.com/kartikgupta-at-anu/spline-calibration - recalibrate.py

'''

import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import pickle

from scipy.special import softmax



def get_spline_calib_func(logits, labels, n=-1, spline_method="natural", splines=6):
    """
    logits: outputs from neural networks *before softmax*
    labels: one-hot labels
    """
    # to run Spline calibration from:
    # https://github.com/kartikgupta-at-anu/spline-calibration
    y_probs_val, y_val = softmax(logits, 1), np.argmax(labels, axis=1)

    scores1, labels1, scores1_class = get_top_results(
        y_probs_val, y_val, n, return_topn_classid=True
    )
    # Get recalibration function, based on scores1
    frecal = get_recalibration_function(scores1, labels1, spline_method, splines)
    # calibrate
    scores1 = np.array([frecal(float(sc)) for sc in scores1])
    scores1[scores1 < 0.0] = 0.0
    scores1[scores1 > 1.0] = 1.0
    p = scores1
    label = labels1  # accuracy
    return frecal, p, label


def spline_calibrate(
    spline_calib_func,
    logits,
    labels,
    n=-1,
):
    # to run Spline calibration from:
    # https://github.com/kartikgupta-at-anu/spline-calibration
    y_probs_test = softmax(logits, 1)
    y_test = np.argmax(labels, axis=1)
    scores2, labels2, scores2_class = get_top_results(
        y_probs_test, y_test, n, return_topn_classid=True
    )
    scores2 = np.array([spline_calib_func(float(sc)) for sc in scores2])
    scores2[scores2 < 0.0] = 0.0
    scores2[scores2 > 1.0] = 1.0
    return scores2, labels2, scores2_class


# copied from utililies/utils
def len0(x):
    # Proper len function that REALLY works.
    # It gives the number of indices in first dimension

    # Lists and tuples
    if isinstance(x, list):
        return len(x)

    if isinstance(x, tuple):
        return len(x)

    # Numpy array
    if isinstance(x, np.ndarray):
        return x.shape[0]

    # Other numpy objects have length zero
    if is_numpy_object(x):
        return 0

    # Unindexable objects have length 0
    if x is None:
        return 0
    if isinstance(x, int):
        return 0
    if isinstance(x, float):
        return 0

    # Do not count strings
    if type(x) == type("a"):
        return 0

    return 0


# copied from utilities/spline
class Spline:
    # Initializer
    def __init__(self, x, y, kx, runout="parabolic"):
        # This calculates and initializes the spline

        # Store the values of the knot points
        self.kx = kx
        self.delta = kx[1] - kx[0]
        self.nknots = len(kx)
        self.runout = runout

        # Now, compute the other matrices
        m_from_ky = self.ky_to_M()  # Computes second derivatives from knots
        my_from_ky = np.concatenate([m_from_ky, np.eye(len(kx))], axis=0)
        y_from_my = self.my_to_y(x)
        y_from_ky = y_from_my @ my_from_ky

        # print (f"\nmain:"
        #      f"\ny_from_my  = \n{utils.str(y_from_my)}"
        #      f"\nm_from_ky = \n{utils.str(m_from_ky)}"
        #      f"\nmy_from_ky = \n{utils.str(my_from_ky)}"
        #      f"\ny_from_ky = \n{utils.str(y_from_ky)}"
        #     )

        # Now find the least squares solution
        ky = np.linalg.lstsq(y_from_ky, y, rcond=-1)[0]

        # Return my
        self.ky = ky
        self.my = my_from_ky @ ky

    def my_to_y(self, vecx):
        # Makes a matrix that computes y from M
        # The matrix will have one row for each value of x

        # Make matrices of the right size
        ndata = len(vecx)
        nknots = self.nknots
        delta = self.delta

        mM = np.zeros((ndata, nknots))
        my = np.zeros((ndata, nknots))

        for i, xx in enumerate(vecx):
            # First work out which knots it falls between
            j = int(np.floor((xx - self.kx[0]) / delta))
            if j >= self.nknots - 1:
                j = self.nknots - 2
            if j < 0:
                j = 0
            x = xx - j * delta

            # Fill in the values in the matrices
            mM[i, j] = -(x ** 3) / (6.0 * delta) + x ** 2 / 2.0 - 2.0 * delta * x / 6.0
            mM[i, j + 1] = x ** 3 / (6.0 * delta) - delta * x / 6.0
            my[i, j] = -x / delta + 1.0
            my[i, j + 1] = x / delta

        # Now, put them together
        M = np.concatenate([mM, my], axis=1)

        return M

    # -------------------------------------------------------------------------------

    def my_to_dy(self, vecx):
        # Makes a matrix that computes y from M for a sequence of values x
        # The matrix will have one row for each value of x in vecx
        # Knots are at evenly spaced positions kx

        # Make matrices of the right size
        ndata = len(vecx)
        h = self.delta

        mM = np.zeros((ndata, self.nknots))
        my = np.zeros((ndata, self.nknots))

        for i, xx in enumerate(vecx):
            # First work out which knots it falls between
            j = int(np.floor((xx - self.kx[0]) / h))
            if j >= self.nknots - 1:
                j = self.nknots - 2
            if j < 0:
                j = 0
            x = xx - j * h

            mM[i, j] = -(x ** 2) / (2.0 * h) + x - 2.0 * h / 6.0
            mM[i, j + 1] = x ** 2 / (2.0 * h) - h / 6.0
            my[i, j] = -1.0 / h
            my[i, j + 1] = 1.0 / h

        # Now, put them together
        M = np.concatenate([mM, my], axis=1)

        return M

    # -------------------------------------------------------------------------------

    def ky_to_M(self):
        # Make a matrix that computes the
        A = 4.0 * np.eye(self.nknots - 2)
        b = np.zeros(self.nknots - 2)
        for i in range(1, self.nknots - 2):
            A[i - 1, i] = 1.0
            A[i, i - 1] = 1.0

        # For parabolic run-out spline
        if self.runout == "parabolic":
            A[0, 0] = 5.0
            A[-1, -1] = 5.0

        # For cubic run-out spline
        if self.runout == "cubic":
            A[0, 0] = 6.0
            A[0, 1] = 0.0
            A[-1, -1] = 6.0
            A[-1, -2] = 0.0

        # The goal
        delta = self.delta
        B = np.zeros((self.nknots - 2, self.nknots))
        for i in range(0, self.nknots - 2):
            B[i, i] = 1.0
            B[i, i + 1] = -2.0
            B[i, i + 2] = 1.0

        B = B * (6 / delta ** 2)

        # Now, solve
        Ainv = np.linalg.inv(A)
        AinvB = Ainv @ B

        # Now, add rows of zeros for M[0] and M[n-1]

        # This depends on the type of spline
        if self.runout == "natural":
            z0 = np.zeros((1, self.nknots))  # for natural spline
            z1 = np.zeros((1, self.nknots))  # for natural spline

        if self.runout == "parabolic":
            # For parabolic runout spline
            z0 = AinvB[0]
            z1 = AinvB[-1]

        if self.runout == "cubic":
            # For cubic runout spline

            # First and last two rows
            z0 = AinvB[0]
            z1 = AinvB[1]
            zm1 = AinvB[-1]
            zm2 = AinvB[-2]

            z0 = 2.0 * z0 - z1
            z1 = 2.0 * zm1 - zm2

        # print (f"ky_to_M:"
        #       f"\nz0 = {utils.str(z0)}"
        #       f"\nz1 = {utils.str(z1)}"
        #       f"\nAinvB = {utils.str(AinvB)}"
        #      )

        # Reshape to (1, n) matrices
        z0 = z0.reshape((1, -1))
        z1 = z1.reshape((1, -1))

        AinvB = np.concatenate([z0, AinvB, z1], axis=0)

        # print (f"\ncompute_spline: "
        #       f"\n A     = \n{utils.str(A)}"
        #       f"\n B     = \n{utils.str(B)}"
        #       f"\n Ainv  = \n{utils.str(Ainv)}"
        #       f"\n AinvB = \n{utils.str(AinvB)}"
        #      )

        return AinvB

    # -------------------------------------------------------------------------------

    def evaluate(self, x):
        # Evaluates the spline at a vector of values
        y = self.my_to_y(x) @ self.my
        return y

    # -------------------------------------------------------------------------------

    def evaluate_deriv(self, x):
        # Evaluates the spline at a vector (or single) point
        y = self.my_to_dy(x) @ self.my
        return y


# ===============================================================================


def main(argv):
    # Random seed
    utils.set_seed()

    # First, get the arguments
    argspec = {
        "+gc": utils.str_to_list_of_int,  # List of classes to plot
        "+d": utils.str_to_list_of_str,  # Data files
    }

    # Get the arguments
    argvals = utils.parseargs(argv, argspec)

    # Now, try a little test
    npoints = 100
    low = 0.0
    high = 1.0
    nknots = 7
    stdev = 0.05
    x = np.linspace(low, high, npoints)
    y = np.sin(7.0 * x + 0.8)
    # y = x + x**3
    y += np.random.normal(0.0, stdev, size=npoints)

    print(f"\nmain:" f"\nx = {utils.str(x)}" f"\ny = {utils.str(y)}")

    # Now, compute the spline
    kx = np.linspace(low, high, nknots)
    spline = Spline(x, y, kx, runout="parabolic")

    # Print the error
    yint = spline.evaluate(x)
    yd = spline.evaluate_deriv(x)
    err = yint - y
    rms = np.sqrt(np.mean(err * err))
    sumsq = np.sum(err * err)
    print(
        "main:"
        f"\ny    = {utils.str(y)}"
        f"\nyint = {utils.str(yint)}"
        f"\nerr  = {utils.str(err)}"
        f"\nrms  = {utils.str(rms)}"
        f"\nnknots = {nknots}, sumsq  = {utils.str(sumsq)}"
    )

    # Also, print out the values of the coefficients for each segment
    M = spline.my[0:nknots]
    delta = 1.0 / (nknots - 1)
    ky = spline.ky
    for i in range(0, nknots - 1):
        a = (M[i + 1] - M[i]) / (6.0 * delta)
        b = M[i] / 2.0
        c = (ky[i + 1] - ky[i]) / delta - delta * (M[i + 1] + 2.0 * M[i]) / 6.0
        d = ky[i]

        print(
            f"s[{i}] = "
            f"{utils.str(a)}*x^3 + "
            f"{utils.str(b)}*x^2 + "
            f"{utils.str(c)}*x + "
            f"{utils.str(d)}"
        )

    # Try plotting them
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.suptitle("Spline fit")
    ax.plot(x, yint, color="b")
    ax.plot(x, yd, color="g")
    ax.scatter(x, y, color="r")

    # Also, put the theoretical derivative
    # ax.plot (x, 1.0+3.0*x**2, color='r')
    ax.plot(x, 7.0 * np.cos(7.0 * x + 0.8), color="r")
    plt.show()


#########

# ==============================================================================


# Open file with pickled variables
def unpickle_probs(fname):
    # Read and open the file
    with open(fname, "rb") as f:
        (y_probs_val, y_val), (y_probs_test, y_test) = pickle.load(f)

    y_probs_val, y_probs_test = softmax(y_probs_val, 1), softmax(y_probs_test, 1)
    return ((y_probs_val, y_val), (y_probs_test, y_test))


# ------------------------------------------------------------------------------


def ensure_numpy(a):
    if not isinstance(a, np.ndarray):
        a = a.numpy()
    return a


# ------------------------------------------------------------------------------


class interpolated_function:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.lastindex = len0(self.x) - 1
        self.low = self.x[0]
        self.high = self.x[-1]

    def __call__(self, x):
        # Finds the interpolated value of the function at x

        # Easiest thing if value is out of range is to give maximum value
        if x >= self.x[-1]:
            return self.y[-1]
        if x <= self.x[0]:
            return self.y[0]

        # Find the first x above.  ind cannot be 0, because of previous test
        # ind cannot be > lastindex, because of last test
        ind = first_above(self.x, x)

        alpha = x - self.x[ind - 1]
        beta = self.x[ind] - x

        # Special case.  This occurs when two values of x are equal
        if alpha + beta == 0:
            return y[ind]

        return float((beta * self.y[ind] + alpha * self.y[ind - 1]) / (alpha + beta))


# ------------------------------------------------------------------------------


def get_recalibration_function(
    scores_in, labels_in, spline_method, splines, title=None
):
    # Find a function for recalibration

    # Change to numpy
    scores = ensure_numpy(scores_in)
    labels = ensure_numpy(labels_in)

    # Sort the data according to score
    order = scores.argsort()
    scores = scores[order]
    labels = labels[order]

    # Accumulate and normalize by dividing by num samples
    nsamples = len0(scores)
    integrated_accuracy = np.cumsum(labels) / nsamples
    integrated_scores = np.cumsum(scores) / nsamples
    percentile = np.linspace(0.0, 1.0, nsamples)

    # Now, try to fit a spline to the accumulated accuracy
    nknots = splines
    kx = np.linspace(0.0, 1.0, nknots)
    #   spline = utils.Spline (percentile, integrated_accuracy - integrated_scores, kx, runout=spline_method)
    spline = Spline(
        percentile, integrated_accuracy - integrated_scores, kx, runout=spline_method
    )

    # Evaluate the spline to get the accuracy
    acc = spline.evaluate_deriv(percentile)
    acc += scores

    # Return the interpolating function -- uses full (not decimated) scores and
    # accuracy
    func = interpolated_function(scores, acc)
    return func


# ------------------------------------------------------------------------------


def get_nth_results(scores, labels, n):
    tscores = np.array([score[n] for score in scores])
    tacc = np.array([1.0 if n == label else 0.0 for label in labels])
    return tscores, tacc


# ------------------------------------------------------------------------------


def get_top_results(scores, labels, nn, inclusive=False, return_topn_classid=False):
    # Different if we want to take inclusing scores
    if inclusive:
        return get_top_results_inclusive(scores, labels, nn=nn)

    #  nn should be negative, -1 means top, -2 means second top, etc
    # Get the position of the n-th largest value in each row
    topn = [np.argpartition(score, nn)[nn] for score in scores]
    nthscore = [score[n] for score, n in zip(scores, topn)]
    labs = [1.0 if int(label) == int(n) else 0.0 for label, n in zip(labels, topn)]

    # Change to tensor
    tscores = np.array(nthscore)
    tacc = np.array(labs)

    if return_topn_classid:
        return tscores, tacc, topn
    else:
        return tscores, tacc


# ------------------------------------------------------------------------------


def get_top_results_inclusive(scores, labels, nn=-1):
    #  nn should be negative, -1 means top, -2 means second top, etc
    # Order scores in each row, so that nn-th score is in nn-th place
    order = np.argpartition(scores, nn)

    # Reorder the scores accordingly
    top_scores = np.take_along_axis(scores, order, axis=-1)[:, nn:]

    # Get the top nn lables
    top_labels = order[:, nn:]

    # Sum the top scores
    sumscores = np.sum(top_scores, axis=-1)

    # See if label is in the top nn
    labs = np.array(
        [1.0 if int(label) in n else 0.0 for label, n in zip(labels, top_labels)]
    )

    return sumscores, labs


# ------------------------------------------------------------------------------


def do_plots(
    y_probs_val,
    y_val,
    y_probs_test,
    y_test,
    ece_criterion,
    spline_method,
    splines,
    outdir,
    title_val=None,
    title_test=None,
):
    results_file_beforecalib = os.path.join(outdir, "beforeCALIB_results.%s")
    results_beforecalib = ResultsLog(
        results_file_beforecalib % "csv", results_file_beforecalib % "html"
    )

    results_file_aftercalib = os.path.join(
        outdir, "afterCALIBspline" + spline_method + str(splines) + "_results.%s"
    )
    results_aftercalib = ResultsLog(
        results_file_aftercalib % "csv", results_file_aftercalib % "html"
    )

    # for top-class calibration error
    n = -1
    # Plot the top estimate n = -1 means top result, n=-2 means second-top, etc
    newtitle_val = title_val + f" Class[{n}]"
    newtitle_test = title_test + f" Class[{n}]"

    scores1, labels1, scores1_class = get_top_results(
        y_probs_val, y_val, n, return_topn_classid=True
    )
    scores2, labels2, scores2_class = get_top_results(
        y_probs_test, y_test, n, return_topn_classid=True
    )

    y_probs_binary_val = np.zeros((y_probs_val.shape[0], 2))
    y_probs_binary_test = np.zeros((y_probs_test.shape[0], 2))
    y_probs_binary_val[np.arange(scores1.shape[0]), 0] = scores1
    y_probs_binary_test[np.arange(scores2.shape[0]), 0] = scores2
    y_probs_binary_val[np.arange(scores1.shape[0]), 1] = 1.0 - scores1
    y_probs_binary_test[np.arange(scores2.shape[0]), 1] = 1.0 - scores2

    y_val_binary_onehot = np.zeros((y_probs_val.shape[0], 2))
    y_test_binary_onehot = np.zeros((y_probs_test.shape[0], 2))
    y_val_binary_onehot[:, 0] = labels1
    y_test_binary_onehot[:, 0] = labels2
    y_val_binary_onehot[:, 1] = 1 - labels1
    y_test_binary_onehot[:, 1] = 1 - labels2

    # Plot the first set
    KSElinf_uncal_val = plot_KS_graphs(
        scores1,
        labels1,
        spline_method,
        splines,
        outdir,
        "uncalibrated_val_class" + str(n),
        title=newtitle_val + " | Uncalibrated",
    )
    KSElinf_uncal_test = plot_KS_graphs(
        scores2,
        labels2,
        spline_method,
        splines,
        outdir,
        "uncalibrated_test_class" + str(n),
        title=newtitle_test + " | Uncalibrated",
    )
    ece_uncal_val = ece_criterion.eval(scores1, labels1)
    ece_uncal_test = ece_criterion.eval(scores2, labels2)

    # Get recalibration function, based on scores1
    frecal = get_recalibration_function(scores1, labels1, spline_method, splines)

    # Recalibrate scores1 and plot
    scores1 = np.array([frecal(float(sc)) for sc in scores1])
    scores1[scores1 < 0.0] = 0.0
    scores1[scores1 > 1.0] = 1.0
    KSElinf_cal_val = plot_KS_graphs(
        scores1,
        labels1,
        spline_method,
        splines,
        outdir,
        "spline_calibrated_val_class" + str(n),
        title=newtitle_val + " | Calibrated",
    )

    # Recalibrate scores2 and plot
    scores2 = np.array([frecal(float(sc)) for sc in scores2])
    scores2[scores2 < 0.0] = 0.0
    scores2[scores2 > 1.0] = 1.0
    KSElinf_cal_test = plot_KS_graphs(
        scores2,
        labels2,
        spline_method,
        splines,
        outdir,
        "spline_calibrated_test_class" + str(n),
        title=newtitle_test + " | Calibrated",
    )
    ece_cal_val = ece_criterion.eval(scores1, labels1)
    ece_cal_test = ece_criterion.eval(scores2, labels2)

    # for top 2nd-class calibration error
    n = -2
    newtitle_val = title_val + f" Class[{n}]"
    newtitle_test = title_test + f" Class[{n}]"

    scores1, labels1 = get_top_results(y_probs_val, y_val, n)
    scores2, labels2 = get_top_results(y_probs_test, y_test, n)

    # Plot the first set
    KSE2linf_uncal_val = plot_KS_graphs(
        scores1,
        labels1,
        spline_method,
        splines,
        outdir,
        "uncalibrated_val_class" + str(n),
        title=newtitle_val + " | Uncalibrated",
    )
    KSE2linf_uncal_test = plot_KS_graphs(
        scores2,
        labels2,
        spline_method,
        splines,
        outdir,
        "uncalibrated_test_class" + str(n),
        title=newtitle_test + " | Uncalibrated",
    )

    # Get recalibration function, based on scores1
    frecal = get_recalibration_function(scores1, labels1, spline_method, splines)

    # Recalibrate scores1 and plot
    scores1 = np.array([frecal(float(sc)) for sc in scores1])
    scores1[scores1 < 0.0] = 0.0
    scores1[scores1 > 1.0] = 1.0
    KSE2linf_cal_val = plot_KS_graphs(
        scores1,
        labels1,
        spline_method,
        splines,
        outdir,
        "spline_calibrated_val_class" + str(n),
        title=newtitle_val + " | Calibrated",
    )

    # Recalibrate scores2 and plot
    scores2 = np.array([frecal(float(sc)) for sc in scores2])
    scores2[scores2 < 0.0] = 0.0
    scores2[scores2 > 1.0] = 1.0
    KSE2linf_cal_test = plot_KS_graphs(
        scores2,
        labels2,
        spline_method,
        splines,
        outdir,
        "spline_calibrated_test_class" + str(n),
        title=newtitle_test + " | Calibrated",
    )

    ###################################################
    ### Estimating Within-k class KS score
    n = -2
    newtitle_val = title_val + f" Class[{n}]"
    newtitle_test = title_test + f" Class[{n}]"

    scores1, labels1 = get_top_results(y_probs_val, y_val, n, inclusive=True)
    scores2, labels2 = get_top_results(y_probs_test, y_test, n, inclusive=True)

    # Plot the first set
    KSE_wn_2linf_uncal_val = plot_KS_graphs(
        scores1,
        labels1,
        spline_method,
        splines,
        outdir,
        "uncalibrated_val_class_wn" + str(n),
        title=newtitle_val + " | Uncalibrated",
    )
    KSE_wn_2linf_uncal_test = plot_KS_graphs(
        scores2,
        labels2,
        spline_method,
        splines,
        outdir,
        "uncalibrated_test_class_wn" + str(n),
        title=newtitle_test + " | Uncalibrated",
    )

    # Get recalibration function, based on scores1
    frecal = get_recalibration_function(scores1, labels1, spline_method, splines)

    # Recalibrate scores1 and plot
    scores1 = np.array([frecal(float(sc)) for sc in scores1])
    scores1[scores1 < 0.0] = 0.0
    scores1[scores1 > 1.0] = 1.0
    KSE_wn_2linf_cal_val = plot_KS_graphs(
        scores1,
        labels1,
        spline_method,
        splines,
        outdir,
        "spline_calibrated_val_class_wn" + str(n),
        title=newtitle_val + " | Calibrated",
    )

    # Recalibrate scores2 and plot
    scores2 = np.array([frecal(float(sc)) for sc in scores2])
    scores2[scores2 < 0.0] = 0.0
    scores2[scores2 > 1.0] = 1.0
    KSE_wn_2linf_cal_test = plot_KS_graphs(
        scores2,
        labels2,
        spline_method,
        splines,
        outdir,
        "spline_calibrated_test_class_wn" + str(n),
        title=newtitle_test + " | Calibrated",
    )

    results_beforecalib.add(
        val_PECE=ece_uncal_val[0],
        test_PECE=ece_uncal_test[0],
        val_PKSE_linf=KSElinf_uncal_val,
        test_PKSE_linf=KSElinf_uncal_test,
        val_KSE2_linf=KSE2linf_uncal_val,
        test_KSE2_linf=KSE2linf_uncal_test,
        val_KSE_wn_2_linf=KSE_wn_2linf_uncal_val,
        test_KSE_wn_2_linf=KSE_wn_2linf_uncal_test,
    )
    results_beforecalib.save()

    results_aftercalib.add(
        val_PECE=ece_cal_val[0],
        test_PECE=ece_cal_test[0],
        val_PKSE_linf=KSElinf_cal_val,
        test_PKSE_linf=KSElinf_cal_test,
        val_KSE2_linf=KSE2linf_cal_val,
        test_KSE2_linf=KSE2linf_cal_test,
        val_KSE_wn_2_linf=KSE_wn_2linf_cal_val,
        test_KSE_wn_2_linf=KSE_wn_2linf_cal_test,
    )
    results_aftercalib.save()

    return (
        ece_uncal_val[0],
        ece_uncal_test[0],
        KSElinf_uncal_val,
        KSElinf_uncal_test,
        KSE2linf_uncal_val,
        KSE2linf_uncal_test,
        KSE_wn_2linf_uncal_val,
        KSE_wn_2linf_uncal_test,
        ece_cal_val[0],
        ece_cal_test[0],
        KSElinf_cal_val,
        KSElinf_cal_test,
        KSE2linf_cal_val,
        KSE2linf_cal_test,
        KSE_wn_2linf_cal_val,
        KSE_wn_2linf_cal_test,
    )


# ------------------------------------------------------------------------------


def first_above(A, val, low=0, high=-1):
    # Find the first time that the array exceeds, or equals val in the range low to high
    # inclusive -- this uses binary search

    # Initialization
    if high == -1:
        high = len0(A) - 1

    # Stopping point, when interval reduces to one element
    if high == low:
        if val <= A[low]:
            return low
        else:
            # The element does not exist.  This means that there is nowhere
            # in the array where A[k] >= val
            return low + 1  # This will be out-of-bounds if the array never exceeds val

    # Otherwise, we subdivide and continue -- mid must be less then high
    # but can equal low, when high-low = 1
    mid = low + (high - low) // 2

    if A[mid] >= val:
        # In this case, the first time must be in the interval [low, mid]
        return first_above(A, val, low, mid)
    else:
        # In this case, the first time A[k] exceeds val must be to the right
        return first_above(A, val, mid + 1, high)


def main(argv):
    # Random seed
    utils.set_seed()

    # First, get the arguments
    argspec = {
        "+gc": utils.str_to_list_of_int,  # List of classes to plot
        "+d": utils.str_to_list_of_str,  # Data files
    }

    # Get the arguments
    argvals = utils.parseargs(argv, argspec)

    # Classes to plot
    CLASSES_TO_PLOT = [-1]  # Meaning the top class

    # data_net_combinations = ["densenet40_c10", "densenet40_c100",
    #                          "densenet161_imgnet", "lenet5_c10",
    #                          "lenet5_c100", "resnet110_c10",
    #                          "resnet110_c100", "resnet110_SD_c10",
    #                          "resnet110_SD_c100", "resnet152_imgnet",
    #                          "resnet152_SD_SVHN", "resnet_wide32_c10",
    #                          "resnet_wide32_c100"]

    # DATA_FILES = ["saved_logits/probs_densenet40_c10_logits.p", "saved_logits/probs_densenet40_c100_logits.p",
    #               "saved_logits/probs_densenet161_imgnet_logits.p", "saved_logits/probs_lenet5_c10_logits.p",
    #               "saved_logits/probs_lenet5_c100_logits.p", "saved_logits/probs_resnet110_c10_logits.p",
    #               "saved_logits/probs_resnet110_c100_logits.p", "saved_logits/probs_resnet110_SD_c10_logits.p",
    #               "saved_logits/probs_resnet110_SD_c100_logits.p", "saved_logits/probs_resnet152_imgnet_logits.p",
    #               "saved_logits/probs_resnet152_SD_SVHN_logits.p", "saved_logits/probs_resnet_wide32_c10_logits.p",
    #               "saved_logits/probs_resnet_wide32_c100_logits.p"]

    # outdirs = ["out/cifar10/densenet40", "out/cifar100/densenet40",
    #            "out/imagenet/densenet161", "out/cifar10/lenet5",
    #            "out/cifar100/lenet5", "out/cifar10/resnet110",
    #            "out/cifar100/resnet110", "out/cifar10/resnet110SD",
    #            "out/cifar100/resnet110SD", "out/imagenet/resnet152",
    #            "out/SVHN/resnet152SD", "out/cifar10/wrn32",
    #            "out/cifar100/wrn32"]

    data_net_combinations = ["DHRNet_c10"]
    DATA_FILES = ["saved_logits/DHRNet_normal_format.pickle"]
    outdirs = ["out/cifar10/DHRNet"]

    # Override with flags, if necessary
    if argvals["+gc"] is not None:
        CLASSES_TO_PLOT = argvals["+gc"]
    if argvals["+d"] is not None:
        DATA_FILES = argvals["+d"]

    # Print a useful message
    print(
        f"\n  Classes to plot : {CLASSES_TO_PLOT}"
        f"\n  Data files      : {DATA_FILES}"
        f"\n"
    )

    ece_criterion = _ECELoss(n_bins=25)

    methods = ["natural"]

    for spline_method in methods:
        for spline in range(6, 7):
            # Plot graphs
            for i, fname in enumerate(DATA_FILES):
                if not os.path.exists(outdirs[i]):
                    os.makedirs(outdirs[i])

                # Read the logit data file
                ((y_probs_val, y_val), (y_probs_test, y_test)) = unpickle_probs(fname)

                print(
                    f"\nmain {fname}:"
                    f"\ny_probs_val  = {np.shape(y_probs_val)}"
                    f"\ny_probs_test = {np.shape(y_probs_test)}"
                    f"\ny_val        = {np.shape(y_val)}"
                    f"\ny_test       = {np.shape(y_test)}"
                    "\n"
                )

                (
                    ece_uncal_val,
                    ece_uncal_test,
                    KSElinf_uncal_val,
                    KSElinf_uncal_test,
                    KSE2linf_uncal_val,
                    KSE2linf_uncal_test,
                    KSE_wn_2linf_uncal_val,
                    KSE_wn_2linf_uncal_test,
                    ece_cal_val,
                    ece_cal_test,
                    KSElinf_cal_val,
                    KSElinf_cal_test,
                    KSE2linf_cal_val,
                    KSE2linf_cal_test,
                    KSE_wn_2linf_cal_val,
                    KSE_wn_2linf_cal_test,
                ) = do_plots(
                    y_probs_val,
                    y_val,
                    y_probs_test,
                    y_test,
                    ece_criterion,
                    spline_method,
                    spline,
                    outdirs[i],
                    title_val="Calib",
                    title_test="Test",
                )

    # Now, display the graphs
    sys.stdout.flush()
    plt.show()

    print("\nFinished successfully")


# Main routing
if __name__ == "__main__":
    main(sys.argv)
