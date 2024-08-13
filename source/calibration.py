import numpy as np
from scipy import optimize
from scipy.special import softmax, logsumexp
from sklearn.isotonic import IsotonicRegression
import torch
from torch.nn import functional as F

import source.spline as spline 


def one_hot(label, num_class):
    y = F.one_hot(torch.Tensor(label).to(torch.int64), num_class)
    return y.numpy()


'''
[TS]
[1] https://github.com/gpleiss/temperature_scaling
[2] https://github.com/zhang64-llnl/Mix-n-Match-Calibration

'''
def ll_t(t, *args):
    logit, label = args
    label = one_hot(label, logit.shape[1])
    logit = logit/t
    p = np.clip(softmax(logit, axis=1),1e-20,1-1e-20)
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))/N
    return ce

def mse_t(t, *args):
    logit, label = args
    label = one_hot(label, logit.shape[1])
    logit = logit/t
    p = softmax(logit, axis=1)
    mse = np.mean((p-label)**2)
    return mse

def train_temperature_scaling(logit,label,loss):
    ## label should be one-hot encoded
    bnds = ((0.05, 5.0),)
    if loss == 'ce':
       t = optimize.minimize(ll_t, 1.0 , args = (logit,label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
    if loss == 'mse':
        t = optimize.minimize(mse_t, 1.0 , args = (logit,label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
    t = t.x
    return t


def calibrate_ts(logit, t):
    logit = logit / t
    p = softmax(logit, axis=1)
    return p


'''
[ETS]
[1] https://github.com/zhang64-llnl/Mix-n-Match-Calibration

'''
def ll_w(w, *args): # ETS with NLL
    p0, p1, p2, label = args
    label = one_hot(label, p0.shape[1])
    p = (w[0]*p0+w[1]*p1+w[2]*p2)
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))/N
    return ce

def mse_w(w, *args): # ETS with MSE
    p0, p1, p2, label = args
    label = one_hot(label, p0.shape[1])
    p = w[0]*p0+w[1]*p1+w[2]*p2
    p = p/np.sum(p,1)[:,None]
    mse = np.mean((p-label)**2)
    return mse

def train_ensemble_scaling(logit,label,t,n_class,loss='ce'):
    p1 = softmax(logit, axis=1)
    logit = logit/t
    p0 = softmax(logit, axis=1)
    p2 = np.ones_like(p0)/n_class
    

    bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),)
    def my_constraint_fun(x): return np.sum(x)-1
    constraints = { "type":"eq", "fun":my_constraint_fun,}
    if loss == 'ce':
        w = optimize.minimize(ll_w, (1.0, 0.0, 0.0) ,
                              args = (p0,p1,p2,label), method='SLSQP',
                              constraints = constraints, bounds=bnds_w,
                              tol=1e-12, options={'disp': False})
    if loss == 'mse':
        w = optimize.minimize(mse_w, (1.0, 0.0, 0.0) ,
                              args = (p0,p1,p2,label), method='SLSQP',
                              constraints = constraints, bounds=bnds_w,
                              tol=1e-12, options={'disp': False})
    w = w.x
    return w

def calibrate_ets(logit, w, t, n_class):
    p1 = softmax(logit, axis=1)
    logit = logit/t
    p0 = softmax(logit, axis=1)
    p2 = np.ones_like(p0)/n_class
    p = w[0]*p0 + w[1]*p1 +w[2]*p2
    return p


'''
[IRM]
[1] https://github.com/zhang64-llnl/Mix-n-Match-Calibration

'''
def train_isotonic_regression(logits, labels):
    labels = one_hot(labels, logits.shape[1])
    p = softmax(logits, axis=1)
    ir = IsotonicRegression(out_of_bounds='clip')
    y_ = ir.fit_transform(p.flatten(), (labels.flatten()))
    return ir

def calibrate_isotonic_regression(logits, ir):
    p_eval = softmax(logits, axis=1)
    yt_ = ir.predict(p_eval.flatten())
    p = yt_.reshape(logits.shape) + 1e-9 * p_eval
    return p


'''
[IROvA]
[1] https://github.com/zhang64-llnl/Mix-n-Match-Calibration

'''
def train_irova(logits, labels):
    labels = one_hot(labels, logits.shape[1])
    p = softmax(logits, axis=1)
    list_ir = []
    for ii in range(p.shape[1]):
        ir = IsotonicRegression(out_of_bounds='clip')
        y_ = ir.fit_transform(p[:, ii].astype('double'), labels[:, ii].astype('double'))
        list_ir.append(ir)
    return list_ir

def calibrate_irova(logits, list_ir):
    p_eval = softmax(logits, axis=1)
    for ii in range(p_eval.shape[1]):
        ir = list_ir[ii]
        p_eval[:, ii] = ir.predict(p_eval[:, ii]) + 1e-9 * p_eval[:, ii]
    return p_eval


'''
[IROvATS]
[1] https://github.com/zhang64-llnl/Mix-n-Match-Calibration

'''
def train_irovats(logits, labels, loss="mse"):
    t = train_temperature_scaling(logits, labels, loss=loss)
    logits = logits / t
    list_ir = train_irova(logits, labels)
    return (t, list_ir)
 
def calibrate_irovats(logits, t, list_ir):
    logits = logits / t
    p_eval = calibrate_irova(logits, list_ir)
    return p_eval


'''
[SPLINE]
[1] https://github.com/kartikgupta-at-anu/spline-calibration
[2] https://github.com/futakw/DensityAwareCalibration

'''
def train_spline(logits, labels):
    labels = one_hot(labels, logits.shape[1])
    SPL_frecal, p_wo_DAC, label_wo_DAC = spline.get_spline_calib_func(logits, labels)
    return SPL_frecal, p_wo_DAC, label_wo_DAC

def calibrate_spline(SPL_frecal, logits, labels):
    labels = one_hot(labels, logits.shape[1])
    calibrated_prob, tacc, predicted_label = spline.spline_calibrate(SPL_frecal, logits, labels)
    sample_num, classnum = logits.shape

    p_eval = np.zeros((sample_num, classnum))
    p_eval[np.arange(sample_num), predicted_label] = calibrated_prob
    return p_eval, tacc



'''
[ Energy Based Instance-wise Calibration ]
'''
from scipy.stats import norm

def mse_ebs(theta, *args):
    pdf_o, pdf_x, t, energy, logit, label = args
    T = 1
    energy = -(T*logsumexp(logit / T, axis=1))

    o_likelihood = pdf_o.pdf(energy)
    x_likelihood = pdf_x.pdf(energy)

    logit = logit/(t - o_likelihood*theta[0] + x_likelihood*theta[1])[:,np.newaxis]
    p = softmax(logit, axis=1)
    mse = np.mean((p-label)**2)
    return mse


def train_energycal(logits, labels, ood_logits, t):
    T = 1
    energy = -(T*logsumexp(logits / T, axis=1))

    # (1) correct energy pdf
    o_indices = np.argmax(softmax(logits, axis=1), axis=1) == labels
    o_samples = energy[o_indices]
    o_mu, o_sigma = norm.fit(o_samples)
    o_pdf = norm(o_mu, o_sigma)

    # (2) incorrect energy pdf
    labels = one_hot(labels, logits.shape[1])
    x_samples = energy[~(o_indices)]
    if ood_logits is not None:
        ood_energy = -(T*logsumexp(ood_logits / T, axis=1))
        x_samples = np.concatenate((x_samples, ood_energy))
        logits = np.concatenate((logits, ood_logits))
        labels = np.concatenate((labels, np.zeros((ood_logits.shape[0], logits.shape[1]))))

    x_mu, x_sigma = norm.fit(x_samples)
    x_pdf = norm(x_mu, x_sigma)

    shuffled_indices = np.random.permutation(logits.shape[0])
    logits = logits[shuffled_indices]
    labels = labels[shuffled_indices]
   
    bnds_theta = ((0.0, 10.0), (0.0, 10.0), (0.0, 10.0))
    def my_constraint_fun(x): return np.sum(x)-1
    constraints = { "type":"eq", "fun":my_constraint_fun,}

    theta = optimize.minimize(mse_ebs, (0.0,0.0,0.0),
                              args = (o_pdf, x_pdf, t, energy, logits, labels),
                              method='L-BFGS-B', constraints = constraints, bounds=bnds_theta, tol=1e-12,
                              options={'disp': True})
    theta = theta.x
    print('theta : ', theta)
    return theta, o_pdf, x_pdf


def calibrate_energycal(logits, t, theta, p_correct, p_incorrect):
    T = 1
    energy = -(T*logsumexp(logits / T, axis=1))
    o_likelihood = p_correct.pdf(energy)
    x_likelihood = p_incorrect.pdf(energy)
    logits = logits/(t - o_likelihood*theta[0] + x_likelihood*theta[1])[:,np.newaxis]
    p_eval = softmax(logits, axis=1)

    return p_eval
