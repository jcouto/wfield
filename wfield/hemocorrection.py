import numpy as np
from .utils import runpar
from scipy.signal import butter,filtfilt

def _hemodynamic_find_coeffs(U,SVTa,SVTb):
    a = np.dot(U,SVTa)
    b = np.dot(U,SVTb)
    return np.nansum(a*b,axis = 1)/np.nansum(b*b,axis = 1)

def hemodynamic_correction(U, SVT, fs=30., highpass = True, nchunks = 1000, run_parallel = True):
    # split channels and subtract the mean to each
    SVTa = SVT[:,0::2] 
    SVTb = SVT[:,1::2]

    # Highpass filter
    if highpass:
        b, a = butter(2,0.1/(fs/2.), btype='highpass');
        SVTa = filtfilt(b, a, SVTa, padlen=50)
        SVTb = filtfilt(b, a, SVTb, padlen=50)
    # subtract the mean
    SVTa = (SVTa.T - np.nanmean(SVTa,axis=1)).T.astype('float32')
    SVTb = (SVTb.T - np.nanmean(SVTb,axis=1)).T.astype('float32')

    npix = U.shape[0]
    idx = np.array_split(np.arange(0,npix),nchunks)
    # find the coefficients
    if run_parallel:     # run in parallel
        rcoeffs = runpar(_hemodynamic_find_coeffs,
                         [U[ind,:] for ind in idx],
                         SVTa=SVTa,
                         SVTb=SVTb)
        rcoeffs = np.hstack(rcoeffs).astype('float32')
    else:                # run in series
        rcoeffs = np.zeros((npix))
        for i,ind in tqdm(enumerate(idx)):
            #     rcoeffs[ind] = _hemodynamic_find_coeffs(U[ind,:],SVTa,SVTb)
            a = np.dot(U[ind,:],SVTa)
            b = np.dot(U[ind,:],SVTb)
            rcoeffs[ind] = np.sum(a*b,axis = 1)/np.sum(b*b,axis = 1)#regression_coeffs(a,b)
    # drop nan
    rcoeffs[np.isnan(rcoeffs)] = 1.
    # find the transformation
    T = np.dot(np.linalg.pinv(U),(U.T*rcoeffs).T)
    # apply correction
    SVTcorr = SVTa - np.dot(T.T,SVTb)
    return SVTcorr.astype('float32'), rcoeffs.astype('float32'), T.astype('float32')
