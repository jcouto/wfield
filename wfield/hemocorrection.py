from .utils import *

def _hemodynamic_find_coeffs(U,SVTa,SVTb):
    a = np.dot(U,SVTa)
    b = np.dot(U,SVTb)
    eps = 1.e-10 # so we never divide by zero
    return np.nansum(a*b,axis = 1)/(np.nansum(b*b,axis = 1)+eps)

def hemodynamic_correction(U, SVT,
                           fs=60.,
                           freq_lowpass = 15.,
                           freq_highpass = 0.1,
                           nchunks = 1024,
                           run_parallel = True):
    # split channels and subtract the mean to each
    SVTa = SVT[:,0::2] 
    SVTb = SVT[:,1::2]

    # reshape U
    dims = U.shape
    U = U.reshape([-1,dims[-1]])

    # Single channel sampling rate
    fs = fs/2.
    # Highpass filter
    if not freq_highpass is None:
        SVTa = highpass(SVTa,w = freq_highpass, fs = fs)
        SVTb = highpass(SVTb,w = freq_highpass, fs = fs)
    if not freq_lowpass is None:
        if freq_lowpass <= fs/2:
            SVTb = lowpass(SVTb,freq_lowpass, fs = fs)
        else:
            print('Skipping lowpass on the violet channel.')
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
            # rcoeffs[ind] = _hemodynamic_find_coeffs(U[ind,:],SVTa,SVTb)
            a = np.dot(U[ind,:],SVTa)
            b = np.dot(U[ind,:],SVTb)
            rcoeffs[ind] = np.sum(a*b,axis = 1)/np.sum(b*b,axis = 1)
    # drop nan
    rcoeffs[np.isnan(rcoeffs)] = 1.e-10
    # find the transformation
    T = np.dot(np.linalg.pinv(U),(U.T*rcoeffs).T)
    # apply correction
    SVTcorr = SVTa - np.dot(T.T,SVTb)
    # return a zero mean SVT
    SVTcorr = (SVTcorr.T - np.nanmean(SVTcorr,axis=1)).T.astype('float32')
    # put U dims back in case its used sequentially
    U = U.reshape(dims)
    
    return SVTcorr.astype('float32'), rcoeffs.astype('float32').reshape(dims[:2]), T.astype('float32')
