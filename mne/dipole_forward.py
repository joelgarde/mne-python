""" Compute forward and backward propagation for a set of sensors.
"""
from dataclasses import dataclass
from mne.transforms import apply_trans

import numpy as np
import scipy.sparse as sp
from utils.experiment import memory
from mne.dipole import _make_guesses

from .bem import _bem_find_surface
from .cov import _ensure_cov, compute_whitener, whiten_evoked
from .forward._compute_forward import (_prep_field_computation, _make_ctf_comp_coils)
from .forward._make_forward import (_get_trans, _prep_eeg_channels,
                                    _prep_meg_channels, _setup_bem)
from .io.pick import pick_types
from .io.proj import _needs_eeg_average_ref_proj, make_projector
from .utils import (logger)
from scipy.optimize import minimize


_MAG_FACTOR = 1e-7  # μ_0 / (4π)

@dataclass
class Coils():
    """ coils data:
    r: integration point.
    n: normal at integration point.
    w: integration points to sensor value matrix. (n_sensors, n_int).
    """
    r: np.ndarray
    n: np.ndarray
    W: sp.spmatrix

    @classmethod
    def from_mne_coils(cls, coils):
        from scipy.sparse import csr_array
        if isinstance(coils, tuple):
            rmags, cosmags, ws, bins = coils
            ws = csr_array((ws, (bins,  np.arange(len(rmags)))), shape=(bins[-1] + 1, len(rmags)))

        else:
            rmags = np.concatenate([coil['rmag'] for coil in coils])
            cosmags = np.concatenate([coil['cosmag'] for coil in coils])
            ws = np.concatenate([coil['w'] for coil in coils])
            lens = np.asarray([len(coil['w']) for coil in coils])
            idptr = np.r_[0, np.cumsum(lens)]
            ws = csr_array((ws, np.arange(len(rmags)), idptr), shape=(len(lens), len(rmags)))
        return cls(rmags, cosmags, ws)
    


def K_EEG_MEG(r, r_bem, SMEG, SEEG, coils:Coils):
    """Compute forward matrix from dipoles at r.

    Args:
        r (n_dip, 3): dipole MRI coords.
        r_bem (n_bem, 3): BEM surface MRI coords.
        SMEG (S_meg, n_bem): Solution for mag field at BEM integration points.
        SEEG (S_eeg, n_bem): Solution for potential at EEG points.
        coils (Coils): MEG coils information, needs to be in MRI coords.

    Returns:
        _type_: _description_
    """
    G = G_infty(r_bem, r)
    Gcoil = G_infty(coils.r, r)
    KEEG = SEEG @ G
    KMEG = SMEG @ G
    K_inf = - np.cross(coils.n[:, np.newaxis, :], Gcoil.reshape((Gcoil.shape[0], -1, 3)), axisa=-1, axisb=-1).reshape((Gcoil.shape[0], -1))
    K_inf = coils.W  @ K_inf
    K_MEG = (K_inf + KMEG) * _MAG_FACTOR
    return np.r_[KEEG, KMEG]


def G_infty(n, r): 
    """compute the K_infty matrix (eq 40 of 10.1109/10.748978), the infinity potential design matrix.
    return:
    ------
    - G(N_nodes, N_dip * 3)
    """
    D = n[:, np.newaxis, :] - r
    d = np.einsum("ijk,ijk->ij", D, D)
    G = np.einsum("ij,ijk->ijk", d ** (-3/2), D)
    return G.reshape((n.shape[0], -1))

def prepare_forward(info, bem, cov, trans, ch_types, guess_grid=0.02):
    picks = pick_types(info, meg=True, eeg=True, ref_meg=False)

    mri_head_t, _  = _get_trans(trans)
    head_mri_t, _ = _get_trans(trans, fro='head', to='mri')
    neeg = len(pick_types(info, meg=False, eeg=True, ref_meg=False,
                        exclude=[])) 
    bem = _setup_bem(bem, f"{bem}", neeg, mri_head_t, verbose=False)
    cov = _ensure_cov(cov)
    whitener, _, rank = compute_whitener(cov, info, picks=picks,
                                        rank=None, return_rank=True)

    ### perform sanity checks.
    if _needs_eeg_average_ref_proj(info):
        raise ValueError('EEG average reference is mandatory for dipole '
                        'fitting.')

    if bem['is_sphere']:
        raise NotImplementedError('method only implemented for mesh bem.')

    # Forward model setup
    megcoils, compcoils, megnames, meg_info = [], [], [], None
    eegels, eegnames = [], []
    if 'grad' in ch_types or 'mag' in ch_types:
        megcoils, compcoils, megnames, meg_info = \
            _prep_meg_channels(info, exclude='bads',
                            accuracy="normal", verbose=False)
    if 'eeg' in ch_types:
        eegels, eegnames = _prep_eeg_channels(info, exclude='bads',
                                            verbose=False)

    fwd_data = dict(coils_list=[megcoils, eegels], infos=[meg_info, None],
                ccoils_list=[compcoils, None], coil_types=['meg', 'eeg'],)

    _prep_field_computation(None, bem, fwd_data, n_jobs=1,
                        verbose=False)

    # initial grid search space.
    min_dist_to_inner_skull = 5.  / 1000.
    guess_mindist = max(0.005, min_dist_to_inner_skull)
    guess_exclude = 0.02
    inner_skull = _bem_find_surface(bem, 'inner_skull')
    guess_src = _make_guesses(inner_skull, guess_grid, guess_exclude,
                                guess_mindist, n_jobs=1)[0]
    
    return guess_src, fwd_data, picks, megcoils, whitener, head_mri_t


def fit_dipole(evoked, cov, bem, trans, verbose=None):
    """
    """

    ### read data in.
    evoked = evoked.copy()
    info = evoked.info
    times = evoked.times.copy()
    comment = evoked.comment
    
    
    (guess_src, 
    fwd_data, 
    picks, 
    megcoils, 
    whitener, 
    head_mri_t) = prepare_forward(info, bem, cov, trans, evoked.get_channel_types(), guess_grid=0.005)
    data = evoked.data[picks]

    
    # mapping to my data:
    rr = guess_src['rr']
    rr_bem = fwd_data['bem_rr']
    SEEG = fwd_data['solutions'][1] or np.empty((0,rr_bem.shape[0])) 
    SMEG = fwd_data['solutions'][0]
    B = whitener @ data
    B2 = np.einsum("ij,ij->j", B, B)
    coils = Coils.from_mne_coils(megcoils)
    n_mri = apply_trans(head_mri_t, coils.n)
    r_mri = apply_trans(head_mri_t, coils.r)
    coils = Coils(r=r_mri, n=n_mri, W=coils.W)


    # fit dipoles!

    K: Never = whitener @ K_EEG_MEG(rr, rr_bem, SMEG, SEEG, coils)
    K = K.reshape((-1, K.shape[-1]//3, 3))
    K = np.swapaxes(K, 0, 1) # batch per dipole.
    U,S,VT = np.linalg.svd(K, full_matrices=False)    
    #U: (n_dipole, n_sensor, 3).
    deflate_mask = S[..., 0] > 5 * S[..., 2]
    U[deflate_mask, :, 2] = 0
    projection = np.swapaxes(U, 1, 2) @ B
    Bm2 = np.einsum("ijk,ijk->ik", projection, projection)
    r0i, t0i = np.unravel_index(np.argmax(Bm2), Bm2.shape)
    r0 = rr[r0i]

    fits = []
    def fit_fn(r):
        K = whitener @ K_EEG_MEG(r, rr_bem, SMEG, SEEG, coils)
        U,S,V = np.linalg.svd(K, full_matrices=False)
        deflate_mask = S[..., 0] > 5 * S[..., 2]
        if np.any(deflate_mask):
            print(f"regularisation applied ! Ratio:  {S[0]/S[2]: .2f}")
        U = np.expand_dims(U, 0)
        U[deflate_mask, : , 2] = 0
        projection = np.swapaxes(U, 1, 2) @ B
        Bm2 = np.einsum("ijk,ijk->ik", projection, projection) # projected  norm2 per dipole per time.
        fit = 1 - (Bm2[0,t0i] / B2[t0i])
        fits.append(fit)
        return fit

    path = [r0]
    def record(r):
        path.append(r)

    rhat = minimize(fit_fn, r0, method='Nelder-Mead', options=dict(disp=True), callback=record, tol=1e-8)

    fits = np.asarray(fits)
    path = np.asarray(path)
    return rhat, path, fits