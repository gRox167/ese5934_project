import bart
from fastmri import tensor_to_complex_np


def espirit_csm_estimation(kspace, num_low_freq)
    kspace = kspace.permute(1, 2, 0, 3).unsqueeze(0)
    kspace = tensor_to_complex_np(kspace)

    # estimate sensitivity maps
    if num_low_freqs is None:
        csms = bart.bart(1, "ecalib -d0 -m1", kspace)
    else:
        csms = bart.bart(1, f"ecalib -d0 -m1 -r {num_low_freqs}", kspace)
    return csms
