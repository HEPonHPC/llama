import os
from pandana import *
import numpy as np
import h5py

from NOvAPandAna.Utils import nd_flux
from NOvAPandAna.Cuts import nuebarcc_cuts

test_file = [
    os.path.join(
        "/Users/derekdoyle/data/",
        "grohmc_h5concat_nd_genie_N1810j0211a_nonswap_rhc_nova_v08_full_ndphysics_contain_v1_small.h5",
    )
]

kCalE = Var(lambda tables: tables["rec.slc"]["calE"])

kNumu = Cut(lambda tables: tables["rec.mc.nu"]["pdg"] == 14)
kIsCC = Cut(lambda tables: tables["rec.mc.nu"]["iscc"].astype(bool))
kSignal = kNumu & kIsCC

kBins = np.linspace(0, 40, 20)

if __name__ == "__main__":
    loader = NOVALoader(test_file)

    data = NOVASpectrum(loader, kSignal, kCalE)

    loader.Go()

    
