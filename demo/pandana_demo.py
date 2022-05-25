import os
from pandana import *
import numpy as np
import h5py

from xsecana.efficiency import SimpleEfficiencyEstimator
from xsecana.signal import SimpleSignalEstimator
from xsecana.flux import SimpleFluxEstimator
from xsecana.unfold import SimpleUnfolder
from xsecana.cross_section import CrossSectionEstimator

from NOvAPandAna.Utils import nd_flux
from NOvAPandAna.Cuts import nuebarcc_cuts

test_file = [
    os.path.join(
        "/Users/derekdoyle/data/",
        "grohmc_h5concat_nd_genie_N1810j0211a_nonswap_rhc_nova_v08_full_ndphysics_contain_v1_small.h5",
    )
]

kCalE = Var(lambda tables: tables["rec.slc"]["calE"])
kTrueENT = Var(lambda tables: tables["neutrino"]["p.E"])
kTrueE = Var(lambda tables: tables["rec.mc.nu"]["p.E"])

kNumu = Cut(lambda tables: tables["rec.mc.nu"]["pdg"] == 14)
kIsCC = Cut(lambda tables: tables["rec.mc.nu"]["iscc"].astype(bool))
kSignal = kNumu & kIsCC
kSliceHasNeutrino = Cut(lambda tables: tables["rec.mc"]["nnu"].astype(bool))
kBackground = ~kSignal | ~kSliceHasNeutrino

kNumuNT = Cut(lambda tables: tables["neutrino"]["pdg"] == 14)
kIsCCNT = Cut(lambda tables: tables["neutrino"]["iscc"].astype(bool))
kSignalNT = kNumuNT & kIsCCNT

kSelection = Cut(lambda tables: tables["rec.slc"]["ncontplanes"] > 4)

kBins = np.linspace(0, 40, 20)

KLN = ["run", "subrun", "cycle", "batch", "evt", "subevt", "rec.mc.nu_idx"]

if __name__ == "__main__":
    loader = Loader(test_file, "evt.seq", "spill", KLN)

    signal_estimator = SimpleSignalEstimator(
        background=Spectrum(loader, kSelection & kBackground, kCalE,)
    )

    efficiency_estimator = SimpleEfficiencyEstimator(
        selected_signal=Spectrum(loader, kSelection & kSignal, kTrueE),
        all_signal=Spectrum(loader, kSignalNT, kTrueENT),
    )

    flux_estimator = SimpleFluxEstimator(
        Spectrum(
            loader,
            nd_flux.kIsNCQEOnCarbon(14) & nuebarcc_cuts.kNuebarCCIncFiducialST,
            nd_flux.kTrueNeutrinoEnergy,
        ),
        Spectrum(
            loader,
            nd_flux.kIsNCQEOnCarbon(14) & nuebarcc_cuts.kNuebarCCIncFiducialST,
            nd_flux.kXSecM2,
        ),
        integrated=True,
        ntargets=1,
    )

    unfolder = SimpleUnfolder(
        reco=Spectrum(loader, kSelection & kSignal, kCalE),
        truth=Spectrum(loader, kSelection & kSignal, kTrueE),
    )

    data = Spectrum(loader, kSelection, kCalE)

    loader.Go()

    cross_section = CrossSectionEstimator(
        signal_estimator, flux_estimator, efficiency_estimator, 1, unfolder, False,
    )(kBins, data)

    print(cross_section)

    np.nan_to_num(cross_section._data, copy=False, nan=-5.0)

    if os.path.isfile("serial_cross_section.h5"):
        with h5py.File("serial_cross_section.h5", "r") as f:
            truth = f.get("cross_section")[:]
            assert (cross_section.data == truth).all()
    else:
        with h5py.File("serial_cross_section.h5", "w") as f:
            f.create_dataset(
                "cross_section", data=cross_section.data,
            )

    """"
    truth = np.array(
        [
            44.77894319,
            100.25684676,
            252.79499359,
            396.67082609,
            393.8564433,
            385.85322179,
            472.40478622,
            399.95632207,
            728.44606428,
            443.79049798,
            297.15472336,
            1016.97534766,
            589.14133862,
            np.nan,
            475.00589971,
            247.52418193,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    """
