import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import math
import os

import llama

plt.style.use(hep.style.ROOT)


def histplot1d(
    plottable,
    ax=None,
    hist=True,
    errors=True,
    color="blue",
    hist_kwargs={"lw": 2},
    errors_kwargs=dict(),
):

    bins = plottable.axes[0].edges
    x = plottable.axes[0].centers
    y = plottable.values()
    yerr = np.sqrt(plottable.variances())
    xerr = np.diff(bins) / 2 if errors and not hist else None
    if ax is None:
        ax = plt.gca()

    if hist:
        ax.hist(x, bins=bins, weights=y, histtype="step", color=color, **hist_kwargs)

    if errors:
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, ls="none", color=color, **errors_kwargs)

    ax.set_xlim((bins[0], bins[-1]))
    ax.set_xlabel(plottable.axes[0].metadata)

    if isinstance(hist, llama.Spectrum):
        ax.set_ylabel("Events")
    return ax


def savefig(name, directory="./"):
    name = os.path.join(directory, name)
    plt.savefig(name)
    print(f"Wrote {name}")
    plt.close()
