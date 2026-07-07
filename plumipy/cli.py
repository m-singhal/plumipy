#!/usr/bin/env python3
"""
PLUMIPY command-line interface — for HPC / terminal use.

Examples
--------
# Adiabatic PES
plumipy run \\
  --structure-gs CONTCAR_gs --structure-es CONTCAR_es \\
  --phonons-gs OUTCAR_gs --zpl 1000 --sigma 3 3 --gamma 2 \\
  --monte-carlo --output results.h5

# Vertical gradient
plumipy run \\
  --structure-gs CONTCAR_gs \\
  --forces-gs OUTCAR_gs --forces-es OUTCAR_es \\
  --phonons-gs OUTCAR_phonons --zpl 1000 --gradient --output results.h5

# Headless plot from saved HDF5
plumipy plot results.h5 --type emission --dpi 300 --output emission.png

# Print results summary
plumipy info results.h5
"""
import sys
import os
import numpy as np

import click

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@click.group()
def cli():
    """PLUMIPY — photoluminescence & absorption spectra from DFT outputs."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# plumipy run
# ─────────────────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--structure-gs",   default=None, help="GS structure (POSCAR/CONTCAR or .npy/.txt)")
@click.option("--structure-es",   default=None, help="ES structure")
@click.option("--phonons-gs",     default=None, help="GS phonons (OUTCAR or band.yaml)")
@click.option("--phonons-es",     default=None, help="ES phonons (optional, for squeezing)")
@click.option("--forces-gs",      default=None, help="GS forces (OUTCAR or .npy/.txt)")
@click.option("--forces-es",      default=None, help="ES forces")
@click.option("--vib-freqs-gs",   default=None, help="GS vibrational frequencies (.npy/.txt)")
@click.option("--vib-freqs-es",   default=None, help="ES vibrational frequencies")
@click.option("--vib-modes-gs",   default=None, help="GS normal modes (.npy/.txt)")
@click.option("--vib-modes-es",   default=None, help="ES normal modes")
@click.option("--masses",         default=None, help="Atomic masses (.npy/.txt), amu")
@click.option("--zpl",            default=1000.0, show_default=True, help="ZPL energy in meV")
@click.option("--sigma",          default=(3.0, 3.0), nargs=2, type=float,
              show_default=True, help="Sideband broadening σ₁ σ₂ in meV")
@click.option("--gamma",          default=2.0, show_default=True, help="Lorentzian broadening in meV")
@click.option("--lorentzian",     is_flag=True, help="Use Lorentzian sidebands (default: Gaussian)")
@click.option("--temperature",    default=0.0, show_default=True, help="Temperature in K")
@click.option("--subtract-modes", default=0,   show_default=True,
              help="Number of low-freq modes to remove")
@click.option("--gradient",       is_flag=True, help="Use vertical gradient (force-based) method")
@click.option("--freq-unit",      default="cm^-1",
              type=click.Choice(["cm^-1", "THz"]), show_default=True,
              help="Unit for external vibrational frequencies")
@click.option("--monte-carlo",    is_flag=True, help="Enable Monte Carlo emission sampling")
@click.option("--squeezing",      is_flag=True, help="Enable displaced-squeezed oscillator")
@click.option("--sigma-sq",       default=1.0,  show_default=True, help="Squeezed broadening σ in meV")
@click.option("--gamma-sq",       default=2.0,  show_default=True, help="Squeezed Lorentzian γ in meV")
@click.option("--output",         default="spectra_output.h5", show_default=True,
              help="Output HDF5 file path")
def run(structure_gs, structure_es, phonons_gs, phonons_es,
        forces_gs, forces_es, vib_freqs_gs, vib_freqs_es,
        vib_modes_gs, vib_modes_es, masses,
        zpl, sigma, gamma, lorentzian, temperature, subtract_modes,
        gradient, freq_unit, monte_carlo, squeezing, sigma_sq, gamma_sq, output):
    """Run the spectral calculation and save results to HDF5."""
    from plumipy import calculate_spectra_analytical
    import h5py

    qk_type = "f" if gradient else "r"

    click.echo("PLUMIPY  —  running calculation…")
    click.echo(f"  ZPL        : {zpl} meV")
    click.echo(f"  Method     : {'Vertical Gradient' if gradient else 'Adiabatic PES'}")
    click.echo(f"  Monte Carlo: {monte_carlo}")
    click.echo(f"  Squeezing  : {squeezing}")
    click.echo()

    try:
        results = calculate_spectra_analytical(
            structure_gs=structure_gs,
            structure_es=structure_es,
            forces_gs=forces_gs,
            forces_es=forces_es,
            phonons_gs=phonons_gs,
            phonons_es=phonons_es,
            vibrational_freqs_gs=vib_freqs_gs,
            vibrational_freqs_es=vib_freqs_es,
            vibrational_modes_gs=vib_modes_gs,
            vibrational_modes_es=vib_modes_es,
            masses=masses,
            qk_calculation_type=qk_type,
            zpl=zpl,
            sigma_init=sigma[0],
            sigma_final=sigma[1],
            gamma=gamma,
            sidebands_broadening_lorentzian=lorentzian,
            vibrational_freqs_unit=freq_unit,
            subtract_modes=subtract_modes,
            temperature=temperature,
            enable_squeezing=squeezing,
            sigma_squeezed=sigma_sq if squeezing else None,
            gamma_squeezed=gamma_sq if squeezing else None,
            monte_carlo_emission=monte_carlo,
            save_to_hdf5=False,
        )
    except Exception as e:
        click.echo(f"ERROR: {e}", err=True)
        sys.exit(1)

    # Print summary
    if "HR" in results:
        click.echo(f"  Huang–Rhys factor S  = {results['HR']:.4f}")
        click.echo(f"  Debye–Waller  e^-S   = {np.exp(-results['HR']):.4f}")
    if "monte_carlo_emission" in results:
        mc = results["monte_carlo_emission"]
        click.echo(f"  MC emission mean     = {mc['mean']:.2f} meV")
        click.echo(f"  MC emission std      = {mc['std']:.2f} meV")

    # Save HDF5
    def _write(group, d):
        for k, v in d.items():
            if isinstance(v, dict):
                _write(group.create_group(k), v)
            elif v is not None:
                try:
                    group.create_dataset(k, data=np.array(v))
                except Exception:
                    pass

    with h5py.File(output, "w") as f:
        _write(f, results)

    click.echo(f"\n✓ Results saved to: {output}")


# ─────────────────────────────────────────────────────────────────────────────
# plumipy info
# ─────────────────────────────────────────────────────────────────────────────
@cli.command()
@click.argument("hdf5_file")
def info(hdf5_file):
    """Print a summary of results stored in an HDF5 file."""
    import h5py
    from plumipy.io import load_hdf5_results

    results = load_hdf5_results(hdf5_file)
    click.echo(f"\nFile: {hdf5_file}")
    click.echo("─" * 50)
    if "HR" in results:
        click.echo(f"  Huang–Rhys factor S   = {results['HR']:.4f}")
        click.echo(f"  Debye–Waller  e^-S    = {np.exp(-float(results['HR'])):.4f}")
    if "Ek_gs" in results:
        Ek = results["Ek_gs"]
        click.echo(f"  Number of modes       = {len(Ek)}")
        click.echo(f"  Energy range          = {Ek.min():.1f} – {Ek.max():.1f} meV")
    if "monte_carlo_emission" in results:
        mc = results["monte_carlo_emission"]
        click.echo(f"  MC emission mean      = {float(mc['mean']):.2f} meV")
        click.echo(f"  MC emission std       = {float(mc['std']):.2f} meV")
        click.echo(f"  MC skewness           = {float(mc['skewness']):.3f}")
    top_keys = [k for k in results if not isinstance(results[k], dict) and not isinstance(results.get(k), np.ndarray)]
    click.echo(f"\n  Top-level keys: {list(results.keys())}")
    click.echo()


# ─────────────────────────────────────────────────────────────────────────────
# plumipy plot
# ─────────────────────────────────────────────────────────────────────────────
@cli.command()
@click.argument("hdf5_file")
@click.option("--type", "plot_type",
              type=click.Choice(["emission", "absorption", "spectral", "modes", "all"]),
              default="all", show_default=True)
@click.option("--output", default=None, help="Output image path (e.g. fig.png)")
@click.option("--dpi",    default=300,  show_default=True, help="Output DPI")
@click.option("--format", "fmt",
              type=click.Choice(["png", "jpg", "svg", "pdf"]),
              default="png", show_default=True)
def plot(hdf5_file, plot_type, output, dpi, fmt):
    """Generate publication-quality figures from a saved HDF5 file."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plumipy.io import load_hdf5_results

    results = load_hdf5_results(hdf5_file)
    base = os.path.splitext(output or hdf5_file)[0]

    BG    = "#1e1e2e"
    AXBG  = "#181825"
    TEXT  = "#cdd6f4"
    BLUE  = "#89b4fa"
    GREEN = "#a6e3a1"
    RED   = "#f38ba8"
    GRID  = "#313244"

    def _style(fig, axes):
        fig.patch.set_facecolor(BG)
        for ax in (axes if hasattr(axes, "__iter__") else [axes]):
            ax.set_facecolor(AXBG)
            ax.tick_params(colors=TEXT)
            ax.xaxis.label.set_color(TEXT)
            ax.yaxis.label.set_color(TEXT)
            ax.title.set_color(TEXT)
            for s in ax.spines.values(): s.set_edgecolor(GRID)
            ax.grid(True, color=GRID, lw=0.5, alpha=0.8)
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(AXBG)
                for t in leg.get_texts(): t.set_color(TEXT)

    def _save(fig, name):
        p = f"{base}_{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=BG)
        click.echo(f"  Saved: {p}")
        plt.close(fig)

    if plot_type in ("emission", "all") and "standard_hr" in results:
        std = results["standard_hr"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(std["E_photon_emission"], np.real(std["I_emission"]),
                color=BLUE, lw=2, label="Analytical")
        if "monte_carlo_emission" in results:
            mc = results["monte_carlo_emission"]
            w = np.mean(np.diff(mc["E_photon_emission"]))
            ax.bar(mc["E_photon_emission"], mc["I_emission"],
                   width=w, color=RED, alpha=0.55, label="Monte Carlo")
        ax.set_xlabel("Photon Energy (meV)"); ax.set_ylabel("PL (arb.)")
        ax.set_title("Emission Spectrum"); ax.legend()
        _style(fig, [ax]); _save(fig, "emission")

    if plot_type in ("absorption", "all") and "standard_hr" in results:
        std = results["standard_hr"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(std["E_photon_absorption"], np.real(std["I_absorption"]),
                color=GREEN, lw=2, label="Absorption")
        ax.set_xlabel("Photon Energy (meV)"); ax.set_ylabel("Abs. (arb.)")
        ax.set_title("Absorption Spectrum"); ax.legend()
        _style(fig, [ax]); _save(fig, "absorption")

    if plot_type in ("spectral", "all") and "standard_hr" in results and "Sk" in results:
        std = results["standard_hr"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax2 = ax.twinx()
        ax.plot(std["E_phonons"], std["S_E"], color=BLUE, lw=2, label="S(E)")
        Ek = results["Ek_gs"]; Sk = results["Sk"]
        w = (Ek.max()-Ek.min())*0.012
        ax2.bar(Ek, Sk, width=w, color=RED, alpha=0.7, label="Sₖ")
        ax.set_xlabel("Phonon Energy (meV)"); ax.set_ylabel("S(E) (meV⁻¹)", color=BLUE)
        ax2.set_ylabel("Sₖ", color=RED)
        ax.set_title(f"Spectral Function  |  S = {results['HR']:.3f}")
        _style(fig, [ax, ax2]); _save(fig, "spectral_function")

    if plot_type in ("modes", "all") and "Sk" in results:
        Sk = results["Sk"]; Ek = results["Ek_gs"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(np.arange(1, len(Sk)+1), Sk, color=RED, alpha=0.8)
        axes[0].set_xlabel("Mode index"); axes[0].set_ylabel("Sₖ")
        axes[0].set_title("Huang–Rhys factors per mode")
        axes[1].scatter(Ek, Sk, c=Sk, cmap="plasma", s=40, alpha=0.8)
        axes[1].set_xlabel("Phonon Energy (meV)"); axes[1].set_ylabel("Sₖ")
        axes[1].set_title("Sₖ vs Phonon Energy")
        _style(fig, list(axes)); _save(fig, "mode_analysis")

    click.echo("\n✓ Done.")


def main():
    cli()


if __name__ == "__main__":
    main()
