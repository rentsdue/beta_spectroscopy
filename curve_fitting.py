import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# === Constants ===
m_e = 0.511        # electron mass (MeV)
R = 0.050          # spectrometer radius in m
factor = 300       # p[MeV] = 300 * B[T] * R[m]
DeltaB_T = 0.0005  # 0.5 mT in tesla

# Background (user-specified)
bg_rate_per_s = 0.32   # counts / second (background)
measurement_time_s = 100.0  # integration time per trial in seconds

# Compute background uncertainty (Poisson) over measurement_time_s
N_bg = bg_rate_per_s * measurement_time_s
sigma_bg_rate = np.sqrt(N_bg) / measurement_time_s

# === Beta spectrum model ===
def beta_spectrum(E, A, Q):
    p = np.sqrt(np.clip(E**2 - m_e**2, 0.0, None))
    spec = A * p * E * np.clip(Q - E, 0.0, None)**2
    return spec

# === Analysis function ===
def analyze_beta_decay(filename, isotope, background_rate=bg_rate_per_s, background_sigma_rate=sigma_bg_rate):
    # Load data
    df = pd.read_csv(filename, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    
    # Convert mT → T
    df["B_T"] = df["Magnetic Field Strength"] * 1e-3
    
    # Mean rate (counts/s)
    df["MeanRate_raw"] = df[["Trial 1","Trial 2","Trial 3","Trial 4","Trial 5"]].mean(axis=1) / measurement_time_s
    df["StatError_rate"] = df[["Trial 1","Trial 2","Trial 3","Trial 4","Trial 5"]].std(axis=1) / measurement_time_s
    
    # Background subtraction
    df["MeanRate"] = df["MeanRate_raw"] - background_rate
    df["MeanRate"] = df["MeanRate"].clip(lower=0)  # avoid negatives
    
    # Combine vertical errors (not needed for R^2, but kept for plotting)
    df["TotalError_rate"] = np.sqrt(df["StatError_rate"]**2 + background_sigma_rate**2)
    
    # Convert B → momentum → energy
    df["p"] = factor * df["B_T"] * R
    df["Energy_MeV"] = np.sqrt(df["p"]**2 + m_e**2)
    
    # Energy uncertainty from ΔB
    df["Delta_p"] = factor * R * DeltaB_T
    df["EnergyError_MeV"] = (df["p"] / df["Energy_MeV"]) * df["Delta_p"]
    
    # Arrays for fitting
    E_data = df["Energy_MeV"].values
    y_data = df["MeanRate"].values
    
    # Fit
    p0 = [max(y_data) if max(y_data) > 0 else 1.0, max(E_data)*0.9]
    popt, pcov = curve_fit(beta_spectrum, E_data, y_data, p0=p0, maxfev=10000)
    A_fit, Q_fit = popt
    A_err, Q_err = np.sqrt(np.diag(pcov))
    
    # R-squared calculation
    y_pred = beta_spectrum(E_data, *popt)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    
    # Print results
    print(f"\n=== {isotope} Fit Results (background subtracted) ===")
    print(f"Background rate subtracted = {background_rate:.3f} counts/s")
    print(f"A = {A_fit:.3e} ± {A_err:.3e}")
    print(f"Q = {Q_fit:.3f} ± {Q_err:.3f} MeV")
    print(f"R-squared = {r_squared:.3f}")
    
    # Plot
    plt.figure(figsize=(9,5))  # make figure slightly wider
    plt.errorbar(df["Energy_MeV"], y_data, xerr=df["EnergyError_MeV"],
                 yerr=df["TotalError_rate"], fmt="o", capsize=3, label="Data (bg subtracted)")
        # Fit curve for plotting
    E_fit = np.linspace(min(E_data), max(E_data), 400)
    y_fit = beta_spectrum(E_fit, *popt)

    # Find the peak of the fitted spectrum
    peak_idx = np.argmax(y_fit)
    peak_E = E_fit[peak_idx]
    peak_y = y_fit[peak_idx]

    # Create fit label
    fit_label = f"Fit:\nQ={Q_fit:.2f} MeV\nA={A_fit:.2f}\n$R^2$={r_squared:.3f}"
    plt.plot(E_fit, y_fit, "r-", label=fit_label)

    # Highlight the peak with a marker
    plt.plot(peak_E, peak_y, "ks", label=f"Peak: {peak_E:.2f} MeV")

    
    # Axis labels and title
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Net count rate (counts/s)")
    plt.title(f"{isotope} Beta Spectrum Fit (bg={background_rate:.2f} c/s)")
    plt.grid(True)

    # Put legend outside to the right
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    plt.tight_layout()


    # Put legend outside to the right
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()


    out_png = f"{isotope.lower().replace(' ', '_')}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    
    return df, popt, pcov, r_squared

# === Run for Sodium-22 and Strontium-90 ===
df_sod, popt_sod, pcov_sod, r2_sod = analyze_beta_decay("sodium.csv", "Sodium-22")
df_str, popt_str, pcov_str, r2_str = analyze_beta_decay("strontium.csv", "Strontium-90")
