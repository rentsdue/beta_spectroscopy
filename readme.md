# Beta Spectroscopy Analysis – Na-22 and Sr-90

This repository contains experimental data and Python analysis scripts for a beta spectroscopy experiment investigating the energy spectra of **Sodium-22 (Na-22)** and **Strontium-90 (Sr-90)** radioactive sources. The analysis was performed using a **simplified Fermi theory regression model** to estimate the endpoint energies (Q-values) of each decay process.

---

## Project Overview

The objective of this experiment was to:
1. Measure the beta emission count rates at different magnetic field strengths using a magnetic spectrometer.
2. Convert the measured field strengths to corresponding electron (or positron) total energies.
3. Fit the experimental energy distribution to a simplified beta spectrum model based on Fermi’s theory:
   \[
   N(E) = A\,p\,E\,(Q - E)^2,
   \]
   where:
   - \(p = \sqrt{E^2 - m_e^2}\) is the electron momentum,
   - \(A\) is a normalization constant,
   - \(Q\) is the endpoint (decay) energy.

This model neglects Coulomb corrections (the Fermi function), radiative corrections, and nuclear structure effects (shape factors), but still provides a reasonable first-order description of the beta spectrum shape.

---

## Contents

| File | Description |
|------|--------------|
| `sodium.csv` | Raw experimental data for Na-22 (magnetic field vs count rate). |
| `strontium.csv` | Raw experimental data for Sr-90 (magnetic field vs count rate). |
| `curve_fitting.py` | Python script performing nonlinear regression using `scipy.optimize.curve_fit`. |
| `sodium_fit.png` | Generated plot showing Na-22 fit results. |
| `strontium_fit.png` | Generated plot showing Sr-90 fit results. |
| `README.md` | This documentation file. |

---

## Regression Methodology

The script performs the following analysis steps:

1. **Data Preprocessing**
   - Reads experimental data (CSV format).
   - Computes the mean and standard error across repeated trials.
   - Converts magnetic field strength \(B\) (mT) to total particle energy \(E\) (MeV) using:
     \[
     p[\text{MeV}/c] = 300\,B[\text{T}]\,R[\text{m}],
     \qquad E = \sqrt{p^2 + m_e^2},
     \]
     where \(R = 0.05~\text{m}\) is the spectrometer radius and \(m_e = 0.511~\text{MeV}\) is the electron mass.

2. **Background Subtraction**
   - A constant background rate of **0.32 counts/s** is subtracted from all mean count rates.

3. **Curve Fitting**
   - Fits the simplified beta spectrum model to the data using nonlinear least squares.
   - Determines the best-fit parameters \(A\) and \(Q\) and their statistical uncertainties.

4. **Goodness of Fit**
   - Computes the coefficient of determination \(R^2\) to evaluate the agreement between the experimental data and the fitted model.

5. **Visualization**
   - Plots the data with vertical (count rate) and horizontal (energy) error bars.
   - Overlays the fitted theoretical curve and highlights the fitted peak energy.

---

## Interpretation Notes

- **Na-22**: The fitted endpoint energy (\(Q \approx 1.38~\text{MeV}\)) is higher than the dominant decay branch (0.546 MeV) but close to the rare 1.820 MeV transition. This discrepancy arises because the model assumes a single decay branch, while the actual Na-22 spectrum is a superposition of two positron-emission branches.

- **Sr-90**: The fitted endpoint energy (\(Q \approx 2.91~\text{MeV}\)) agrees reasonably with the dominant 2.274 MeV decay of Y-90. In this case, the higher-energy Y-90 decay dominates the overall spectrum since Sr-90 and Y-90 are in secular equilibrium.

---

## Dependencies

This project uses:
- Python ≥ 3.10  
- NumPy  
- Pandas  
- Matplotlib  
- SciPy  

To install the dependencies:
```bash
pip install numpy pandas matplotlib scipy
