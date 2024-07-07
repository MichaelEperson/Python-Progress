# -*- coding: utf-8 -*-
"""
Title: PHYS10362 - Assignment 2 (Doppler)

This code can be used for doppler spectroscopy in the detection of extrasolar
planets. In space, any two bodies orbiting eachother orbit a common centre of
mass. From this, variations in a star's emitted wavelength of light suggested
it is sharing a centre of mass with a nearby planet.

This code is able to read in, validate, and add together multiple data sets
containting information on the doppler shift of a distance orbiting star. From
this data, it compares known mathematical fits for periodic orbits and
is able to find the orbiting velocity, angular frequency, and the phase 
difference from the sinusoidal fit of the star, through minimising the 
chi-sqaured of the data and the fit. Furthermore, the code can then calculate 
the mass of the planet, and the separation between the star and the planet.

All important calculations are paired together with reasonable uncertainties.

Last Updated: 01/05/2024
@author: s29682me
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as pc
from scipy.optimize import fmin

# Speed of light.
C = pc.c
# Sationary wavelength of the star.
WAVELENGTH_0 = 656.281E-9
# Gravitational constant.
G = pc.G
STAR_MASS = 2.78 * 1.989E30
# 1 astronimical unit in meters.
AU = pc.astronomical_unit
# 1 Julian year in seconds
YEAR = pc.Julian_year
# Guesses for the 3 parameters to be optimised.
V_0_GUESS = 50
ANG_FREQ_GUESS = 3E-8
PHASE_GUESS = np.pi


def initial_data_filtering(filename):
    """
    This function reads in data files and removes any lines containing elements
    that would cause issues later on:
        - Significant outliers (further than ±3 standard deviations from the 
          mean).
        - Non-numerical.

    Parameters
    ----------
    filename : txt file, dataset with coloumns for the time(yr), wavelength(nm)
               , and the uncertainty on the wavelength(nm), in that order.

    Returns
    -------
    doppler_data : array, a cleaner version of the inputted data file, now able
                   to be used in mathematical functions without errors.

    Raises
    ------
    FileNotFoundError : if the file name if not found in the local location of
                        the code file.

    """
    try:
        doppler_data_raw = np.genfromtxt(filename, delimiter=',')
    except FileNotFoundError:
        print(f"File '{filename}' not found. Skipping...")
        return None

    valid_lines = [line for line in doppler_data_raw if line[2] > 0 and
                   np.all(~np.isnan(line))]

    # The second column of the array is the wavelengths(nm) of the star.
    mean = np.mean(np.array(valid_lines)[:, 1])
    std = np.std(np.array(valid_lines)[:, 1])
    doppler_data = np.array(valid_lines)[np.abs((np.array(valid_lines)[:, 1]) -
                                                mean) < 3 * std]

    return doppler_data


def chi_squared(params, doppler_data, angle_of_sight):
    """
    This functions calculates the chi-squared between a set of data and an 
    expected fit.

    Parameters
    ----------
    params : floats, the orbital velocity, angular frequency, and phase shift 
             of the expected fit.
    doppler_data : array, the data to be compared with the expected fit.
    angle_of_sight : float, another parameter used in the expected fit. It is
                     not grouped with the others as it is not required to be
                     optimised later on when the chi-squared is minimised.

    Returns
    -------
    chi_sq : float, the value of the chi-squared between the data and expected
             fit.

    """
    v_0, ang_freq, phase = params

    # The first column of the array is time(s).
    # The third column of the array is the uncertainty on the wavelength(m)
    velocity = v_0 * np.sin((ang_freq * doppler_data[:, 0]) + phase)
    wavelength = (C + (velocity * np.sin(angle_of_sight))) * (WAVELENGTH_0 / C)
    chi_sq = np.sum(((doppler_data[:, 1] - wavelength)/doppler_data[:, 2])**2)

    return chi_sq


def final_data_filtering(params, doppler_data, angle_of_sight):
    """
    Once the data can be used in mathematical operations, and an appropriate 
    expected fit has been obtained, the data can be further filtered. Any lines 
    futher than ±3 standard deviations from the expected fit are moved into 
    their own array containing only outliers. 

    Parameters
    ----------
    params : floats, the orbital velocity, angular frequency, and phase shift 
             of the expected fit.
    doppler_data : array, the data that will be filered even further when 
                   compared to the expected fit
    angle_of_sight : float, another parameter used in the expected fit.

    Returns
    -------
    clean_lines : array, the indices of the data which passed the filtration 
                  process.
    outlier_lines : array, the indices of the data which not not pass the 
                    filtration process.

    """
    v_0, ang_freq, phase = params
    velocity = v_0 * np.sin((ang_freq * doppler_data[:, 0]) + phase)
    wavelength = (C + (velocity*np.sin(angle_of_sight))) * (WAVELENGTH_0 / C)

    difference = np.abs(doppler_data[:, 1] - wavelength)

    clean_lines = np.where(difference <= (3 * doppler_data[:, 2]))
    outlier_lines = np.where(difference > (3 * doppler_data[:, 2]))

    return clean_lines, outlier_lines


def find_uncertainties(params, doppler_data, angle_of_sight,
                       minimised_chi_squared):
    """
    This function generates the uncertainties on the orbital velocity and the
    angular frequency. This is done by finding the values at which the
    chi-squared has increased by +1, as this is the point at which the values
    are roughly ±1 standard deviation away from their optimal value.

    Parameters
    ----------
    params : floats, the orbital velocity, angular frequency, and phase shift 
             of the expected fit.
    doppler_data : array, used to control how long the arrays are for when they
                   are passed into other functions which require a specific
                   length.
    angle_of_sight : float, another parameter used in the expected fit.
    minimised_chi_squared : float, value of the miniised chi-squared value. It
                            is used to find the values where this has increased
                            or decreased by 1.

    Returns
    -------
    v_0_uncertainty : float, the uncertainty on the orbital velocity.
    ang_freq_uncertainty : float, the uncertainty on the angular frequency.

    """
    # The first value of params is the orbital velocity of the star.
    v_0_array = np.linspace(
        params[0], (1.1 * params[0]), len(doppler_data[:, 0]))
    # The second value of params is the angular frequency of the star.
    ang_freq_array = np.linspace(
        params[1], (1.1 * params[1]), len(doppler_data[:, 0]))

    chi_sq_v_0 = np.zeros_like(v_0_array)
    chi_sq_ang_freq = np.zeros_like(ang_freq_array)

    # The third value of params is the phase of the stars orbit.
    chi_sq_v_0 = np.array([chi_squared(
        (v_0, params[1], params[2]), doppler_data, angle_of_sight) for v_0 in
        v_0_array])
    chi_sq_ang_freq = np.array([chi_squared(
        (params[0], ang_freq, params[2]), doppler_data, angle_of_sight) for
        ang_freq in ang_freq_array])

    v_0_uncertainty_index = np.argmax(
        chi_sq_v_0 >= (minimised_chi_squared + 1))
    ang_freq_uncertainty_index = np.argmax(
        chi_sq_ang_freq >= (minimised_chi_squared + 1))

    v_0_uncertainty = np.abs(params[0] - v_0_array[v_0_uncertainty_index])
    ang_freq_uncertainty = np.abs(
        params[1] - ang_freq_array[ang_freq_uncertainty_index])

    return v_0_uncertainty, ang_freq_uncertainty


def plots(params, doppler_data, angle_of_sight, outliers, v_0_uncertainty,
          ang_freq_uncertainty, reduced_chi_squared):
    """
    This function takes in all of the final data and creates a figure
    showcasing:
        - the main analysis of the data and the fit
        - the residuals of the main analysis
        - a countour plot of the chi-squared
        - the final results that can be read from these plots.

    Parameters
    ----------
    params : floats, the orbital velocity, angular frequency, and phase shift 
             of the expected fit.
    doppler_data : array, the fully filtered data.
    angle_of_sight : float, another parameter used in the expected fit.
    outliers : array, the outliers found in the second round of filtering.
    v_0_uncertainty : float, the uncertainty on the orbital velocity.
    ang_freq_uncertainty : float, the uncertainty on the angular frequency.
    reduced_chi_squared : float, the value of the reduced chi-squared

    Returns
    -------
    None.

    """
    expected_velocity = params[0] * \
        np.sin((params[1] * doppler_data[:, 0]) + params[2])
    expected_wavelength = (
        C + (expected_velocity * np.sin(angle_of_sight))) * (WAVELENGTH_0 / C)
    residuals = doppler_data[:, 1] - expected_wavelength

    v_0_values = np.linspace(
        (0.9 * params[0]), (1.1 * params[0]), len(doppler_data[:, 0]))
    ang_freq_values = np.linspace(
        (0.9 * params[1]), (1.1 * params[1]), len(doppler_data[:, 0]))
    chi_squared_values = np.array([[chi_squared((v_0, ang_freq, params[2]),
                                  doppler_data, angle_of_sight) for v_0 in
                                  v_0_values] for ang_freq in ang_freq_values])

    fig = plt.figure(figsize=(12, 10))

    ax_1 = fig.add_subplot(221)
    ax_1.errorbar(doppler_data[:, 0], doppler_data[:, 1],
                  yerr=doppler_data[:, 2], fmt='o', color='g',
                  label='Clean data')
    ax_1.errorbar(outliers[:, 0], outliers[:, 1], yerr=outliers[:, 2], fmt='o',
                  color='r', label='Outliers')
    ax_1.plot(time_array, wavelength_fit, 'blue', label='Wavelength fit')
    ax_1.set_xlabel(r'Time (s)')
    ax_1.set_ylabel(r'Wavelength (m)')
    ax_1.set_title(r'Doppler Shift Analysis')
    ax_1.legend()

    ax_2 = fig.add_subplot(425)
    ax_2.errorbar(doppler_data[:, 0], residuals, yerr=doppler_data[:, 2],
                  fmt='o', color='g', label='Clean data')
    ax_2.axhline(y=0, color='blue')
    ax_2.set_xlabel(r'Time (s)')
    ax_2.set_ylabel(r'Residuals (m)')
    ax_2.set_title(r'Doppler Shift Analysis Residuals')

    ax_3 = fig.add_subplot(222)
    contour_plot_filled = ax_3.contourf(
        v_0_values, ang_freq_values, chi_squared_values, 100)
    fig.colorbar(contour_plot_filled, ax=ax_3)
    ax_3.set_xlabel(r'Velocity (m/s)')
    ax_3.set_ylabel(r'Angular Frequency (rad/s)')
    ax_3.set_title(r'Chi-Squared Variation')
    ax_3.scatter(params[0], params[1], color='red',
                 label='Minimised Chi-Squared')
    ax_3.axvline(x=params[0] + v_0_uncertainty, color='black', linestyle='--',
                 dashes=(10, 7), label='Upper and Lower Limit Values')
    ax_3.axvline(x=params[0] - v_0_uncertainty, color='black', linestyle='--',
                 dashes=(10, 7))
    ax_3.axhline(y=params[1] + ang_freq_uncertainty, color='black',
                 linestyle='--', dashes=(10, 7))
    ax_3.axhline(y=params[1] - ang_freq_uncertainty, color='black',
                 linestyle='--', dashes=(10, 7))
    ax_3.legend()

    fig.text(0.52, 0.45, f'v_0 = {params[0]:.4g} ± {v_0_uncertainty:.2g} m/s',
             fontsize=20, verticalalignment='bottom',
             horizontalalignment='left')
    fig.text(0.52, 0.40, f'\u03c9 = {fit_params_final[1]:.4g} ± '
             f'{ang_freq_error:.2g} rad/s', fontsize=20,
             verticalalignment='bottom', horizontalalignment='left')
    fig.text(0.52, 0.35, f'\u03c7\u00b2 = {reduced_chi_squared:.3g}',
             fontsize=20, verticalalignment='bottom',
             horizontalalignment='left')

    plt.subplots_adjust(hspace=0.5)
    plt.savefig('doppler_shift_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


# Adjust this if the orbit of the star is not along the line of sight from
# Earth.
angle_of_sight_input = np.pi/2

doppler_data_a_initial = initial_data_filtering('doppler_data_1.csv')
doppler_data_b_initial = initial_data_filtering('doppler_data_2.csv')

if doppler_data_a_initial is None and doppler_data_b_initial is None:
    raise FileNotFoundError(
        "Both data files are missing. Halting execution...")

if doppler_data_a_initial is None:
    doppler_data_initial = np.vstack(doppler_data_b_initial)
elif doppler_data_b_initial is None:
    doppler_data_initial = np.vstack(doppler_data_a_initial)
else:
    doppler_data_initial = np.vstack(
        (np.vstack(doppler_data_a_initial), np.vstack(doppler_data_b_initial)))

# Time(s)
doppler_data_initial[:, 0] *= YEAR
# Wavelength(m)
doppler_data_initial[:, 1] *= 1E-9
# Wavelength Uncertainty(m)
doppler_data_initial[:, 2] *= 1E-9

fit_params_initial = fmin(chi_squared,
                          (V_0_GUESS, ANG_FREQ_GUESS, PHASE_GUESS),
                          args=(doppler_data_initial, angle_of_sight_input),
                          disp=False)

clean_indices, outlier_indices = final_data_filtering(fit_params_initial,
                                                      doppler_data_initial,
                                                      angle_of_sight_input)

doppler_data_final = doppler_data_initial[clean_indices]

doppler_data_outliers = doppler_data_initial[outlier_indices]

# [v_0, ang_freq, phase]
fit_params_final = fmin(chi_squared, fit_params_initial,
                        args=(doppler_data_final, angle_of_sight_input),
                        disp=False)

minimised_chi_sq = chi_squared(fit_params_final, doppler_data_final,
                               angle_of_sight_input)
reduced_chi_sq = minimised_chi_sq / (len(doppler_data_final[:, 0]) - 3)

v_0_error, ang_freq_error = find_uncertainties(fit_params_final,
                                               doppler_data_final,
                                               angle_of_sight_input,
                                               minimised_chi_sq)

time_array = np.linspace(0, doppler_data_final[-1, 0], 1000)
velocity_fit = fit_params_final[0] * np.sin((fit_params_final[1] * time_array) +
                                            fit_params_final[2])
wavelength_fit = (C + (velocity_fit * np.sin(angle_of_sight_input))) * \
    (WAVELENGTH_0 / C)

separation = (((G*STAR_MASS*(((2*np.pi)/fit_params_final[1])**2)) /
               (4*(np.pi**2)))**(1/3))/AU
planet_mass = (fit_params_final[0] *
               (((separation*AU) * STAR_MASS)/G)**(1/2))/(1.898E27)

separation_error = separation * (ang_freq_error / fit_params_final[1]) * \
    np.sqrt(2/3)
planet_mass_error = planet_mass * \
    np.sqrt(((v_0_error / fit_params_final[0])**2) +
            ((1/2)*((separation_error / separation)**2)))

print(f"The orbital velocity of the star = {fit_params_final[0]:.4g} ±",
      f"{v_0_error:.2g} m/s\n\nThe angular frequency of the star =",
      f"{fit_params_final[1]:.4g} ± {ang_freq_error:.2g} rad/s\n\nThe",
      "reduced chi-squared between the data and the expected wavelength =",
      f"{reduced_chi_sq:.3g}\n\nThe separation between the star and the",
      f"orbiting planet = {separation:.4g} ± {separation_error:.2g} AU\n\n",
      f"The mass of the orbiting planet = {planet_mass:.4g} ±",
      f"{planet_mass_error:.2g} Jovian Masses")

plots(fit_params_final, doppler_data_final, angle_of_sight_input,
      doppler_data_outliers, v_0_error, ang_freq_error, reduced_chi_sq)
