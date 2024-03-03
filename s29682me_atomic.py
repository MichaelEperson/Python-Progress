# -*- coding: utf-8 -*-
# Header
"""
Title: PHYS10362 - Assignment 1 (Atomic Box)

Determining how the fraction, between the number of atoms at a given time, and
the initial number of atoms, changes with time.

This is done using user inputs for:
    - The half_life of type_A atoms (0s < and < 50s).
    - The frequency of which the type_B atom population oscillates
    (0.1Hz - 10Hz).
    - The minimum value of the fraction stated in line 6.

The code calculates:
    - The highest time, to 3 decimal places, when the fraction, stated in
    line 6, has the same value as the minimum provided by the user.
    - The number of oscillation made before the fraction, from line 6, drops 
    below the provided minimum for the final time.

Last Updated: 27/02/24
@author: s29682me
"""

# Important Statements
import matplotlib.pyplot as plt
import numpy as np

# Constant Definitions
# A = 1/(2*B), where B is a constant allowing the mean number of type-B atoms
# to equal the inital number of type-A atoms (0.08).
A = 6.25

# Function Definitions


def calculate_decay_constant_a():
    '''
    1. Asks the user for an input value for the half-life of the type-A atoms.
    2. Calculates the decay constant of the type-A atoms.
    3. Calculates the ratio of the number of type-A atoms at a given time, over
    the initial number of type-A atoms.

    Uses the methodology that the decay constant is the reciprocal of the mean
    lifetime of the atoms, and that the mean lifetime of the atoms is the
    halflife divided by natural logarithm of 2.

    Parameters
    ----------
    half_life_a_input : float, user inputted value for the half-life of the
                        type-A atoms.

    Returns
    -------
    decay_constant_a_output : float, the calculated decay constant of the
                              type-A atoms.

    Raises
    ------
    ValueError : if the inputted value for half_life_a_input is not a float.

    '''
    while True:
        try:
            half_life_a_input = float(input(
                'What is the half-life of the type-A atoms, in seconds? '))
            if 0 < half_life_a_input < 50:
                decay_constant_a_output = np.log(2) / half_life_a_input
                return decay_constant_a_output
            print('Please enter a value below above 0s and below 50s. ')
        except ValueError:
            print('Please enter your input as a float. ')


def calculate_angular_frequency_b():
    '''
    1. Asks the user for an input value for the frequency of which the
    population of the type-B atoms oscillates.
    2. Calculates the angular frequency.

    Parameters
    ----------
    frequency_b_input : float, oscillatory frequency of the type-B atoms

    Returns
    -------
    angular_frequencyy_b_output : float, the calculated angular frequency of
                                  the type-B atoms.

    Raises
    ------
    ValueError : if the inputted value for frequency_B_input is not a float.

    '''
    while True:
        try:
            frequency_b_input = float(
                input('What is the frequency at which the population of the '
                      'type-B atoms oscillate, in hertz? '))
            if 0.1 <= frequency_b_input <= 10:
                angular_frequency_b_output = np.pi * frequency_b_input
                return angular_frequency_b_output
            print('Please enter a value between 0.1Hz and 10Hz. ')
        except ValueError:
            print('Please enter your input as a float. ')


def get_fraction_minimum():
    '''
    Asks the user for the minimum value of the fraction of the number of atoms
    at a given time, reltaive to the inital number of atoms.

    Parameters
    ----------
    fraction_minimum_input : float, the inputted value of the minimum fraction
                             value.

    Returns
    -------
    fraction_minimum_output : float, the outputted minimum value of the 
                              fraction, now able to be automatically inputted
                              into further code.

    Raises
    ------
    ValueError: if the inputted value for fraction_minimum_input is not a 
                float.

    '''
    while True:
        try:
            fraction_minimum_input = float(
                input('What is the the minimum value of the fraction of the '
                      'number of atoms at a given time, relative to the inital'
                      ' number of atoms? '))
            if 0.18 <= fraction_minimum_input <= 0.98:
                fraction_minimum_output = fraction_minimum_input
                return fraction_minimum_output
            print('Please enter a value between 0.18 and 0.98. ')
        except ValueError:
            print('Please enter your input as a float. ')


def calculate_fraction(ratio_a_input, ratio_b_input):
    '''
    Calculates the fraction of the number of atoms at a given time, relative to
    the inital number of atoms.

    Parameters
    ----------
    ratio_A_input : float, uses the calculated value for ratio_a from the main 
                    code.
    ratio_B_input : float, uses the calculated value for ratio_b from the main
                    code.

    Returns
    -------
    fraction_output : float, the outputted value for the fraction of the
                      the total number atoms, relative to the intial number of
                      total atoms.

    '''
    fraction_output = ((A * ratio_a_input) + ratio_b_input) / (A + 1)
    return fraction_output


def find_maximum_time_index(fraction_input, fraction_minimum_input):
    '''
    Finds the maximum index before the value of the fraction goes below the
    value for fraction_minimum for the final time.

    Parameters:
    -----------
    fraction_input : array, values for the fraction at different times inputted
                     from the main code.
    fraction_minimum_input : float, minimum value of the fraction inputted from
                             the main code.

    Returns
    -------
    highest_index : int, the highest index of the fraction array that can be
                    classified as a peak of an oscillation, whilst also being
                    above the value of fraction_minimum_input.

    '''
    for i, value in enumerate(fraction_input[:-1]):
        if fraction_input[i+1] < value > fraction_minimum_input:
            highest_index = i
    return highest_index


def find_oscillation_number(fraction_input, fraction_minimum_input):
    '''
    Finds the number of oscillations made by the fraction before it falls below
    the inputted minimum value for the final time.

    Parameters
    ----------
    fraction_input : array, values for the fraction at different times inputted
                     from the main code.
    fraction_minimum_input : float, minimum value of the fraction inputted from
                             the main code.

    Returns
    -------
    oscillation_number_output : int, the length of the array containing all the
                                indices where the value of fraction_input is a
                                maximum, whilst being greater than
                                fraction_minimum_input.

    '''
    oscillations = []
    for i, value in enumerate(fraction_input[:-1]):
        if (i > 0 and fraction_input[i-1] < value > fraction_input[i + 1]
                and value > fraction_minimum_input):
            oscillations.append(i)
    oscillation_number_output = len(oscillations)
    return oscillation_number_output


# Main Code
decay_constant_a = calculate_decay_constant_a()
print(f'The decay constant of the type-A atoms is {decay_constant_a:.3f}s^-1 '
      '(3d.p).')

angular_frequency_b = calculate_angular_frequency_b()
print('The the angular frequency of the type-B atoms is '
      f'{angular_frequency_b:.3f}rads^-1 (3d.p).')

fraction_minimum = get_fraction_minimum()

half_life_a = np.log(2) / decay_constant_a
times = np.linspace(0, (6*half_life_a), 100000)

# The ratio of the number of type-A atoms at given times relative to the intial
# number of type-A atoms.
ratio_a = np.exp(-decay_constant_a * times)
# The ratio of the number of type-B atoms at given times relative to the intial
# number of type-B atoms.
ratio_b = (np.cos(angular_frequency_b * times))**2
# The ratio of the total number of atoms at a given time relative to the intial
# number of total atom.
fraction = calculate_fraction(ratio_a, ratio_b)

maximum_time_index = find_maximum_time_index(fraction, fraction_minimum)
print('The highest time when the fraction is above the minimum value '
      'is 'f'{times[maximum_time_index]:.3f}s (3d.p).')

oscillation_number = find_oscillation_number(fraction, fraction_minimum)
print('The final oscillation which has a higher-value fraction than the '
      f'minimum provided by the user is {oscillation_number}.')

plt.plot(times, fraction)
plt.axhline(y=fraction_minimum, color='r', linestyle='--', label='f_min')
plt.axvline(x=times[maximum_time_index], color='g',
            linestyle='--', label='t_max')
plt.text(0, fraction_minimum, 'f_min',
         color='r', ha='left', va='bottom', rotation='horizontal')
plt.text(times[maximum_time_index], (fraction_minimum + 0.1), 't_max',
         color='g', ha='right', va='bottom', rotation='vertical')
plt.xlabel('Time (s)')
plt.ylabel('Fraction')
plt.show()
