import pandas as pd
import numpy as np
from math import modf, floor

def datetime_to_julian(dt):
    year = dt.dt.year.copy()
    month = dt.dt.month.copy()
    day = dt.dt.day
    hour = dt.dt.hour
    minute = dt.dt.minute
    second = dt.dt.second

    # Correction for january and February
    # Julian day calculation requires January and February to be treated as months 13 and 14 of the previous year
    mask = month <= 2
    year.loc[mask] -= 1
    month.loc[mask] += 12

    A = np.floor(year / 100)
    B = 2 - A + np.floor(A / 4)

    JD = (np.floor(365.25 * (year + 4716)) +
          np.floor(30.6001 * (month + 1)) +
          day + B - 1524.5 +
          (hour + minute / 60 + second / 3600) / 24)

    return JD


def jd_to_gregorian(jd):
    jd += 0.5
    F, I = modf(jd)
    I = int(I)
    
    if I >= 2299161:
        # Calendario gregoriano
        A = int((I - 1867216.25) / 36524.25)
        B = I + 1 + A - int(A / 4)
    else:
        # Calendario juliano
        B = I
    
    C = B + 1524
    D = int((C - 122.1) / 365.25)
    E = int(365.25 * D)
    G = int((C - E) / 30.6001)
    
    day = C - E + F - int(30.6001 * G)
    if G < 14:
        month = G - 1
    else:
        month = G - 13
    
    if month > 2:
        year = D - 4716
    else:
        year = D - 4715
    
    # Extraer parte entera y decimal del día para horas, minutos, segundos
    day_int = int(floor(day))
    day_frac = day - day_int
    
    hours = int(day_frac * 24)
    minutes = int((day_frac * 24 - hours) * 60)
    seconds = int((((day_frac * 24 - hours) * 60) - minutes) * 60)
    
    return year, month, day_int, hours, minutes, seconds

