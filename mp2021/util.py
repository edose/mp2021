__author__ = "Eric Dose, Albuquerque"

""" This module: Utilities in service of other modules. """

# Python core:
import os

# External packages:
import pandas as pd

# Author's packages:
import mp2021.ini as ini


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')

EARLIEST_APPLICABLE_AN = 20000000
LATEST_APPLICABLE_AN = 21000000
VALID_FITS_FILE_EXTENSIONS = ('.fits', '.fit', '.fts')


class SessionLogFileError(Exception):
    """ Raised on any problem with log file (because we can't continue)."""


def parse_mp_id(mp_id):
    """ Return MP ID string for valid MP ID
        Modified from mp_phot package, util.py, .get_mp_and_an_string().
    :param mp_id: raw MP identification, either number or other ID, e.g., 5802 (int), '5802', or
         '1992 SG4'. [int or string]
    :return: for numbered MP, give simply the string, e.g. '5802'.
             for other MP ID, give the string prepended wtih '~', e.g., '~1992 SG4'.
         """
    if isinstance(mp_id, int):
        if mp_id < 1:
            raise ValueError('MP ID must be a positive integer.')
        mp_id_string = str(mp_id)  # e.g., '1108' for numbered MP ID 1108 (if passed in as int).
    elif isinstance(mp_id, str):
        if mp_id[0] not in '0123456789':
            raise ValueError('MP ID does not appear valid.')
        try:
            _ = int(mp_id)  # a test only
        except ValueError:
            mp_id_string = '~' + mp_id   # e.g., '*1997 TX3' for unnumbered MP ID '1997 TX3'.
        else:
            mp_id_string = mp_id.strip()  # e.g., '1108' for numbered MP ID 1108 (if passed in as string).
    else:
        raise TypeError('mp_id must be an integer or string representing a valid MP ID.')
    return mp_id_string


def parse_an_date(an_id):
    """ Return MP ID string for valid MP ID
        Modified from mp_phot package, util.py, .get_mp_and_an_string().
    :param an_id: Astronight ID yyyymmdd. [int, or string representing an int]
    :return: an_id_string, always an 8 character string 'yyyymmdd' representing a proper Astronight ID.
    """
    if not isinstance(an_id, (str, int)):
        raise TypeError('an_id must be a string or an integer representing a valid Astronight yyyymmdd.')
    an_string = str(an_id).strip()
    try:
        an_integer = int(an_string)  # a test only
    except ValueError:
        raise ValueError('an_id must be a string or an integer representing a valid Astronight yyyymmdd.')
    if an_integer < EARLIEST_APPLICABLE_AN or int(an_string) > LATEST_APPLICABLE_AN:
        raise ValueError('an_id is outside applicable range of dates.')
    return an_string


def get_mp_filenames(directory):
    """ Get only filenames in directory like MP_xxxxx.[ext], where [ext] is a legal FITS extension.
        The order of the return list is alphabetical (which may or may not be in time order).
        Adapted from mp_phot package, util.get_mp_filenames().
    :param directory: path to directory holding MP FITS files. [string]
    """
    all_filenames = pd.Series([entry.name for entry in os.scandir(directory) if entry.is_file()])
    extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in all_filenames])
    is_fits = [ext.lower() in VALID_FITS_FILE_EXTENSIONS for ext in extensions]
    fits_filenames = all_filenames[is_fits]
    mp_filenames = [fn for fn in fits_filenames if fn.startswith('MP_')]
    mp_filenames.sort()
    return mp_filenames


def fits_header_value(hdu, key):
    """ Take FITS hdu and a key, return value associated with key (or None if key absent).
        Adapted from package photrix, class FITS.header_value.
    :param hdu: astropy fits header+data unit object.
    :param key: FITS header key [string] or list of keys to try [list of strings]
    :return: value of FITS header entry, typically [float] if possible, else None. [string or None]
    """
    if isinstance(key, str):  # case of single key to try.
        return hdu.header.get(key, None)
    for k in key:             # case of list of keys to try.
        value = hdu.header.get(k, None)
        if value is not None:
            return value
    return None


def fits_is_plate_solved(hdu):
    """ Take FITS hdu, return True iff FITS image appears to be plate-solved.
        Adapted loosely from package photrix, class FITS. Returns boolean.
    :param hdu: astropy fits header+data unit object.
    :return: True iff FITS image appears to be plate-solved, else False. [boolean]
    """
    # TODO: tighten these tests, prob. by checking for reasonable numerical values.
    plate_solution_keys = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']
    values = [fits_header_value(hdu, key) for key in plate_solution_keys]
    for v in values:
        if v is None:
            return False
        if not isinstance(v, float):
            try:
                _ = float(v)
            except ValueError:
                return False
            except TypeError:
                return False
    return True


def fits_is_calibrated(hdu):
    """ Take FITS hdu, return True iff FITS image appears to be calibrated by flats & darks.
        Adapted from package photrix, class FITS. Currently requires calibration by Maxim 5 or 6,
           but could be extended.
    :param hdu: astropy fits header+data unit object.
    :return: True iff FITS image appears to be calibrated, else False. [boolean]
    """
    # First, define all calibration functions as internal, nested functions:
    def _is_calibrated_by_maxim_5_or_6(hdu):
        calibration_value = fits_header_value(hdu, 'CALSTAT')
        if calibration_value is not None:
            if calibration_value.strip().upper() == 'BDF':
                return True
        return False

    # If any is function signals valid, then fits is calibrated:
    is_calibrated_list = [_is_calibrated_by_maxim_5_or_6(hdu)]  # expand later if more calibration fns.
    return any([is_cal for is_cal in is_calibrated_list])


def fits_focal_length(hdu):
    """ Takes FITS hdu, return best estimate of imaging system's focal length.
        Adapted from package photrix, class FITS. Returns float if focal length appears valid, else None.
    :param hdu: astropy fits header+data unit object.
    :return: Best estimate of focal length, in mm, or None if invalid. [float, or None]
    """
    fl = fits_header_value(hdu, 'FOCALLEN')
    if fl is not None:
        return fl  # in millimeters.
    x_pixel_size = fits_header_value(hdu, 'XPIXSZ')
    y_pixel_size = fits_header_value(hdu, 'YPIXSZ')
    x_pixel_scale = fits_header_value(hdu, 'CDELT1')
    y_pixel_scale = fits_header_value(hdu, 'CDELT2')
    if any([value is None for value in [x_pixel_size, y_pixel_size, x_pixel_scale, y_pixel_scale]]):
        return None
    fl_x = x_pixel_size / abs(x_pixel_scale) * (206265.0 / (3600 * 1000))
    fl_y = y_pixel_size / abs(y_pixel_scale) * (206265.0 / (3600 * 1000))
    return (fl_x + fl_y) / 2.0
