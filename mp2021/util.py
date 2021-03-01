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



