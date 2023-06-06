__author__ = "Eric Dose, Albuquerque"

""" This module: Utilities in service of other mp2021 modules. """

# Python core:
import os
from datetime import datetime, timezone, timedelta
from math import floor

# External packages:
import pandas as pd

# Author's packages:
import mp2021.ini as ini


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')

EARLIEST_APPLICABLE_AN = 20000000
LATEST_APPLICABLE_AN = 21000000
VALID_FITS_FILE_EXTENSIONS = ('.fits', '.fit', '.fts')

SOURCE_RADIUS_IN_FWHM = 1.8
AP_GAP_IN_FWHM = 0.75
BACKGROUND_WIDTH_IN_FWHM = 1.2

CURRENT_MPFILE_VERSION = '1.1'


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
            if calibration_value.strip().upper() in ('BDF', 'DF'):
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


class Instrument:
    """ Contains data (from instrument_dict) for instrument and does calculations dependent on it."""
    def __init__(self, instrument_dict):
        self.instrument_dict = instrument_dict
        self.x_size = instrument_dict['x pixels']
        self.y_size = instrument_dict['y pixels']
        self.x_center = self.x_size / 2
        self.y_center = self.y_size / 2
        self.dist2_at_corner = self.x_center ** 2 + self.y_center ** 2
        self.vignetting_drop_at_1024 = (instrument_dict['vignetting pct at corner'] / 100.0) * \
                                       ((1024 ** 2) / self.dist2_at_corner)
        self.saturation_adu = instrument_dict['saturation adu']

    @property
    def gain(self):
        return self.instrument_dict.get('ccd gain', 1)

    @property
    def nominal_ap_profile(self):
        source_radius = SOURCE_RADIUS_IN_FWHM * self.instrument_dict['nominal fwhm pixels']
        gap = AP_GAP_IN_FWHM * self.instrument_dict['nominal fwhm pixels']
        background_width = BACKGROUND_WIDTH_IN_FWHM * self.instrument_dict['nominal fwhm pixels']
        return source_radius, gap, background_width

    def x1024(self, x):
        return (x - self.x_center) / 1024.0

    def y1024(self, y):
        return (y - self.y_center) / 1024.0

    def is_saturated(self, x, y, adu):
        """ Return True if (vignetting-corrected = in original image) adu was probably saturated. """
        dist2_at_xy = self.x1024(x) ** 2 + self.y1024(y) ** 2
        expected_vignette_drop = self.vignetting_drop_at_1024 * (dist2_at_xy / (1024 ** 2))
        pixel_saturated = adu >= (self.saturation_adu * (1.0 - expected_vignette_drop))
        return pixel_saturated


class MPfile:
    """ One object contains all current-apparition (campaign) data for one MP.
    Fields:
        .format_version [str, currently '1.0']
        .number: MP number [str representing an integer]
        .name: text name of MP, e.g., 'Dido' or '1952 TX'. [str]
        .family: MP family and family code. [str]
        .apparition: identifier (usually year) of this apparition, e.g., '2020'. [str]
        .motive: special reason to do photometry, or 'Pet' if simply a favorite. [str]
        .period: expected rotational period, in hours. [float]
        .period_certainty: LCDB certainty code, e.g., '1' or '2-'. [str]
        .amplitude: expected amplitude, in magnitudes. [float]
        .priority: priority code, 0=no priority, 10=top priority, 6=normal. [int]
        .brightest_utc: given date that MP is brightest, this apparition. [python datetime UTC]
        .eph_range: first & last date within the ephemeris (not observations). [2-tuple of datetime UTC]
        .obs_jd_ranges: list of previous observation UTC ranges. [list of lists of floats]
        .eph_dict_list: One dict per MPC ephemeris time (which are all at 00:00 UTC). [list of dicts]
            dict elements:
                'DateString': UTC date string for this MPC ephemeris line. [str as yyyy-mm-dd]
                'DatetimeUTC': UTC date. [py datetime object]
                'RA': right ascension, in degrees (0-360). [float]
                'Dec': declination, in degrees (-90-+90). [float]
                'Delta': distance Earth (observatory) to MP, in AU. [float]
                'R': distance Sun to MP, in AU. [float]
                'Elong': MP elongation from Sun, in degrees (0-180). [float]
                'Phase': Phase angle Sun-MP-Earth, in degrees. [float]
                'V_mag': Nominal V magnitude. [float]
                'MotionRate': MP speed across sky, in arcsec/minute. [float]
                'MotionDirection': MP direction across sky, in degrees, from North=0 toward East. [float]
                'PAB_longitude': phase angle bisector longitude, in degrees. [float]
                'PAB_latitude': phase angle bisector latitude, in degrees. [float]
                'MoonPhase': -1 to 1, where neg=waxing, 0=full, pos=waning. [float]
                'MoonDistance': Moon-MP distance in sky, in degrees. [float]
                'Galactic_longitude': in degrees. [float]
                'Galactic_latitude': in degrees. [float]
        .df_eph: the same data as in eph_dict_list, with dict keys becoming column names,
                    row index=DateUTC string. [pandas Dataframe]
        .is_valid: True iff all data looks OK. [boolean]
    """
    def __init__(self, mpfile_name, mpfile_directory=None):
        defaults_dict = ini.make_defaults_dict()
        if mpfile_directory is None:
            mpfile_directory = defaults_dict['mpfile directory']
        mpfile_fullpath = os.path.join(mpfile_directory, mpfile_name)
        if os.path.exists(mpfile_fullpath) and os.path.isfile(mpfile_fullpath):
            with open(mpfile_fullpath) as mpfile:
                lines = mpfile.readlines()
            self.is_valid = True  # conditional on parsing in rest of __init__()
        else:
            print('>>>>> MP file \'' + mpfile_fullpath + '\' not found. MPfile object invalid.')
            self.is_valid = False
            return
        lines = [line.split(";")[0] for line in lines]  # remove all comments.
        lines = [line.strip() for line in lines]  # remove leading and trailing whitespace.

        # ---------- Header section:
        self.format_version = MPfile._directive_value(lines, '#VERSION')
        if self.format_version != CURRENT_MPFILE_VERSION:
            print(' >>>>> ERROR: ' + mpfile_name + ':  Version Error. MPfile object invalid.')
            self.is_valid = False
            return
        self.number = self._directive_value(lines, '#MP')
        self.name = self._directive_value(lines, '#NAME')
        if self.name is None:
            print(' >>>>> Warning: Name is missing. (MP=' + self.number + ')')
            self.name = None
        self.family = self._directive_value(lines, '#FAMILY')
        self.apparition = self._directive_value(lines, '#APPARITION')
        self.motive = self._directive_value(lines, '#MOTIVE')
        words = self._directive_words(lines, '#PERIOD')
        if words is not None:
            try:
                self.period = float(words[0])
            except ValueError:
                # print(' >>>>> Warning: Period present but non-numeric,'
                # '[None] stored. (MP=' + self.number + ')')
                self.period = None
            if len(words) >= 2:
                self.period_certainty = words[1]
            else:
                self.period_certainty = '?'
        amplitude_string = self._directive_value(lines, '#AMPLITUDE')
        if amplitude_string is None:
            print(' >>>>> Warning: Amplitude is missing. [None] stored. (MP=' + self.number + ')')
            self.amplitude = None
        else:
            try:
                self.amplitude = float(amplitude_string)
            except ValueError:
                # print(' >>>>> Warning: Amplitude present but non-numeric,'
                # '[None] stored. (MP=' + self.number + ')')
                self.amplitude = None
        priority_string = self._directive_value(lines, '#PRIORITY')
        try:
            self.priority = int(priority_string)
        except ValueError:
            print(' >>>>> ERROR: Priority present but incorrect. (MP=' + self.number + ')')
            self.priority = None

        brightest_string = self._directive_value(lines, '#BRIGHTEST')
        try:
            year_str, month_str, day_str = tuple(brightest_string.split('-'))
            self.brightest_utc = datetime(int(year_str), int(month_str),
                                          int(day_str)).replace(tzinfo=timezone.utc)
        except ValueError:
            print(' >>>>> ERROR: Brightest incorrect. (MP=' + self.number + ')')
            self.brightest_utc = None
        eph_range_strs = self._directive_words(lines, '#EPH_RANGE')[:2]
        # self.utc_range = self._directive_words(lines, '#EPH_RANGE')[:2]
        self.eph_range = []
        for utc_str in eph_range_strs[:2]:
            year_str, month_str, day_str = tuple(utc_str.split('-'))
            utc_dt = datetime(int(year_str), int(month_str), int(day_str)).replace(tzinfo=timezone.utc)
            self.eph_range.append(utc_dt)

        # ---------- Observations (already made) section:
        obs_strings = [line[len('#OBS'):].strip() for line in lines if line.upper().startswith('#OBS')]
        obs_jd_range_strs = [value.split() for value in obs_strings]  # nested list of strings (not floats)
        self.obs_jd_ranges = []
        for range in obs_jd_range_strs:
            if len(range) >= 2:
                self.obs_jd_ranges.append([float(range[0]), float(range[1])])
            else:
                print(' >>>>> ERROR: missing #OBS field for MP', self.number, self.name)

        # ---------- Ephemeris section:
        eph_dict_list = []
        i_eph_directive = None
        for i, line in enumerate(lines):
            if line.upper().startswith('#EPHEMERIS'):
                i_eph_directive = i
                break
        if ((not (lines[i_eph_directive + 1].startswith('==========')) or
             (not lines[i_eph_directive + 3].strip().startswith('UTC')) or
             (not lines[i_eph_directive + 4].strip().startswith('----------')))):
            print(' >>>>> ERROR: ' + mpfile_name +
                  ':  MPEC header doesn\'t match expected from minorplanet.info page.')
            self.is_valid = False
            return
        eph_lines = lines[i_eph_directive + 5:]
        for line in eph_lines:
            eph_dict = dict()
            words = line.split()
            eph_dict['DateString'] = words[0]
            date_parts = words[0].split('-')
            eph_dict['DatetimeUTC'] = datetime(year=int(date_parts[0]),
                                               month=int(date_parts[1]),
                                               day=int(date_parts[2])).replace(tzinfo=timezone.utc)
            eph_dict['RA'] = 15.0 * (float(words[1]) + float(words[2]) / 60.0 + float(words[3]) / 3600.0)
            dec_sign = -1 if words[4].startswith('-') else 1.0
            dec_abs_value = abs(float(words[4])) + float(words[5]) / 60.0 + float(words[6]) / 3600.0
            eph_dict['Dec'] = dec_sign * dec_abs_value
            eph_dict['Delta'] = float(words[7])           # earth-MP, in AU
            eph_dict['R'] = float(words[8])               # sun-MP, in AU
            eph_dict['Elong'] = float(words[9])           # from sun, in degrees
            eph_dict['Phase'] = float(words[10])          # phase angle, in degrees
            eph_dict['V_mag'] = float(words[11])
            eph_dict['MotionRate'] = float(words[12])     # MP speed in arcseconds per minute.
            eph_dict['MotionAngle'] = float(words[13])    # MP direction, from North=0 toward East.
            eph_dict['PAB_longitude'] = float(words[14])  # phase angle bisector longitude, in degrees
            eph_dict['PAB_latitude'] = float(words[15])   # phase angle bisector latitude, in degrees
            eph_dict['MoonPhase'] = float(words[16])      # -1 to 1, where neg is waxing, pos is waning.
            eph_dict['MoonDistance'] = float(words[17])   # in degrees from MP
            eph_dict['Galactic_longitude'] = float(words[18])  # in degrees
            eph_dict['Galactic_latitude'] = float(words[19])   # in degrees
            eph_dict_list.append(eph_dict)
        self.eph_dict_list = eph_dict_list
        self.df_eph = pd.DataFrame(data=eph_dict_list)
        self.df_eph.index = self.df_eph['DatetimeUTC'].values
        self.is_valid = True

    @staticmethod
    def _directive_value(lines, directive_string, default_value=None):
        for line in lines:
            if line.upper().startswith(directive_string):
                return line[len(directive_string):].strip()
        return default_value  # if directive absent.

    def _directive_words(self, lines, directive_string):
        value = self._directive_value(lines, directive_string, default_value=None)
        if value is None:
            return None
        return value.split()

    def eph_from_utc(self, datetime_utc):
        """ Interpolate data from mpfile object's ephemeris; return dict, or None if bad datetime input.
            Current code requires that ephemeris line spacing spacing = 1 day.
        :param datetime_utc: target utc date and time. [python datetime object]
        :return: dict of results specific to this MP and datetime, or None if bad datetime input. [dict]
        """
        mpfile_first_date_utc = self.eph_dict_list[0]['DatetimeUTC']
        i = (datetime_utc - mpfile_first_date_utc).total_seconds() / 24 / 3600  # a float.
        if not(0 <= i < len(self.eph_dict_list) - 1):  # i.e., if outside date range of eph table.
            return None
        return_dict = dict()
        i_floor = int(floor(i))
        i_fract = i - i_floor
        for k in self.eph_dict_list[0].keys():
            value_before, value_after = self.eph_dict_list[i_floor][k], self.eph_dict_list[i_floor + 1][k]
            # Add interpolated value if not a string;
            #    (use this calc form, because you can subtract but not add datetime objects):
            if isinstance(value_before, datetime) or isinstance(value_before, float):
                return_dict[k] = value_before + i_fract * (value_after - value_before)  # interpolated val.
        return return_dict


def all_mpfile_names(mpfile_directory):
    """ Returns list of all MPfile names (from filenames in mpfile_directory). """
    mpfile_names = [fname for fname in os.listdir(mpfile_directory)
                    if (fname.endswith(".txt")) and (fname.startswith("MP_"))]
    return mpfile_names