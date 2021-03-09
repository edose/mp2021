__author__ = "Eric Dose, Albuquerque"

""" This module: Workflow for MP (minor planet) photometry. 
    The workflow is applied to a "session" = one MP's images from one imaging night.
    Intended for lightcurves in support of determining MP rotation rates.    
"""

# Python core:
import os
from datetime import datetime, timezone, timedelta
from collections import Counter, OrderedDict

# External packages:
import astropy.io.fits as apyfits
import numpy as np
import pandas as pd

# Author's packages:
import mp2021.util as util
import mp2021.ini as ini
from astropak.image import FITS, aggregate_bounding_ra_dec, PointSourceAp, MovingSourceAp
from astropak.catalogs import Refcat2
from astropak.util import RaDec, jd_from_datetime_utc


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')

FOCAL_LENGTH_MAX_PCT_DEVIATION = 1.0


class SessionIniFileError(Exception):
    """ Raised on any fatal problem with session ini file."""


class SessionLogFileError(Exception):
    """ Raised on any fatal problem with session log file."""


class SessionDataError(Exception):
    """ Raised on any fatal problem with data, esp. with contents of FITS files."""


def start(session_top_directory=None, mp_id=None, an_date=None, filter=None):
    # Adapted from mp_phot workflow_session.start().
    """ Launch one session of MP photometry workflow.
        Adapted from package mp_phot, workflow_session.py.start().
        Example usage: session.start('C:/Astro/MP Photometry/', 1111, 20200617, 'Clear')
    :param session_top_directory: path of lowest directory common to all MP lightcurve FITS, e.g.,
               'C:/Astro/MP Photometry'. None will use .ini file default (normal case). [string]
    :param mp_id: either a MP number, e.g., 1602 for Indiana [integer or string], or for an id string
               for unnumbered MPs only, e.g., ''. [string only]
    :param an_date: Astronight date representation, e.g., '20191106'. [integer or string]
    :param filter: name of filter for this session, or None to use default from instrument file. [string]
    :return: [None]
    """
    defaults_dict = ini.make_defaults_dict()
    if session_top_directory is None:
        session_top_directory = defaults_dict['session top directory']
    if mp_id is None or an_date is None:
        print(' >>>>> Usage: start(top_directory, mp_id, an_date)')
        return
    mp_string = util.parse_mp_id(mp_id)
    an_string = util.parse_an_date(an_date)

    # Construct directory path, and make it the working directory:
    mp_directory = os.path.join(session_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    os.chdir(mp_directory)
    print('Working directory set to:', mp_directory)

    # Initiate (or overwrite) log file:
    log_filename = defaults_dict['session log filename']
    with open(log_filename, mode='w') as log_file:
        log_file.write('Session Log File.' + '\n')
        log_file.write(mp_directory + '\n')
        log_file.write('MP: ' + mp_string + '\n')
        log_file.write('AN: ' + an_string + '\n')
        log_file.write('FILTER:' + filter + '\n')
        log_file.write('This log started: ' +
                       '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    print('Log file started.')
    print('Next: assess()')


def resume(session_top_directory=None, mp_id=None, an_date=None, filter=None):
    # Adapted from package mpc, mp_phot.assess().
    """ Restart a workflow in its correct working directory,
        but *keep* the previous log file--DO NOT overwrite it.
        Adapted from package mp_phot, workflow_session.py.resume().
        Example usage: session.resume('C:/Astro/MP Photometry/', 1111, 20200617, 'Clear')
    Parameters are exactly as for .start().
    :return: [None]
    """
    defaults_dict = ini.make_defaults_dict()
    if session_top_directory is None:
        session_top_directory = defaults_dict['session top directory']
    if mp_id is None or an_date is None:
        print(' >>>>> Usage: resume(top_directory, mp_id, an_date)')
        return
    mp_string = util.parse_mp_id(mp_id).upper()
    an_string = util.parse_an_date(an_date)

    # Construct directory path and make it the working directory:
    this_directory = os.path.join(session_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    os.chdir(this_directory)
    print('Working directory set to:', this_directory)
    log_this_directory, log_mp_string, log_an_string, log_filter_string = _get_session_context()
    if log_mp_string.upper() != mp_string:
        raise SessionLogFileError(' '.join(['MP string does not match that of session log.',
                                            log_mp_string, mp_string]))
    if log_an_string != an_string:
        raise SessionLogFileError(' '.join(['AN string does not match that of session log.',
                                            log_an_string, an_string]))
    print('Resuming in', this_directory)


def assess(return_results=False):
    """  First, verify that all required files are in the working directory or otherwise accessible.
         Then, perform checks on FITS files in this directory before performing the photometry proper.
         Modeled after and extended from assess() found in variable-star photometry package 'photrix',
             then adapted from package mpc, mp_phot.assess()
    :return: [None], or dict of summary info and warnings. [py dict]
    """
    # Setup, including initializing return_dict:
    # (Can't use orient_this_function(), because session.ini may not exist yet.)
    this_directory, mp_string, an_string, filter_string = _get_session_context()
    defaults_dict = ini.make_defaults_dict()
    return_dict = {
        'file not read': [],         # list of filenames
        'filter not read': [],       # "
        'file count by filter': [],  # list of tuples (filter, file count)
        'warning count': 0,          # total count of all warnings.
        'not platesolved': [],       # list of filenames
        'not calibrated': [],        # "
        'unusual fwhm': [],          # list of tuples (filename, fwhm)
        'unusual focal length': []}  # list of tuples (filename, focal length)

    # Count FITS files by filter, write totals
    #    (we've stopped classifying files by intention; now we include all valid FITS in dfs):
    filter_counter = Counter()
    valid_fits_filenames = []
    all_fits_filenames = util.get_mp_filenames(this_directory)
    for filename in all_fits_filenames:
        fullpath = os.path.join(this_directory, filename)
        try:
            hdu = apyfits.open(fullpath)[0]
        except FileNotFoundError:
            print(' >>>>> WARNING: can\'t find file', fullpath, 'Skipping file.')
            return_dict['file not read'].append(filename)
        except (OSError, UnicodeDecodeError):
            print(' >>>>> WARNING: can\'t read file', fullpath, 'as FITS. Skipping file.')
            return_dict['file not read'].append(filename)
        else:
            fits_filter = util.fits_header_value(hdu, 'FILTER')
            if fits_filter is None:
                print(' >>>>> WARNING: filter in', fullpath, 'cannot be read. Skipping file.')
                return_dict['filter not read'].append(filename)
            else:
                valid_fits_filenames.append(filename)
                filter_counter[fits_filter] += 1
    for filter in filter_counter.keys():
        print('   ' + str(filter_counter[filter]), 'in filter', filter + '.')
        return_dict['file count by filter'].append((filter, filter_counter[filter]))

    # Start dataframe for main FITS integrity checks:
    fits_extensions = pd.Series([os.path.splitext(f)[-1].lower() for f in valid_fits_filenames])
    df = pd.DataFrame({'Filename': valid_fits_filenames,
                       'Extension': fits_extensions.values}).sort_values(by=['Filename'])
    df = df.set_index('Filename', drop=False)
    df['PlateSolved'] = False
    df['Calibrated'] = False
    df['FWHM'] = np.nan
    df['FocalLength'] = np.nan

    # Populate df with FITS header info needed for validity tests below:
    for filename in df.index:
        fullpath = os.path.join(this_directory, filename)
        hdu = apyfits.open(fullpath)[0]  # already known to be valid, from above.
        df.loc[filename, 'PlateSolved'] = util.fits_is_plate_solved(hdu)
        df.loc[filename, 'Calibrated'] = util.fits_is_calibrated(hdu)
        df.loc[filename, 'FWHM'] = util.fits_header_value(hdu, 'FWHM')
        df.loc[filename, 'FocalLength'] = util.fits_focal_length(hdu)
        jd_start = util.fits_header_value(hdu, 'JD')
        exposure = util.fits_header_value(hdu, 'EXPOSURE')
        jd_mid = jd_start + (exposure / 2) / 24 / 3600
        df.loc[filename, 'JD_mid'] = jd_mid  # needed only to write session.ini stub.

    # Warn of FITS without plate solution:
    filenames_not_platesolved = df.loc[~ df['PlateSolved'], 'Filename']
    if len(filenames_not_platesolved) >= 1:
        print('NO PLATE SOLUTION:')
        for fn in filenames_not_platesolved:
            print('    ' + fn)
            return_dict['not platesolved'].append(fn)
        print('\n')
    else:
        print('All platesolved.')
    return_dict['warning count'] += len(filenames_not_platesolved)

    # Warn of FITS without calibration:
    filenames_not_calibrated = df.loc[~ df['Calibrated'], 'Filename']
    if len(filenames_not_calibrated) >= 1:
        print('\nNOT CALIBRATED:')
        for fn in filenames_not_calibrated:
            print('    ' + fn)
            return_dict['not calibrated'].append(fn)
        print('\n')
    else:
        print('All calibrated.')
    return_dict['warning count'] += len(filenames_not_calibrated)

    # Warn of FITS with very large or very small FWHM:
    odd_fwhm_list = []
    instrument_dict = ini.make_instrument_dict(defaults_dict)
    # settings = Settings()
    min_fwhm = 0.5 * instrument_dict['nominal fwhm pixels']
    max_fwhm = 2.0 * instrument_dict['nominal fwhm pixels']
    for fn in df['Filename']:
        fwhm = df.loc[fn, 'FWHM']
        if fwhm < min_fwhm or fwhm > max_fwhm:  # too small or large:
            odd_fwhm_list.append((fn, fwhm))
    if len(odd_fwhm_list) >= 1:
        print('\nUnusual FWHM (in pixels):')
        for fn, fwhm in odd_fwhm_list:
            print('    ' + fn + ' has unusual FWHM of ' + '{0:.2f}'.format(fwhm) + ' pixels.')
            return_dict['unusual fwhm'].append((fn, fwhm))
        print('\n')
    else:
        print('All FWHM values seem OK.')
    return_dict['warning count'] += len(odd_fwhm_list)

    # Warn of FITS with abnormal Focal Length:
    odd_fl_list = []
    median_fl = df['FocalLength'].median()
    for fn in df['Filename']:
        fl = df.loc[fn, 'FocalLength']
        focal_length_pct_deviation = 100.0 * abs((fl - median_fl)) / median_fl
        if focal_length_pct_deviation > FOCAL_LENGTH_MAX_PCT_DEVIATION:
            odd_fl_list.append((fn, fl))
    if len(odd_fl_list) >= 1:
        print('\nUnusual FocalLength (vs median of ' + '{0:.1f}'.format(median_fl) + ' mm:')
        for fn, fl in odd_fl_list:
            print('    ' + fn + ' has unusual Focal length of ' + str(fl))
            return_dict['unusual focal length'].append((fn, fl))
        print('\n')
    else:
        print('All Focal Lengths seem OK.')
    return_dict['warning count'] += len(odd_fl_list)

    # Summarize and write instructions for user's next steps:
    session_ini_filename = defaults_dict['session control filename']
    session_log_filename = defaults_dict['session log filename']
    session_log_fullpath = os.path.join(this_directory, session_log_filename)
    with open(session_log_fullpath, mode='w') as log_file:
        if return_dict['warning count'] == 0:
            print('\n >>>>> ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.')
            print('Next: (1) enter MP pixel positions in', session_ini_filename,
                  'AND SAVE it,\n      (2) measure_mp()')
            log_file.write('assess(): ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.' + '\n')
        else:
            print('\n >>>>> ' + str(return_dict['warning count']) + ' warnings (see listing above).')
            print('        Correct these and rerun assess() until no warnings remain.')
            log_file.write('assess(): ' + str(return_dict['warning count']) + ' warnings.' + '\n')

    df_temporal = df.loc[:, ['Filename', 'JD_mid']].sort_values(by=['JD_mid'])
    filenames_temporal_order = df_temporal['Filename']
    _write_session_ini_stub(this_directory, filenames_temporal_order)  # if it doesn't already exist.
    if return_results:
        return return_dict


def make_dfs():
    """ Perform aperture photometry for one session of lightcurve photometry only.
        For color index determination, see .make_color_dfs().
        """
    context, defaults_dict, session_dict, log_file = _session_setup('make_dfs()')
    this_directory, mp_string, an_string, filter_string = context
    fits_filenames = util.get_mp_filenames(this_directory)
    if not fits_filenames:
        raise SessionDataError('No FITS files found in session directory ' + this_directory)

    # Validate MP XY filenames:
    mp_location_filenames = [mp_xy_entry[0] for mp_xy_entry in session_dict['mp xy']]
    if any([fn not in fits_filenames for fn in mp_location_filenames]):
        raise SessionIniFileError('A MP XY file is missing from session directory ' + this_directory)

    # Assemble valid, time-sorted FITS objects into (1) a list and (2) an OrderedDict by filename:
    all_fits_objects = [FITS(this_directory, '', fn) for fn in fits_filenames]
    invalid_fits_filenames = [fo.filename for fo in all_fits_objects if not fo.is_valid]
    if invalid_fits_filenames:
        for fn in invalid_fits_filenames:
            print(' >>>>> WARNING: Invalid FITS filename ' + fn +
                  ' is being skipped. User should explicitly exclude it.')
    fits_objects = [fo for fo in all_fits_objects if fo.is_valid]
    fits_objects = sorted(fits_objects, key=lambda fo: fo.utc_mid)
    fits_object_dict = OrderedDict((fo.filename, fo) for fo in fits_objects)

    # Get ATLAS refcat2 comp stars covering bounding region of all images:
    aggr_ra_deg_min, aggr_ra_deg_max, aggr_dec_deg_min, aggr_dec_deg_max = \
        aggregate_bounding_ra_dec(fits_objects, extension_percent=3)
    refcat2 = Refcat2(ra_deg_range=(aggr_ra_deg_min, aggr_ra_deg_max),
                      dec_deg_range=(aggr_dec_deg_min, aggr_dec_deg_max))
    info_lines = screen_comps_for_photometry(refcat2, session_dict)
    utc_mids = [fo.utc_mid for fo in fits_objects]
    utc_mid_session = min(utc_mids) + (max(utc_mids) - min(utc_mids)) / 2
    refcat2.update_epoch(utc_mid_session)
    print('\n'.join(info_lines), '\n')
    log_file.write('\n'.join(info_lines), '\n')

    # Make comp star apertures (dict of lists of ap objects):
    instrument_dict = ini.make_instrument_dict(defaults_dict)
    disc_radius = 1.8 * instrument_dict['nominal fwhm pixels']
    gap = 1.2 * instrument_dict['nominal fwhm pixels']
    outer_radius = 4.0 * instrument_dict['nominal fwhm pixels']
    background_width = outer_radius - disc_radius - gap
    ap_radec_centers = [RaDec(ra, dec)
                        for (ra, dec) in zip(refcat2.df_selected['RA_deg'], refcat2.df_selected['Dec_deg'])]
    comp_apertures = OrderedDict()
    for fo in fits_objects:
        xy_centers = [fo.xy_from_radec(radec) for radec in ap_radec_centers]
        ap_list = [PointSourceAp(fo.image, xy, disc_radius, gap, background_width) for xy in xy_centers]
        comp_apertures[fo.filename] = ap_list

    # Make MP apertures (dict of ap objects, one ap per image):
    utc0, ra0, dec0, ra_per_second, dec_per_second = \
        calc_mp_motion(session_dict, fits_object_dict, log_file)
    mp_apertures = OrderedDict()
    for fo in fits_objects:
        dt_start = (fo.utc_start - utc0).total_seconds()
        dt_end = dt_start + fo.exposure
        ra_start = ra0 + dt_start * ra_per_second
        ra_end = ra0 + dt_end * ra_per_second
        dec_start = dec0 + dt_start * dec_per_second
        dec_end = dec0 + dt_end * dec_per_second
        xy_start = fo.xy_from_radec(RaDec(ra_start, dec_start))
        xy_end = fo.xy_from_radec(RaDec(ra_end, dec_end))
        ap = MovingSourceAp(fo.image, xy_start, xy_end, disc_radius, gap, background_width)
        mp_apertures[fo.filename] = ap

    # Gather data for df_image and df_obs:
    image_dict_list = []
    df_image_obs_list = []
    # TODO: Resume here after straightening out Ap class and subclasses (sigh).











_____SUPPORT_FUNCTIONS________________________________________ = 0


def _get_session_context():
    """ This is run at beginning of workflow functions (except start() or resume()) to orient the function.
        Assumes python current working directory = the relevant AN subdirectory with session.log in place.
        Adapted from package mp_phot, workflow_session.py._get_session_context(). Required for .resume().
        TESTED OK 2021-01-08.
    :return: 4-tuple: (this_directory, mp_string, an_string, filter_string) [4 strings]
    """
    this_directory = os.getcwd()
    defaults_dict = ini.make_defaults_dict()
    session_log_filename = defaults_dict['session log filename']
    session_log_fullpath = os.path.join(this_directory, session_log_filename)
    if not os.path.isfile(session_log_fullpath):
        raise SessionLogFileError('No log file found. You probably need to run start() or resume().')
    with open(session_log_fullpath, mode='r') as log_file:
        lines = log_file.readlines()
    if len(lines) < 5:
        raise SessionLogFileError('Too few lines.')
    if not lines[0].lower().startswith('session log file'):
        raise SessionLogFileError('Header line cannot be parsed.')
    directory_from_session_log = lines[1].strip().lower().replace('\\', '/').replace('//', '/')
    directory_from_cwd = this_directory.strip().lower().replace('\\', '/').replace('//', '/')
    if directory_from_session_log != directory_from_cwd:
        print()
        print(directory_from_session_log, directory_from_cwd)
        raise SessionLogFileError('Header line does not match current working directory.')
    mp_string = lines[2][3:].strip().upper()
    an_string = lines[3][3:].strip()
    filter_string = lines[4][7:].strip()
    return this_directory, mp_string, an_string, filter_string


def _write_session_ini_stub(this_directory, filenames_temporal_order):
    """ Write session's initial control (.ini) file, later to be edited by user.
        Called only by (at the end of) .assess().  DO NOT overwrite if session.ini exists.
    :param this_directory:
    :param filenames_temporal_order: FITS filenames in ascending time order. [list of strings]
    :return:
    """
    # Do not overwrite existing session ini file:
    defaults_dict = ini.make_defaults_dict()
    session_ini_filename = defaults_dict['session control filename']
    fullpath = os.path.join(this_directory, session_ini_filename)
    if os.path.exists(fullpath):
        return

    filename_earliest = filenames_temporal_order[0]
    filename_latest = filenames_temporal_order[-1]
    header_lines = [
        '# This is ' + fullpath + '.',
        '']

    ini_lines = [
        '[Ini Template]',
        'Filename = session.template',
        '']
    bulldozer_lines = [
        '[Bulldozer]',
        '# At least 3 ref star XY, one per line, all from one FITS only if at all possible:',
        'Ref Star XY = ' + filename_earliest + ' 000.0  000.0',
        '              ' + filename_earliest + ' 000.0  000.0',
        '              ' + filename_earliest + ' 000.0  000.0',
        '# Exactly 2 MP XY, one per line (typically earliest and latest FITS):',
        'MP XY = ' + filename_earliest + ' 000.0  000.0',
        '        ' + filename_latest + ' 000.0  000.0',
        '']
    selection_criteria_lines = [
        '[Selection Criteria]',
        'Omit Comps =',
        'Omit Obs =',
        'Omit Images =',
        'Min Catalog r mag = 10.0',
        'Max Catalog r mag = 16.0',
        'Max Catalog dr mmag = 15.0',
        'Min Catalog ri color = 0.04',
        'Max Catalog ri color = 0.40',
        '']
    regression_lines = [
        '[Regression]',
        'MP ri color = +0.220',
        '# Fit Transform, one of: Fit=1, Fit=2, Use [val1], Use [val1] [val2]:',
        'Fit Transform = Use +0.4 -0.6',
        '# Fit Extinction, one of: Yes, Use [val]:',
        'Fit Extinction = Use +0.16',
        'Fit Vignette = Yes',
        'Fit XY = No',
        'Fit JD = Yes']
    raw_lines = header_lines + ini_lines + bulldozer_lines + selection_criteria_lines + regression_lines
    ready_lines = [line + '\n' for line in raw_lines]
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(ready_lines)
    print('New ' + session_ini_filename + ' file written.\n')


def _session_setup(calling_function_name='[FUNCTION NAME NOT GIVEN]'):
    """ Typically called at the top of lightcurve workflow functions, to collect commonly required data.
    :return: tuple of data elements: context [tuple], defaults_dict [py dict], log_file [file object].
    """
    context = _get_session_context()
    if context is None:
        return
    this_directory, mp_string, an_string, filter_string = context
    defaults_dict = ini.make_defaults_dict()
    session_dict = ini.make_session_dict()
    log_filename = defaults_dict['session log filename']
    log_file = open(log_filename, mode='a')  # set up append to log file.
    log_file.write('\n===== ' + calling_function_name + '()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    return context, defaults_dict, session_dict, log_file


def screen_comps_for_photometry(refcat2, session_dict):
    """ Applies ATLAS refcat2 screens to refcat2 object IN-PLACE. Returns info text.
    :param refcat2: ATLAS refcat2 catalog from astropak.catalog.py. [Refcat2 object]
    :param session_dict:
    :return info: text documenting actions taken. [list of strings]
    """
    info = []
    refcat2.select_min_r_mag(session_dict['min catalog r mag'])
    info.append('Refcat2: min(g) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_max_r_mag(session_dict['max catalog r mag'])
    info.append('Refcat2: max(g) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    # refcat2.select_max_g_uncert(MAX_G_UNCERT)
    # info.append('Refcat2: max(dg) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_max_r_uncert(session_dict['max catalog dr mmag'])
    info.append('Refcat2: max(dr) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    # refcat2.select_max_i_uncert(MAX_I_UNCERT)
    # info.append('Refcat2: max(di) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_sloan_ri_color(session_dict['min catalog ri color'],
                                  session_dict['max catalog ri color'])
    info.append('Refcat2: Sloan ri color screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    # refcat2.select_dgaia()
    # lines.append('Refcat2: dgaia screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.remove_overlapping()
    info.append('Refcat2: overlaps removed to ' + str(len(refcat2.df_selected)) + ' stars.')
    return info


def calc_mp_motion(session_dict, fits_object_dict, log_file):
    """ From two user-selected images and x,y locations, return data sufficient to locate MP in all images.
    :param session_dict:
    :param fits_object_dict:
    :param log_file: log file object ready for use in write(), or None not to write to log file.
    :return: utc0, ra0, dec0, ra_per_second, dec_per_second [tuple of floats]
    """
    # Get MP's RA and Dec in user-selected images:
    mp_location_filenames = [item[0] for item in session_dict['mp xy']][:2]
    mp_location_fits_objects = [fits_object_dict[fn] for fn in mp_location_filenames]
    mp_location_xy = [(item[1], item[2]) for item in session_dict['mp xy']][:2]
    mp_datetime, mp_ra_deg, mp_dec_deg = [], [], []
    for i in range(2):
        fo = mp_location_fits_objects[i]
        x, y = mp_location_xy[i][0], mp_location_xy[i][1]
        radec = fo.radec_from_xy(x, y)
        mp_datetime.append(fo.utc_mid)
        mp_ra_deg.append(radec.ra)
        mp_dec_deg.append(radec.dec)

    # Calculate MP reference location and motion:
    utc0, ra0, dec0 = mp_datetime[0], mp_ra_deg[0], mp_dec_deg[0]
    span_seconds = (mp_datetime[1] - utc0).total_seconds()
    ra_per_second = (mp_ra_deg[1] - ra0) / span_seconds
    dec_per_second = (mp_dec_deg[1] - dec0) / span_seconds
    if log_file is not None:
        log_file.write('MP at JD ' + '{0:.5f}'.format(jd_from_datetime_utc(utc0)) + ':  RA,Dec='
                       + '{0:.5f}'.format(ra0) + u'\N{DEGREE SIGN}' + ', '
                       + '{0:.5f}'.format(dec0) + u'\N{DEGREE SIGN}' + ',  d(RA,Dec)/hour='
                       + '{0:.6f}'.format(ra_per_second * 3600.0) + ', '
                       + '{0:.6f}'.format(dec_per_second * 3600.0) + '\n')
    return utc0, ra0, dec0, ra_per_second, dec_per_second


