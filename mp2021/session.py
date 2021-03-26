__author__ = "Eric Dose, Albuquerque"

from mp2021.util import Instrument

""" This module: Workflow for MP (minor planet) photometry. 
    The workflow is applied to a "session" = one MP's images from one imaging night.
    Intended for lightcurves in support of determining MP rotation rates.    
"""

# Python core:
import os
from datetime import datetime, timezone
from collections import Counter, OrderedDict
from math import floor, log10, log, sin, ceil

# External packages:
import astropy.io.fits as apyfits
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astroplan import Observer
import numpy as np
import pandas as pd
# import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm


# Author's packages:
import mp2021.util as util
import mp2021.ini as ini
from astropak.image import FITS, aggregate_bounding_ra_dec, PointSourceAp, MovingSourceAp
from astropak.catalogs import Refcat2
from astropak.util import RaDec, jd_from_datetime_utc, datetime_utc_from_jd, ra_as_hours, dec_as_hex
from astropak.reference import DEGREES_PER_RADIAN
from astropak.stats import MixedModelFit


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')

FOCAL_LENGTH_MAX_PCT_DEVIATION = 1.0
MINIMUM_COMP_OBS_COUNT = 5

# The following apply only to initial screening of comps from catalog to dataframe(s):
INITIAL_MIN_R_MAG = 10
INITIAL_MAX_R_MAG = 16
INITIAL_MAX_DR_MMAG = 20
INITIAL_MAX_DI_MMAG = 20
INITIAL_MIN_RI_COLOR = 0.00
INITIAL_MAX_RI_COLOR = 0.44

ALCDEF_BASE_DATA = {'contactname': 'Eric V. Dose',
                    'contactinfo': 'MP@ericdose.com',
                    'observers': 'Dose, E.V.',
                    'filter': 'C',
                    'magband': 'SR'}
SESSION_PLOT_FILE_PREFIX = 'Image_Session_'


class SessionIniFileError(Exception):
    """ Raised on any fatal problem with session ini file."""


class SessionLogFileError(Exception):
    """ Raised on any fatal problem with session log file."""


class SessionDataError(Exception):
    """ Raised on any fatal problem with data, esp. with contents of FITS files or missing data."""


class SessionSpecificationError(Exception):
    """ Raised on any fatal problem in specifying the session or processing, esp. in _make_df_all(). """


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
                       '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n\n')
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
    try:
        context = _get_session_context()
    except SessionLogFileError as e:
        print(' >>>>> ERROR: ' + str(e))
        return
    this_directory, mp_string, an_string, filter_string = context
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
    with open(session_log_fullpath, mode='a') as log_file:
        if return_dict['warning count'] == 0:
            print('\n >>>>> ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.')
            print('Next: (1) enter MP pixel positions in', session_ini_filename,
                  'AND SAVE it,\n      (2) make_dfs()')
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
    context, defaults_dict, session_dict, log_file = _session_setup('make_dfs')
    this_directory, mp_string, an_string, filter_string = context
    instrument_dict = ini.make_instrument_dict(defaults_dict)
    instrument = Instrument(instrument_dict)
    disc_radius, gap, background_width = instrument.nominal_ap_profile
    site_dict = ini.make_site_dict(defaults_dict)

    fits_filenames = util.get_mp_filenames(this_directory)
    if not fits_filenames:
        raise SessionDataError('No FITS files found in session directory ' + this_directory)

    # Validate MP XY filenames:
    mp_location_filenames = [mp_xy_entry[0] for mp_xy_entry in session_dict['mp xy']]
    if any([fn not in fits_filenames for fn in mp_location_filenames]):
        raise SessionIniFileError('At least 1 MP XY file not found in session directory ' + this_directory)

    fits_objects, fits_object_dict = make_fits_objects(this_directory, fits_filenames)
    df_images = _make_df_images(fits_objects)

    # Get and screen catalog entries for comp stars:
    refcat2 = get_refcat2_comp_stars(fits_objects)
    info_lines = initial_screen_comps(refcat2)  # in-place screening.
    print('\n'.join(info_lines), '\n')
    log_file.write('\n'.join(info_lines) + '\n')

    # Make comp-star apertures, comps dataframe, and comp obs dataframe:
    df_comps = _make_df_comps(refcat2)
    comp_apertures_dict = _make_comp_apertures(fits_objects, df_comps, disc_radius, gap, background_width)
    df_comp_obs = _make_df_comp_obs(comp_apertures_dict, df_comps, instrument, df_images)

    # Make MP apertures and MP obs dataframe:
    starting_mp_obs_id = max(df_comp_obs['ObsID'].astype(int)) + 1
    mp_apertures_dict, mp_mid_radec_dict = _make_mp_apertures(fits_object_dict, mp_string, session_dict,
                                                              disc_radius, gap, background_width, log_file,
                                                              starting_obs_id=starting_mp_obs_id)
    df_mp_obs = _make_df_mp_obs(mp_apertures_dict, mp_mid_radec_dict, instrument, df_images)

    # Post-process dataframes:
    _remove_images_without_mp_obs(fits_object_dict, df_images, df_comp_obs, df_mp_obs)
    _add_obsairmass_df_comp_obs(df_comp_obs, site_dict, df_comps, df_images)
    _add_obsairmass_df_mp_obs(df_mp_obs, site_dict, df_images)
    _add_ri_color_df_comps(df_comps)

    # Write dataframes to CSV files:
    _write_df_images_csv(df_images, this_directory, defaults_dict, log_file)
    _write_df_comps_csv(df_comps, this_directory, defaults_dict, log_file)
    _write_df_comp_obs_csv(df_comp_obs, this_directory, defaults_dict, log_file)
    _write_df_mp_obs_csv(df_mp_obs, this_directory, defaults_dict, log_file)

    log_file.close()
    print('\nNext: (1) enter comp selection limits and model options in ' +
          defaults_dict['session control filename'],
          '\n      (2) run do_session()\n')


def do_session():
    """ Primary lightcurve photometry for one session. Takes all data incl. color index, generates:
    Takes the 4 CSV files from make_dfs().
    Generates:
    * diagnostic plots for iterative regression refinement,
    * results in Canopus-import format,
    * ALCDEF-format file.
    Typically iterated, pruning comp-star ranges and outliers, until converged and then simply stop.
    NB: One may choose the FITS files by filter (typically 'Clear' or 'BB'), but
        * output lightcurve passband is fixed as Sloan 'r', and
        * color index is fixed as Sloan (r-i).
    :returns None. Writes all info to files.
    USAGE: do_session()   [no return value]
    """
    context, defaults_dict, session_dict, log_file = _session_setup('do_session')
    this_directory, mp_string, an_string, filter_string = context
    log_filename = defaults_dict['session log filename']
    log_file = open(log_filename, mode='a')
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    site_dict = ini.make_site_dict(defaults_dict)
    # log_file.write('\n===== do_session(' + filter_string + ')  ' +
    #                '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')

    df_comp_master, df_mp_master = _make_df_masters(filters_to_include=filter_string,
                                                    require_mp_obs_each_image=True)
    df_model = _make_df_model(df_comp_master)  # comps only.
    df_model = _mark_user_selections(df_model, session_dict)
    model = SessionModel(df_model, filter_string, session_dict, df_mp_master, this_directory)

    _write_mpfile_line(mp_string, an_string, model)
    _write_canopus_file(mp_string, an_string, this_directory, model)
    _write_alcdef_file(mp_string, an_string, defaults_dict, session_dict, site_dict, this_directory, model)
    _make_session_diagnostic_plots(model, df_model)


_____SUPPORT_for_make_dfs_____________________________________________ = 0


def _make_df_images(fits_objects):
    earliest_jd_mid = min([jd_from_datetime_utc(fo.utc_mid) for fo in fits_objects])
    jd_floor = floor(earliest_jd_mid)
    image_dict_list = []
    for fo in fits_objects:
        jd_mid = jd_from_datetime_utc(fo.utc_mid)
        jd_fract = jd_mid - jd_floor
        image_dict = {
            'FITSfile': fo.filename,
            'JD_mid': jd_mid,
            'Filter': fo.filter,
            'Exposure': fo.exposure,
            'Airmass': fo.airmass,  # will be overwritten by per-obs airmass.
            'JD_start': jd_from_datetime_utc(fo.utc_start),
            'UTC_start': fo.utc_start,
            'UTC_mid': fo.utc_mid,
            'JD_fract': jd_fract}
        image_dict_list.append(image_dict)
    df_image = pd.DataFrame(data=image_dict_list).sort_values(by='JD_mid')
    df_image.index = df_image['FITSfile'].values
    return df_image


def _make_df_comps(refcat2):
    df_comps = refcat2.selected_columns(['RA_deg', 'Dec_deg', 'RP1', 'R1', 'R10',
                                         'g', 'dg', 'r', 'dr', 'i', 'di', 'z', 'dz',
                                         'BminusV', 'APASS_R', 'T_eff', 'CatalogID'])
    comp_ids = [str(i + 1) for i in range(len(df_comps))]
    df_comps.index = comp_ids
    df_comps.insert(0, 'CompID', comp_ids)
    return df_comps


def _make_comp_apertures(fits_objects, df_comps, disc_radius, gap, background_width):
    comp_radec_dict = {comp_id: RaDec(df_comps.loc[comp_id, 'RA_deg'], df_comps.loc[comp_id, 'Dec_deg'])
                       for comp_id in df_comps.index}
    comp_apertures_dict = OrderedDict()
    obs_id = 1
    for fo in fits_objects:
        ap_list = []
        for comp_id in df_comps.index:
            xy = fo.xy_from_radec(comp_radec_dict[comp_id])
            # NB: fo.image_fits is (y,x) as Ap expects. Do not use .image or .image_xy which are (x,y).
            raw_ap = PointSourceAp(fo.image_fits, xy, disc_radius, gap, background_width,
                                   str(comp_id), str(obs_id))
            if raw_ap.is_valid and raw_ap.all_inside_image:  # severest screen available from PointSourceAp.
                ap = raw_ap.recenter(max_iterations=3)
                if ap.is_valid and ap.all_inside_image:
                    ap_list.append(ap)
                    obs_id += 1
                    if obs_id % 10 == 0:
                        print('.', end='', flush=True)
                    if obs_id % 1000 == 0:
                        print('{:6d} raw apertures done'.format(obs_id), flush=True)
        comp_apertures_dict[fo.filename] = ap_list
    print()
    return comp_apertures_dict


def _make_df_comp_obs(comp_apertures, df_comps, instrument, df_images):
    gain = instrument.gain
    comp_obs_dict_list = []
    for filename in comp_apertures.keys():
        exposure = df_images.loc[filename, 'Exposure']
        for ap in comp_apertures[filename]:
            if ap.is_valid and ap.net_flux > 0:
                x1024 = instrument.x1024(ap.xy_center.x)
                y1024 = instrument.y1024(ap.xy_center.y)
                vignette = x1024 ** 2 + y1024 ** 2
                any_pixel_saturated = instrument.is_saturated(ap.xy_center.x, ap.xy_center.y,
                                                              ap.foreground_max)
                if not any_pixel_saturated:
                    comp_obs_dict = {
                        'FITSfile': filename,
                        'CompID': ap.source_id,
                        'ObsID': ap.obs_id,
                        'Type': 'Comp',
                        'InstMag': -2.5 * log10(ap.net_flux / exposure),
                        'InstMagSigma': (2.5 / log(10)) * (ap.flux_stddev(gain) / ap.net_flux),
                        'DiscRadius': ap.foreground_radius,
                        'SkyRadiusInner': ap.foreground_radius + ap.gap,
                        'SkyRadiusOuter': ap.foreground_radius + ap.gap + ap.background_width,
                        'FWHM': ap.fwhm,
                        'Elongation': ap.elongation,
                        'SkyADU': ap.background_level,
                        'SkySigma': ap.background_std,
                        'Vignette': vignette,
                        'X1024': x1024,
                        'Y1024': y1024,
                        'Xcentroid': ap.xy_centroid[0],
                        'Ycentroid': ap.xy_centroid[1],
                        'RA_deg': df_comps.loc[ap.source_id, 'RA_deg'],
                        'Dec_deg': df_comps.loc[ap.source_id, 'Dec_deg']}
                    comp_obs_dict_list.append(comp_obs_dict)
    df_comp_obs = pd.DataFrame(data=comp_obs_dict_list)
    df_comp_obs.index = df_comp_obs['ObsID'].values
    df_comp_obs = df_comp_obs.sort_values(by='ObsID', key=lambda ids: ids.astype('int64'))
    return df_comp_obs


def _make_mp_apertures(fits_object_dict, mp_string, session_dict,
                       disc_radius, gap, background_width, log_file, starting_obs_id):
    """ Make mp_apertures, one row per MP (thus one row per FITS file).
    :param fits_object_dict:
    :param mp_string:
    :param session_dict:
    :param disc_radius:
    :param gap:
    :param background_width:
    :param log_file:
    :param starting_obs_id: [int]
    :return:
    """
    utc0, ra0, dec0, ra_per_second, dec_per_second = \
        calc_mp_motion(session_dict, fits_object_dict, log_file)
    mp_apertures_dict = OrderedDict()
    mp_mid_radec_dict = OrderedDict()
    obs_id = starting_obs_id
    for filename, fo in fits_object_dict.items():
        dt_start = (fo.utc_start - utc0).total_seconds()
        dt_end = dt_start + fo.exposure
        ra_start = ra0 + dt_start * ra_per_second
        ra_end = ra0 + dt_end * ra_per_second
        dec_start = dec0 + dt_start * dec_per_second
        dec_end = dec0 + dt_end * dec_per_second
        xy_start = fo.xy_from_radec(RaDec(ra_start, dec_start))
        xy_end = fo.xy_from_radec(RaDec(ra_end, dec_end))
        # NB: fo.image_fits is (y,x) as Ap expects. Do not use .image or .image_xy which are (x,y).
        raw_ap = MovingSourceAp(fo.image_fits, xy_start, xy_end, disc_radius, gap, background_width,
                                source_id=mp_string, obs_id=str(obs_id))
        if raw_ap.is_valid and raw_ap.all_inside_image:  # severest screen available from PointSourceAp.
            ap = raw_ap.recenter(max_iterations=3)
            if ap.is_valid and ap.all_inside_image:
                # *************** START temporary TEST CODE:
                # # xy_mid = (xy_start[0] + xy_end[0]) / 2, (xy_start[1] + xy_end[1]) / 2
                # round_ap = PointSourceAp(fo.image_fits, ap.xy_center, disc_radius, gap,
                #                          background_width, source_id=mp_string, obs_id=str(obs_id))
                # print(filename, '{:.0f}'.format(ap.net_flux), '{:.0f}'.format(round_ap.net_flux),
                #       '{:.4f}'.format(ap.net_flux / round_ap.net_flux), '(using PointSourceAp)')
                # ap.net_flux = round_ap.net_flux
                # *************** END temporary TEST CODE
                mp_apertures_dict[fo.filename] = ap
                ra_mid = (ra_start + ra_end) / 2
                dec_mid = (dec_start + dec_end) / 2
                mp_mid_radec_dict[filename] = RaDec(ra_mid, dec_mid)
                obs_id += 1
    return mp_apertures_dict, mp_mid_radec_dict


def _make_df_mp_obs(mp_apertures, mp_mid_radec_dict, instrument, df_images):
    gain = instrument.gain
    mp_obs_dict_list = []
    for filename, ap in mp_apertures.items():
        exposure = df_images.loc[filename, 'Exposure']
        if ap.is_valid and ap.net_flux > 0:
            x1024 = instrument.x1024(ap.xy_center.x)
            y1024 = instrument.y1024(ap.xy_center.y)
            vignette = x1024 ** 2 + y1024 ** 2
            any_pixel_saturated = instrument.is_saturated(ap.xy_center.x, ap.xy_center.y,
                                                          ap.foreground_max)
            if not any_pixel_saturated:
                mp_obs_dict = {
                    'FITSfile': filename,
                    'MP_ID': 'MP_' + ap.source_id,
                    'ObsID': ap.obs_id,
                    'Type': 'MP',
                    'InstMag': -2.5 * log10(ap.net_flux / exposure),
                    'InstMagSigma': (2.5 / log(10)) * (ap.flux_stddev(gain) / ap.net_flux),
                    'SkyADU': ap.background_level,
                    'SkySigma': ap.background_std,
                    'DiscRadius': ap.foreground_radius,
                    'SkyRadiusInner': ap.foreground_radius + ap.gap,
                    'SkyRadiusOuter': ap.foreground_radius + ap.gap + ap.background_width,
                    'FWHM': ap.fwhm,
                    'Elongation': ap.elongation,
                    'Vignette': vignette,
                    'X1024': x1024,
                    'Y1024': y1024,
                    'Xcentroid': ap.xy_centroid[0],
                    'Ycentroid': ap.xy_centroid[1],
                    'Xstart': ap.xy_start.x,
                    'Ystart': ap.xy_start.y,
                    'Xend': ap.xy_end.x,
                    'Yend': ap.xy_end.y,
                    'RA_deg_mid': mp_mid_radec_dict[filename].ra,
                    'Dec_deg_mid': mp_mid_radec_dict[filename].dec}
                mp_obs_dict_list.append(mp_obs_dict)
    df_mp_obs = pd.DataFrame(data=mp_obs_dict_list)
    df_mp_obs.index = df_mp_obs['FITSfile'].values
    df_mp_obs = df_mp_obs.sort_values(by='ObsID', key=lambda ids: ids.astype('int64'))
    return df_mp_obs


def _remove_images_without_mp_obs(fits_object_dict, df_images, df_comp_obs, df_mp_obs):
    for filename in fits_object_dict.keys():
        mp_obs_count = sum(df_mp_obs['FITSfile'] == filename)
        if mp_obs_count != 1:
            df_images = df_images.loc[df_images['FITSfile'] != filename, :]
            df_comp_obs = df_comp_obs.loc[df_comp_obs['FITSfile'] != filename, :]
            df_mp_obs = df_mp_obs.loc[df_mp_obs['FITSfile'] != filename, :]


def _add_obsairmass_df_comp_obs(df_comp_obs, site_dict, df_comps, df_images):
    observer = Observer(longitude=site_dict['longitude'] * u.deg,
                        latitude=site_dict['latitude'] * u.deg,
                        elevation=site_dict['elevation'] * u.m)
    df_comp_obs['ObsAirmass'] = None
    skycoord_dict = {comp_id: SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
                     for (comp_id, ra_deg, dec_deg)
                     in zip(df_comps.index, df_comps['RA_deg'], df_comps['Dec_deg'])}
    altaz_frame_dict = {filename: observer.altaz(Time(jd_mid, format='jd'))
                        for (filename, jd_mid) in zip(df_images['FITSfile'], df_images['JD_mid'])}
    print('ObsAirmasses: ', end='', flush=True)
    done_count = 0
    for obs, filename, comp_id in zip(df_comp_obs.index, df_comp_obs['FITSfile'], df_comp_obs['CompID']):
        alt = skycoord_dict[comp_id].transform_to(altaz_frame_dict[filename]).alt.value
        df_comp_obs.loc[obs, 'ObsAirmass'] = 1.0 / sin(alt / DEGREES_PER_RADIAN)
        done_count += 1
        if done_count % 100 == 0:
            print('.', end='', flush=True)
    print('\nObsAirmasses written to df_comp_obs:', str(len(df_comp_obs)))


def _add_obsairmass_df_mp_obs(df_mp_obs, site_dict, df_images):
    observer = Observer(longitude=site_dict['longitude'] * u.deg,
                        latitude=site_dict['latitude'] * u.deg,
                        elevation=site_dict['elevation'] * u.m)
    df_mp_obs['ObsAirmass'] = None
    skycoord_dict = {filename: SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
                     for (filename, ra_deg, dec_deg) in zip(df_mp_obs['FITSfile'],
                                                            df_mp_obs['RA_deg_mid'],
                                                            df_mp_obs['Dec_deg_mid'])}
    altaz_frame_dict = {filename: observer.altaz(Time(jd_mid, format='jd'))
                        for (filename, jd_mid) in zip(df_images['FITSfile'], df_images['JD_mid'])}
    for obs, filename in zip(df_mp_obs.index, df_mp_obs['FITSfile']):
        alt = skycoord_dict[filename].transform_to(altaz_frame_dict[filename]).alt.value
        df_mp_obs.loc[obs, 'ObsAirmass'] = 1.0 / sin(alt / DEGREES_PER_RADIAN)
    print('ObsAirmasses written to df_mp_obs:', str(len(df_mp_obs)))


def _add_ri_color_df_comps(df_comps):
    df_comps['ri_color'] = df_comps['r'] - df_comps['i']



_____SUPPORT_for_do_session______________________________________________ = 0


def _make_df_masters(filters_to_include=None, require_mp_obs_each_image=True):
    """ Get, screen and merge dataframes df_images_all, df_comps_all, df_comp_obs_all, and df_mp_obs_all
        into two master dataframes dataframe df_comp_master and df_mp_master.
    :param filters_to_include: either one filter name, or a list of filters.
               Only observations in that filter or filters will be retained.
               None includes ALL filters given in input dataframes [None, or string, or list of strings]
    :param require_mp_obs_each_image: True to remove all obs from images without MP observation. [boolean]
    :return: df_comp_master, df_mp_master, the two master tables of data, one row per retained observation.
                 [2-tuple of pandas DataFrames]
"""
    if isinstance(filters_to_include, str):
        filters_to_include = [filters_to_include]
    context, defaults_dict, session_dict, log_file = _session_setup('do_session()')
    this_directory, mp_string, an_string, filter_string = context

    df_images_all = _read_session_csv(this_directory, defaults_dict['df_images filename'])
    df_comps_all = _read_session_csv(this_directory, defaults_dict['df_comps filename'],
                                     dtype_dict={'CompID': str})
    df_comp_obs_all = _read_session_csv(this_directory, defaults_dict['df_comp_obs filename'],
                                        dtype_dict={'CompID': str, 'ObsID': str})
    df_mp_obs_all = _read_session_csv(this_directory, defaults_dict['df_mp_obs filename'],
                                      dtype_dict={'ObsID': str})

    # Keep only rows in specified filters:
    image_rows_to_keep = df_images_all['Filter'].isin(filters_to_include)
    df_images = df_images_all.loc[image_rows_to_keep, :]
    if len(df_images) <= 0:
        raise SessionDataError('No images found in specified filter(s): ' + str(filters_to_include))
    comp_obs_rows_to_keep = df_comp_obs_all['FITSfile'].isin(df_images['FITSfile'])
    df_comp_obs = df_comp_obs_all.loc[comp_obs_rows_to_keep, :]
    if len(df_comp_obs) <= 0:
        raise SessionDataError('No comp obs found in specified filters(s)' + str(filters_to_include))
    mp_obs_rows_to_keep = df_mp_obs_all['FITSfile'].isin(df_images['FITSfile'])
    df_mp_obs = df_mp_obs_all.loc[mp_obs_rows_to_keep, :]
    if len(df_mp_obs) <= 0:
        raise SessionDataError('No MP obs found in specified filters(s)' + str(filters_to_include))

    # Remove images and all obs from images having no MP obs, if requested:
    if require_mp_obs_each_image:
        images_with_mp_obs = df_mp_obs['FITSfile'].drop_duplicates()
        image_rows_to_keep = df_images['FITSfile'].isin(images_with_mp_obs)
        df_images = df_images.loc[image_rows_to_keep, :]
        comp_obs_rows_to_keep = df_comp_obs['FITSfile'].isin(images_with_mp_obs)
        df_comp_obs = df_comp_obs.loc[comp_obs_rows_to_keep, :]
        mp_obs_rows_to_keep = df_mp_obs['FITSfile'].isin(images_with_mp_obs)
        df_mp_obs = df_mp_obs.loc[mp_obs_rows_to_keep, :]

    # Perform merges to produce master comp dataframes:
    df_comp_obs = pd.merge(left=df_comp_obs, right=df_images, how='left', on='FITSfile')
    df_comp_master = pd.merge(left=df_comp_obs, right=df_comps_all.drop(columns=['RA_deg', 'Dec_deg']),
                              how='left', on='CompID')
    df_comp_master.index = df_comp_master['ObsID'].values
    df_mp_master = pd.merge(left=df_mp_obs, right=df_images, how='left', on='FITSfile')
    df_mp_master.index = df_mp_master['ObsID'].values
    return df_comp_master, df_mp_master


def _read_session_csv(this_directory, filename, dtype_dict=None):
    """ Simple utility to read specified CSV file into a pandas Dataframe. """
    fullpath = os.path.join(this_directory, filename)
    df = pd.read_csv(fullpath, sep=';', index_col=0, header=0, dtype=dtype_dict)
    df.index = df.index.astype(str)
    return df


def _make_df_model(df_comp_master):
    """ Assemble and return df_model, the comp-only dataframe containing all input data need for the
        mixed-model regression at the center of this lightcurve workflow.
        Keep only comps that are present in every image.
    :param df_comp_master:
    :return: df_model, one row per comp observation [pandas Dataframe]
    """
    # Count comp obs in each image:
    comp_id_list = df_comp_master['CompID'].drop_duplicates()
    image_count = len(df_comp_master['FITSfile'].drop_duplicates())
    comp_obs_count_each_image = df_comp_master.groupby('CompID')[['FITSfile', 'CompID']].count()
    comp_ids_in_every_image = [id for id in comp_id_list
                               if comp_obs_count_each_image.loc[id, 'FITSfile'] == image_count]
    rows_with_qualified_comp_ids = df_comp_master['CompID'].isin(comp_ids_in_every_image)
    df_model = df_comp_master.loc[rows_with_qualified_comp_ids, :]
    return df_model


def _mark_user_selections(df_model, session_dict):
    """ Add UseInModel column to df_model, to be True for each row iff row (obs) passes all user criteria
        allowing it to be actually used in constructing the mixed-model.
    :param df_model: [pandas DataFrame]
    :return: df_model with new UseInModel column. [pandas DataFrame]
    """
    deselect_for_obs_id = df_model['ObsID'].isin(session_dict['omit obs'])
    deselect_for_comp_id = df_model['CompID'].isin(session_dict['omit comps'])
    images_to_omit = session_dict['omit images'] + [name + '.fts' for name in session_dict['omit images']]
    deselect_for_image = df_model['FITSfile'].isin(images_to_omit)
    deselect_for_low_r_mag = (df_model['r'] < session_dict['min catalog r mag'])
    deselect_for_high_r_mag = (df_model['r'] > session_dict['max catalog r mag'])
    deselect_for_high_dr_mmag = (df_model['dr'] > session_dict['max catalog dr mmag'])
    deselect_for_high_di_mmag = (df_model['di'] > session_dict['max catalog di mmag'])
    deselect_for_low_ri_color = (df_model['ri_color'] < session_dict['min catalog ri color'])
    deselect_for_high_ri_color = (df_model['ri_color'] > session_dict['max catalog ri color'])
    obs_to_deselect = pd.Series(deselect_for_obs_id | deselect_for_comp_id | deselect_for_image
                                | deselect_for_low_r_mag | deselect_for_high_r_mag
                                | deselect_for_high_dr_mmag | deselect_for_high_di_mmag
                                | deselect_for_low_ri_color | deselect_for_high_ri_color)
    df_model.loc[:, 'UseInModel'] = ~ obs_to_deselect
    return df_model


class SessionModel:
    """ Generates and holds mixed-model regression model, affords prediction for MP magnitudes. """
    def __init__(self, df_model, filter_string, session_dict, df_mp_master, this_directory):
        self.df_model = df_model
        self.filter_string = filter_string
        self.session_dict = session_dict
        self.df_used_comps_obs = self.df_model.copy().loc[self.df_model['UseInModel'], :]
        images_in_used_comps = self.df_used_comps_obs['FITSfile'].drop_duplicates()
        mp_rows_to_use = df_mp_master['FITSfile'].isin(images_in_used_comps)
        self.df_used_mp_obs = df_mp_master.loc[mp_rows_to_use, :]
        self.this_directory = this_directory

        self.dep_var_name = 'InstMag_with_offsets'
        self.mm_fit = None      # placeholder for the fit result [MixedModelFit object].
        self.transform = None   # placeholder for this fit parameter result [scalar].
        self.transform_fixed = None  # "
        self.extinction = None  # "
        self.vignette = None    # "
        self.x = None           # "
        self.y = None           # "
        self.jd1 = None         # "

        self._prep_and_do_regression()
        self.df_mp_mags = self._calc_mp_mags()

    def _prep_and_do_regression(self):
        """ Using MixedModelFit class (which wraps statsmodels.MixedLM.from_formula()).
            Use ONLY selected comp data in the model itself.
            Use model's .predict() to calculate best MP mags from model and MP observations.
        :return: [None]
        """
        fit_summary_lines = []
        dep_var_offset = self.df_used_comps_obs['r'].copy()
        fixed_effect_var_list = []

        # Handle transform (Color Index) option:
        # Options: ('fit', '1'), ('fit', '2'), ('use', [Tr1]), ('use', [Tr1], [Tr2]).
        self.df_used_comps_obs['CI'] = self.df_used_comps_obs['ri_color']
        self.df_used_comps_obs['CI2'] = [ci ** 2 for ci in self.df_used_comps_obs['CI']]
        transform_option = self.session_dict['fit transform']
        if transform_option == ('fit', '1'):
            fixed_effect_var_list.append('CI')
            msg = ' Transform first-order fit to Color Index.'
        elif transform_option == ('fit', '2'):
            fixed_effect_var_list.extend(['CI', 'CI2'])
            msg = ' Transform second-order fit to Color Index.'
        elif transform_option[0] == 'use':
            if len(transform_option) == 2:
                transform_offset = float(transform_option[1]) * self.df_used_comps_obs['CI']
            elif len(transform_option) == 3:
                transform_offset = float(transform_option[1]) * self.df_used_comps_obs['CI'] +\
                                   float(transform_option[2]) * self.df_used_comps_obs['CI2']
            else:
                raise SessionSpecificationError('Invalid \'Fit Transform\' option in session.ini')
            dep_var_offset += transform_offset
            msg = ' Transform (Color Index) not fit: 1st, 2nd order values fixed at' + \
                  transform_option[1] + transform_option[2]
        else:
            raise SessionSpecificationError('Invalid \'Fit Transform\' option in session.ini')

        # Handle extinction (ObsAirmass) option:
        # Option chosen from: 'yes', ('use', [ext]).
        extinction_option = self.session_dict['fit extinction']
        if extinction_option == 'yes':
            fixed_effect_var_list.append('ObsAirmass')
            msg = ' Extinction fit on ObsAirmass data.'
        elif isinstance(extinction_option, tuple) and len(extinction_option) == 2 and \
            (extinction_option[0] == 'use'):
            extinction_offset = float(extinction_option[1]) * self.df_used_comps_obs['ObsAirmass']
            dep_var_offset += extinction_offset
            msg = ' Extinction (ObsAirmass) not fit: value fixed at default of' + \
                  ' {0:.3f}'.format(float(extinction_option[1]))
        else:
            raise SessionSpecificationError('Invalid \'Fit Extinction\' option in session.ini')

        # Build all other fixed-effect (x) variable lists and dep-var offsets:
        if self.session_dict['fit vignette']:
            fixed_effect_var_list.append('Vignette')
        if self.session_dict['fit xy']:
            fixed_effect_var_list.extend(['X1024', 'Y1024'])
        if self.session_dict['fit jd']:
            fixed_effect_var_list.append('JD_fract')
        if len(fixed_effect_var_list) == 0:
            fixed_effect_var_list = ['Vignette']  # as statsmodels requires >= 1 fixed-effect variable.

        # Build 'random-effect' and dependent (y) variables:
        random_effect_var_name = 'FITSfile'  # cirrus effect is per-image
        self.df_used_comps_obs[self.dep_var_name] = self.df_used_comps_obs['InstMag'] - dep_var_offset

        # Execute regression:
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)
        self.mm_fit = MixedModelFit(data=self.df_used_comps_obs,
                                    dep_var=self.dep_var_name,
                                    fixed_vars=fixed_effect_var_list,
                                    group_var=random_effect_var_name)
        n_comps_used = len(self.df_used_comps_obs['CompID'].drop_duplicates())
        print(self.mm_fit.statsmodels_object.summary())
        print('comps =', str(n_comps_used), ' used.')
        print('sigma =', '{0:.1f}'.format(1000.0 * self.mm_fit.sigma), 'mMag.')
        if not self.mm_fit.converged:
            msg = ' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.'
            print(msg)
            fit_summary_lines.append(msg)

        write_text_file(self.this_directory, 'fit_summary.txt',
                        'Regression (mp2021) for: ' + self.this_directory + '\n\n' +
                        '\n'.join(fit_summary_lines) + '\n\n' +
                        self.mm_fit.statsmodels_object.summary().as_text() +
                        '\ncomps = ' + str(n_comps_used) + ' used' +
                        '\nsigma = ' + '{0:.1f}'.format(1000.0 * self.mm_fit.sigma) + ' mMag.')

    def _calc_mp_mags(self):
        """ Use model and MP instrument magnitudes to get best estimates of MP absolute magnitudes."""
        bogus_cat_mag = 0.0  # we'll need this below, to correct raw predictions.
        self.df_used_mp_obs['CatMag'] = bogus_cat_mag  # totally bogus local value, corrected for later.
        best_mp_ri_color = self.session_dict['mp ri color']
        self.df_used_mp_obs['CI'] = best_mp_ri_color
        self.df_used_mp_obs['CI2'] = best_mp_ri_color ** 2
        raw_predictions = self.mm_fit.predict(self.df_used_mp_obs, include_random_effect=True)
        dep_var_offset = pd.Series(len(self.df_used_mp_obs) * [0.0], index=raw_predictions.index)

        # Handle transform offsets for MPs:
        transform_option = self.session_dict['fit transform']
        if transform_option[0] == 'use':
            if len(transform_option) == 2:
                transform_offset = float(transform_option[1]) * self.df_used_mp_obs['CI']
            else:  # len(transform_option) == 3 (as previously vetted).
                transform_offset = float(transform_option[1]) * self.df_used_mp_obs['CI'] + \
                                   float(transform_option[2]) * self.df_used_mp_obs['CI2']
            dep_var_offset += transform_offset

        # Handle extinction offsets for MPs:
        extinction_option = self.session_dict['fit extinction']
        if isinstance(extinction_option, tuple):  # will be ('use', [ext]), as previously vetted.
            extinction_offset = float(extinction_option[1]) * self.df_used_mp_obs['ObsAirmass']
            dep_var_offset += extinction_offset

        # Calculate best MP magnitudes, incl. effect of assumed bogus_cat_mag:
        mp_mags = self.df_used_mp_obs['InstMag'] - dep_var_offset - raw_predictions + bogus_cat_mag
        df_mp_mags = pd.DataFrame(data={'MP_Mags': mp_mags}, index=list(mp_mags.index))
        df_mp_mags = pd.merge(left=df_mp_mags,
                              right=self.df_used_mp_obs.loc[:, ['JD_mid', 'FITSfile',
                                                                'InstMag', 'InstMagSigma']],
                              how='left', left_index=True, right_index=True, sort=False)
        return df_mp_mags


def _write_mpfile_line(mp_string, an_string, model):
    """ Write to console one line for this session. (Paste line into this MP campaign's MPfile.) """
    model_jds = model.df_mp_mags['JD_mid']
    print(' >>>>> Please add this line to MPfile', mp_string + ':',
          '  #OBS', '{0:.5f}'.format(model_jds.min()), ' {0:.5f}'.format(model_jds.max()),
          ' ;', an_string, '::', mp_string)


def _write_canopus_file(mp_string, an_string, this_directory, model):
    """ Write MP results in format that can be imported directly into Canopus. """
    df = model.df_mp_mags
    fulltext = '\n'.join([','.join(['{0:.6f}'.format(jd), '{0:.4f}'.format(mag), '{0:.4f}'.format(s), f])
                          for (jd, mag, s, f) in zip(df['JD_mid'], df['MP_Mags'],
                                                     df['InstMagSigma'], df['FITSfile'])])
    fullpath = os.path.join(this_directory, 'canopus_MP_' + mp_string + '_' + an_string + '.txt')
    with open(fullpath, 'w') as f:
        f.write(fulltext)


def all_mpfile_names(mpfile_directory):
    """ Returns list of all MPfile names (from filenames in mpfile_directory). """
    mpfile_names = [fname for fname in os.listdir(mpfile_directory)
                    if (fname.endswith(".txt")) and (fname.startswith("MP_"))]
    return mpfile_names


def _write_alcdef_file(mp_string, an_string, defaults_dict, session_dict, site_dict, this_directory, model):
    mpfile_names = all_mpfile_names(defaults_dict['mpfile directory'])
    name_list = [name for name in mpfile_names if name.startswith('MP_' + mp_string + '_')]
    if len(name_list) <= 0:
        print(' >>>>> WARNING: No MPfile can be found for MP', mp_string, '--> NO ALCDEF file written')
        return
    if len(name_list) >= 2:
        print(' >>>>> WARNING: Multiple MPfiles were found for MP', mp_string, '--> NO ALCDEF file written')
        return
    mpfile = util.MPfile(name_list[0])
    df = model.df_mp_mags

    # Build data that will go into file:
    lines = list()
    lines.append('# ALCDEF file for MP ' + mp_string + '  AN ' + an_string + ' (mp2021 workflow)')
    lines.append('STARTMETADATA')
    lines.append('REVISEDDATA=FALSE')
    lines.append('OBJECTNUMBER=' + mp_string)
    lines.append('OBJECTNAME=' + mpfile.name)
    lines.append('ALLOWSHARING=TRUE')
    # lines.append('MPCDESIG=')
    lines.append('CONTACTNAME=' + ALCDEF_BASE_DATA['contactname'])
    lines.append('CONTACTINFO=' + ALCDEF_BASE_DATA['contactinfo'])
    lines.append('OBSERVERS=' + ALCDEF_BASE_DATA['observers'])
    if site_dict['longitude'] <= 180:
        alcdef_longitude = site_dict['longitude']
    else:
        alcdef_longitude = site_dict['longitude'] - 360.0
    lines.append('OBSLONGITUDE=' + '{0:.4f}'.format(alcdef_longitude))
    lines.append('OBSLATITUDE=' + '{0:.4f}'.format(site_dict['latitude']))
    lines.append('FACILITY=' + site_dict['name'])
    lines.append('MPCCODE=' + site_dict['mpc code'])
    # lines.append('PUBLICATION=')
    jd_session_start = min(df['JD_mid'])
    jd_session_end = max(df['JD_mid'])
    jd_session_mid = (jd_session_start + jd_session_end) / 2.0
    utc_session_mid = datetime_utc_from_jd(jd_session_mid)  # (needed below)
    dt_split = utc_session_mid.isoformat().split('T')
    lines.append('SESSIONDATE=' + dt_split[0])
    lines.append('SESSIONTIME=' + dt_split[1].split('+')[0].split('.')[0])
    lines.append('FILTER=' + ALCDEF_BASE_DATA['filter'])
    lines.append('MAGBAND=' + ALCDEF_BASE_DATA['magband'])
    lines.append('LTCTYPE=NONE')
    # lines.append('LTCDAYS=0')
    # lines.append('LTCAPP=NONE')
    lines.append('REDUCEDMAGS=NONE')
    session_eph_dict = mpfile.eph_from_utc(utc_session_mid)
    # earth_mp_au = session_eph_dict['Delta']
    # sun_mp_au = session_eph_dict['R']
    # reduced_mag_correction = -5.0 * log10(earth_mp_au * sun_mp_au)
    #  lines.append('UCORMAG=' + '{0:.4f}'.format(reduced_mag_correction))  # removed to avoid confusion.
    lines.append('OBJECTRA=' + ra_as_hours(session_eph_dict['RA']).rsplit(':', 1)[0])
    lines.append('OBJECTDEC=' + ' '.join(dec_as_hex(round(session_eph_dict['Dec'])).split(':')[0:2]))
    lines.append('PHASE=+' + '{0:.1f}'.format(abs(session_eph_dict['Phase'])))
    lines.append('PABL=' + '{0:.1f}'.format(abs(session_eph_dict['PAB_longitude'])))
    lines.append('PABB=' + '{0:.1f}'.format(abs(session_eph_dict['PAB_latitude'])))
    lines.append('COMMENT=These results from submitter\'s ATLAS-refcat2 based workflow')
    lines.append('COMMENT=as described in SAS Symposium 2020, using code at github.com/edose/mp2021.')
    lines.append('COMMENT=This session used ' +
                 str(len(model.df_used_comps_obs['CompID'].drop_duplicates())) +
                 ' comp stars. COMPNAME etc lines are omitted.')
    lines.append('CICORRECTION=TRUE')
    lines.append('CIBAND=SRI')
    lines.append('CITARGET=' + '{0:+.3f}'.format(session_dict['mp ri color']) +
                 '  # origin: ' + session_dict['mp ri color origin'])
    lines.append('DELIMITER=PIPE')
    lines.append('ENDMETADATA')
    data_lines = ['DATA=' + '|'.join(['{0:.6f}'.format(jd), '{0:.3f}'.format(mag), '{0:.3f}'.format(sigma)])
                  for (jd, mag, sigma) in zip(df['JD_mid'], df['MP_Mags'], df['InstMagSigma'])]
    lines.extend(data_lines)
    lines.append('ENDDATA')

    # Write the file and exit:
    fulltext = '\n'.join(lines) + '\n'
    fullpath = os.path.join(this_directory, 'alcdef_MP_' + mp_string + '_' + an_string + '.txt')
    with open(fullpath, 'w') as f:
        f.write(fulltext)


def _make_session_diagnostic_plots(model, df_model):
    """  Display and write to file several diagnostic plots, to:
         * decide which obs, comps, images might need removal by editing session.ini, and
         * possibly adjust regression parameters, also by editing session.ini. """
    context, defaults_dict, session_dict, log_file = _session_setup('(plots)')
    this_directory, mp_string, an_string, filter_string = context

    # Delete any previous plot files from current directory:
    session_plot_filenames = [f for f in os.listdir('.')
                       if f.startswith(SESSION_PLOT_FILE_PREFIX) and f.endswith('.png')]
    for f in session_plot_filenames:
        os.remove(f)

    # Wrangle needed data into convenient forms:
    df_plot_comp_obs = pd.merge(left=df_model.loc[df_model['UseInModel'], :].copy(),
                                right=model.mm_fit.df_observations,
                                how='left', left_index=True, right_index=True, sort=False)
    df_plot_comp_obs = pd.merge(left=df_plot_comp_obs,
                                right=model.df_used_comps_obs['InstMag_with_offsets'],
                                how='left', left_index=True, right_index=True, sort=False)
    df_image_effect = model.mm_fit.df_random_effects
    df_image_effect.rename(columns={"GroupName": "FITSfile", "Group": "ImageEffect"}, inplace=True)
    sigma = model.mm_fit.sigma
    # (Skip value of transform for now.)
    comp_ids = df_plot_comp_obs['CompID'].drop_duplicates()
    n_comps = len(comp_ids)
    jd_floor = floor(min(df_model['JD_mid']))
    xlabel_jd = 'JD(mid)-' + str(jd_floor)

    # ################ SESSION FIGURE 1: Q-Q plot of mean comp effects (1 pt per comp star used in model).
    window_title = 'Q-Q Plot (by comp):  MP ' + mp_string + '   AN ' + an_string
    page_title = 'MP ' + mp_string + '   AN ' + an_string + '   ::   Q-Q plot by comp (mean residual)'
    plot_annotation = str(n_comps) + ' comps used in model.' + '\n(tags: comp star ID)'
    df_y = df_plot_comp_obs.loc[:, ['CompID', 'Residual']].groupby(['CompID']).mean()
    df_y = df_y.sort_values(by='Residual')
    y_data = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y.index.values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
                          SESSION_PLOT_FILE_PREFIX + '1_QQ_comps.png')

    # ################ SESSION FIGURE 2: Q-Q plot of comp residuals (one point per comp obs).
    window_title = 'Q-Q Plot (by comp observation):  MP ' + mp_string + '   AN ' + an_string
    page_title = 'MP ' + mp_string + '   AN ' + an_string + '   ::   Q-Q plot by comp observation'
    plot_annotation = str(len(df_plot_comp_obs)) + ' observations of ' +\
                      str(n_comps) + ' comps used in model.' + \
                      '\n (tags: observation ID)'
    df_y = df_plot_comp_obs.loc[:, ['ObsID', 'Residual']]
    df_y = df_y.sort_values(by='Residual')
    y_data = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y['ObsID'].values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
                          SESSION_PLOT_FILE_PREFIX + '2_QQ_obs.png')

    # ################ SESSION FIGURE 3: Catalog and Time plots:
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(11, 8.5))  # (width, height) in "inches", was 15,9
    fig.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
    fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
    fig.suptitle('MP ' + mp_string + '   AN ' + an_string + '     ::     catalog and time plots',
                 color='darkblue', fontsize=20)
    fig.canvas.set_window_title('Catalog and Time Plots: ' + 'MP ' + mp_string + '   AN ' + an_string)
    subplot_text = 'rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
    fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center', fontsize=12, color='dimgray')

    # Catalog mag uncertainty plot (comps only, one point per comp, x=cat r mag, y=cat r uncertainty):
    ax = axes[0, 0]
    make_9_subplot(ax, 'Catalog Mag Uncertainty (dr)', 'Catalog Mag (r)', 'mMag', '', False,
                   x_data=df_plot_comp_obs['r'], y_data=df_plot_comp_obs['dr'])

    # Catalog color plot (comps only, one point per comp, x=cat r mag, y=cat color (r-i)):
    ax = axes[0, 1]
    make_9_subplot(ax, 'Catalog Color Index', 'Catalog Mag (r)', 'CI Mag', '', zero_line=False,
                   x_data=df_plot_comp_obs['r'], y_data=(df_plot_comp_obs['r'] - df_plot_comp_obs['i']))
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=len(model.df_mp_mags) * [session_dict['mp ri color']],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)  # add MP points.

    # Inst Mag plot (comps only, one point per obs, x=cat r mag, y=InstMagSigma):
    ax = axes[0, 2]
    make_9_subplot(ax, 'Instrument Magnitude Uncertainty', 'Catalog Mag (r)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['r'], y_data=df_plot_comp_obs['InstMagSigma'])
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=model.df_mp_mags['InstMagSigma'],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)  # add MP points.

    # Cirrus plot (comps only, one point per image, x=JD_fract, y=Image Effect):
    ax = axes[1, 0]
    df_this_plot = pd.merge(df_image_effect, df_plot_comp_obs.loc[:, ['FITSfile', 'JD_fract']],
                            how='left', on='FITSfile', sort=False).drop_duplicates()
    make_9_subplot(ax, 'Image effect (cirrus plot)', xlabel_jd, 'mMag', '', False,
                   x_data=df_this_plot['JD_fract'], y_data=1000.0 * df_this_plot['ImageEffect'],
                   alpha=1.0, jd_locators=True)
    ax.invert_yaxis()  # per custom of plotting magnitudes brighter=upward

    # SkyADU plot (comps only, one point per obs: x=JD_fract, y=SkyADU):
    ax = axes[1, 1]
    make_9_subplot(ax, 'SkyADU vs time', xlabel_jd, 'ADU', '', False,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=df_plot_comp_obs['SkyADU'],
                   jd_locators=True)

    # FWHM plot (comps only, one point per obs: x=JD_fract, y=FWHM):
    ax = axes[1, 2]
    make_9_subplot(ax, 'FWHM vs time', xlabel_jd, 'FWHM (pixels)', '', False,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=df_plot_comp_obs['FWHM'],
                   jd_locators=True)

    # InstMagSigma plot (comps only, one point per obs; x=JD_fract, y=InstMagSigma):
    ax = axes[2, 0]
    make_9_subplot(ax, 'Inst Mag Sigma vs time', xlabel_jd, 'mMag', '', False,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=1000.0 * df_plot_comp_obs['InstMagSigma'],
                   jd_locators=True)

    # Obs.Airmass plot (comps only, one point per obs; x=JD_fract, y=ObsAirmass):
    ax = axes[2, 1]
    make_9_subplot(ax, 'Obs.Airmass vs time', xlabel_jd, 'Obs.Airmass', '', False,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=df_plot_comp_obs['ObsAirmass'],
                   jd_locators=True)

    # Session Lightcurve plot (comps only, one point per obs; x=JD_fract, y=MP best magnitude):
    ax = axes[2, 2]
    make_9_subplot(ax, 'MP Lightcurve for this session', xlabel_jd, 'Mag (r)', '', False,
                   x_data=model.df_mp_mags['JD_mid'] - jd_floor, y_data=model.df_mp_mags['MP_Mags'],
                   alpha=1.0, jd_locators=True)
    ax.errorbar(x=model.df_mp_mags['JD_mid'] - jd_floor, y=model.df_mp_mags['MP_Mags'],
                yerr=model.df_mp_mags['InstMagSigma'], fmt='none', color='black',
                linewidth=0.5, capsize=3, capthick=0.5, zorder=-100)
    ax.invert_yaxis()  # per custom of plotting magnitudes brighter=upward

    plt.show()
    fig.savefig(SESSION_PLOT_FILE_PREFIX + '3_Catalog_and_Time.png')

    # ################ SESSION FIGURE 4: Residual plots:
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(11, 8.5))  # (width, height) in "inches", was 15, 9
    fig.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
    fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
    fig.suptitle('MP ' + mp_string + '   AN ' + an_string + '     ::     residual plots',
                 color='darkblue', fontsize=20)
    fig.canvas.set_window_title('Residual Plots: ' + 'MP ' + mp_string + '   AN ' + an_string)
    subplot_text = str(len(df_plot_comp_obs)) + ' obs   ' + \
                   str(n_comps) + ' comps    ' + \
                   'sigma=' + '{0:.0f}'.format(1000.0 * sigma) + ' mMag' + \
                   (12 * ' ') + ' rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
    fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center', fontsize=12, color='dimgray')

    # Comp residual plot (comps only, one point per obs: x=catalog r mag, y=model residual):
    ax = axes[0, 0]
    make_9_subplot(ax, 'Model residual vs r (catalog)', 'Catalog Mag (r)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['r'], y_data=1000.0 * df_plot_comp_obs['Residual'])
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=len(model.df_mp_mags) * [0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)  # add MP points.
    draw_x_line(ax, session_dict['min catalog r mag'])
    draw_x_line(ax, session_dict['max catalog r mag'])

    # Comp residual plot (comps only, one point per obs: x=raw Instrument Mag, y=model residual):
    ax = axes[0, 1]
    make_9_subplot(ax, 'Model residual vs raw Instrument Mag', 'Raw instrument mag', 'mMag', '', True,
                   x_data=df_plot_comp_obs['InstMag'], y_data=1000.0 * df_plot_comp_obs['Residual'])
    ax.scatter(x=model.df_mp_mags['InstMag'], y=len(model.df_mp_mags) * [0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)  # add MP points.

    # Comp residual plot (comps only, one point per obs: x=catalog r-i color, y=model residual):
    ax = axes[0, 2]
    make_9_subplot(ax, 'Model residual vs Color Index (cat)', 'Catalog Color (r-i)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['r'] - df_plot_comp_obs['i'],
                   y_data=1000.0 * df_plot_comp_obs['Residual'])
    ax.scatter(x=[session_dict['mp ri color']], y=[0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)  # add MP points.

    # Comp residual plot (comps only, one point per obs: x=Julian Date fraction, y=model residual):
    ax = axes[1, 0]
    make_9_subplot(ax, 'Model residual vs JD', xlabel_jd, 'mMag', '', True,
                   x_data=df_plot_comp_obs['JD_fract'], y_data=1000.0 * df_plot_comp_obs['Residual'],
                   jd_locators=True)

    # Comp residual plot (comps only, one point per obs: x=ObsAirmass, y=model residual):
    ax = axes[1, 1]
    make_9_subplot(ax, 'Model residual vs Obs.Airmass', 'ObsAirmass', 'mMag', '', True,
                   x_data=df_plot_comp_obs['ObsAirmass'], y_data=1000.0 * df_plot_comp_obs['Residual'])

    # Comp residual plot (comps only, one point per obs: x=Sky Flux (ADUs), y=model residual):
    ax = axes[1, 2]
    make_9_subplot(ax, 'Model residual vs Sky Flux', 'Sky Flux (ADU)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['SkyADU'], y_data=1000.0 * df_plot_comp_obs['Residual'])

    # Comp residual plot (comps only, one point per obs: x=X in images, y=model residual):
    ax = axes[2, 0]
    make_9_subplot(ax, 'Model residual vs X in image', 'X from center (pixels)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['X1024'] * 1024.0, y_data=1000.0 * df_plot_comp_obs['Residual'])
    draw_x_line(ax, 0.0)

    # Comp residual plot (comps only, one point per obs: x=Y in images, y=model residual):
    ax = axes[2, 1]
    make_9_subplot(ax, 'Model residual vs Y in image', 'Y from center (pixels)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['Y1024'] * 1024.0, y_data=1000.0 * df_plot_comp_obs['Residual'])
    draw_x_line(ax, 0.0)

    # Comp residual plot (comps only, one point per obs: x=vignette (dist from center), y=model residual):
    ax = axes[2, 2]
    make_9_subplot(ax, 'Model residual vs distance from center', 'dist from center (pixels)', 'mMag',
                   '', True,
                   x_data=1024*np.sqrt(df_plot_comp_obs['Vignette']),
                   y_data=1000.0 * df_plot_comp_obs['Residual'])

    plt.show()
    fig.savefig(SESSION_PLOT_FILE_PREFIX + '4_Residuals.png')

    # ################ FIGURE(S) 5: Variability plots:
    # Several comps on a subplot, vs JD, normalized by (minus) the mean of all other comps' responses.
    # Make df_offsets (one row per obs, at first with only raw offsets):
    make_comp_variability_plots(df_plot_comp_obs, mp_string, an_string, xlabel_jd, sigma,
                                SESSION_PLOT_FILE_PREFIX)


_____SUPPORT_for_PLOTTING___________________________________________________ = 0


def make_qq_plot_fullpage(window_title, page_title, plot_annotation,
                          y_data, y_labels, filename, figsize=(11, 8.5)):
    """ Make full-page QQ plot from data passed in. Plot on console, and save plot file. """
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=figsize)  # (width, height) in "inches"
    ax = axes  # not subscripted if just one subplot in Figure
    ax.set_title(page_title, color='darkblue', fontsize=20, pad=30)
    ax.set_xlabel('t (sigma.residuals = ' + str(round(pd.Series(y_data).std(), 1)) + ' mMag)')
    ax.set_ylabel('Residual (mMag)')
    ax.grid(True, color='lightgray', zorder=-1000)
    df_y = pd.DataFrame({'Y': y_data, 'Label': y_labels}).sort_values(by='Y')
    n = len(df_y)
    t_values = [norm.ppf((k - 0.5) / n) for k in range(1, n + 1)]
    ax.scatter(x=t_values, y=df_y['Y'], alpha=0.6, color='dimgray', zorder=+1000)
    # Label potential outliers:
    z_score_y = (df_y['Y'] - df_y['Y'].mean()) / df_y['Y'].std()
    is_outlier = (abs(z_score_y) >= 2.0)
    for x, y, label, add_label in zip(t_values, df_y['Y'], df_y['Label'], is_outlier):
        if add_label:
            ax.annotate(label, xy=(x, y), xytext=(4, -4),
                        textcoords='offset points', ha='left', va='top', rotation=-40)
    # Add reference line:
    x_low = 1.10 * min(t_values)
    x_high = 1.10 * max(t_values)
    y_low = x_low * df_y['Y'].std()
    y_high = x_high * df_y['Y'].std()
    ax.plot([x_low, x_high], [y_low, y_high], color='gray', zorder=-100, linewidth=1)
    # Finish FIGURE 1:
    fig.text(x=0.5, y=0.87, s=plot_annotation,
             verticalalignment='top', horizontalalignment='center', fontsize=12)
    fig.canvas.set_window_title(window_title)
    plt.show()
    fig.savefig(filename)


def make_9_subplot(ax, title, x_label, y_label, text, zero_line, x_data, y_data,
                   size=14, alpha=0.3, color='black', jd_locators=False):
    """ Make a subplot sized to 3x3 subplots/page. Frame w/labels only if x_data is None or y_data is None.
    :param ax: axis location of this subplot. [matplotlib Axes object]
    :param title: text atop the plot. [string]
    :param x_label: x-axis label. [string]
    :param y_label: y_axis label. [string]
    :param text: text inside top border (rarely used). [string]
    :param zero_line: iff True, plot a light line along Y=0. [boolean]
    :param x_data: vector of x-values to plot. [iterable of floats]
    :param y_data: vector of y-values to plot, must equal x_data in length. [iterable of floats]
    :param size: size of points to plot each x,y. [float, weird matplotlib scale]
    :param alpha: opacity of each point. [float, 0 to 1]
    :param color: name of color to plot each point. [string, matplotlib color]
    :param jd_locators: iff True, make x-axis ticks convenient to plotting JDs of one night. [boolean]
    :return: [no return value]
    """
    ax.set_title(title, loc='center', pad=-3)  # pad in points
    ax.set_xlabel(x_label, labelpad=-29)  # labelpad in points
    ax.set_ylabel(y_label, labelpad=-5)  # "
    ax.text(x=0.5, y=0.95, s=text,
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    if zero_line is True:
        ax.axhline(y=0, color='lightgray', linewidth=1, zorder=-100)
    if x_data is not None and y_data is not None:
        ax.scatter(x=x_data, y=y_data, s=size, alpha=alpha, color=color)
    if jd_locators:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))


def make_comp_variability_plots(df_plot_comp_obs, mp_string, an_string, xlabel_jd, sigma, image_prefix):
    """ Make pages of 9 subplots, 4 comps per subplot, showing jackknife comp residuals.
    :param df_plot_comp_obs: table of data from which to generate all subplots. [pandas Dataframe]
    :param mp_string:
    :param an_string:
    :param xlabel_jd: label to go under x-axis (JD axis) for each subplot. [string]
    :param sigma: std deviation of comp responses from fit. [float]
    :param image_prefix: prefix for output images' filenames. [string]
    :return: [None]
    """
    comp_ids = df_plot_comp_obs['CompID'].drop_duplicates()
    n_comps = len(comp_ids)
    dict_list = []
    for comp_id in comp_ids:
        is_comp_id = df_plot_comp_obs['CompID'] == comp_id
        obs_ids = df_plot_comp_obs.loc[is_comp_id, 'ObsID']
        jd_fracts = df_plot_comp_obs.loc[is_comp_id, 'JD_fract']
        inst_mags = df_plot_comp_obs.loc[is_comp_id, 'InstMag']
        r_catmag = df_plot_comp_obs.loc[is_comp_id, 'r']
        # color_ri = r_catmag - df_plot_comp_obs.loc[is_comp_id, 'i']
        # transform_offset = 0.4 * color_ri - 0.6 * color_ri ** 2
        # raw_offsets_per_mpc = inst_mags - r_catmag - transform_offset
        raw_offsets = df_plot_comp_obs.loc[is_comp_id, 'InstMag_with_offsets']
        # raw_offsets = inst_mags - r_catmag  # - transform_1 * color_ri  # pd Series
        for i in range(len(obs_ids)):
            comp_dict = dict()
            comp_dict['CompID'] = comp_id
            comp_dict['ObsID'] = obs_ids.iloc[i]
            comp_dict['JD_fract'] = jd_fracts.iloc[i]
            comp_dict['RawOffset'] = raw_offsets.iloc[i]
            dict_list.append(comp_dict)
    df_comp_offsets = pd.DataFrame(data=dict_list)
    df_comp_offsets.index = df_comp_offsets['ObsID']

    # Normalize offsets by subtracting mean of *other* comps' offsets at each JD_fract:
    all_jd_fracts = df_comp_offsets['JD_fract'].copy().drop_duplicates().sort_values()
    df_comp_offsets['NormalizedOffset'] = None
    df_comp_offsets['LatestNormalizedOffset'] = None
    for comp_id in comp_ids:
        is_comp_id = (df_comp_offsets['CompID'] == comp_id)
        valid_jd_fracts = []
        for jd_fract in all_jd_fracts:
            is_this_jd_fract = (df_comp_offsets['JD_fract'] == jd_fract)
            mean_other_offsets = df_comp_offsets.loc[(~is_comp_id) & is_this_jd_fract, 'RawOffset'].mean()
            is_this_obs = is_comp_id & is_this_jd_fract
            this_raw_offset = df_comp_offsets.loc[is_this_obs, 'RawOffset'].mean()
            normalized_offset = this_raw_offset - mean_other_offsets
            df_comp_offsets.loc[is_this_obs, 'NormalizedOffset'] = normalized_offset
            if not np.isnan(normalized_offset):
                valid_jd_fracts.append(jd_fract)
        is_latest_jd_fract = df_comp_offsets['JD_fract'] == valid_jd_fracts[-1]
        latest_normalized_offset = df_comp_offsets.loc[is_comp_id & is_latest_jd_fract, 'NormalizedOffset']
        df_comp_offsets.loc[is_comp_id, 'LatestNormalizedOffset'] = latest_normalized_offset.iloc[0]

    df_comp_offsets = df_comp_offsets.sort_values(by=['LatestNormalizedOffset', 'CompID', 'JD_fract'],
                                                  ascending=[False, True, True])
    df_plot_index = df_comp_offsets[['CompID']].drop_duplicates()
    df_plot_index['PlotIndex'] = range(len(df_plot_index))
    df_comp_offsets = pd.merge(left=df_comp_offsets, right=df_plot_index,
                               how='left', on='CompID', sort=False)

    # Plot the normalized offsets vs JD for each comp_id, 4 comp_ids to a subplot:
    n_cols, n_rows = 3, 3
    n_plots_per_figure = n_cols * n_rows
    n_comps_per_plot = 4
    n_plots = ceil(n_comps / n_comps_per_plot)
    plot_colors = ['r', 'g', 'm', 'b']
    n_figures = ceil(n_plots / n_plots_per_figure)
    jd_range = max(all_jd_fracts) - min(all_jd_fracts)
    jd_low_limit = min(all_jd_fracts) - 0.05 * jd_range
    jd_high_limit = max(all_jd_fracts) + 0.40 * jd_range
    normalized_offset_mmag = 1000.0 * df_comp_offsets['NormalizedOffset']
    offset_range = max(normalized_offset_mmag) - min(normalized_offset_mmag)
    offset_low_limit = min(normalized_offset_mmag) - 0.05 * offset_range
    offset_high_limit = max(normalized_offset_mmag) + 0.05 * offset_range
    plotted_comp_ids = []
    for i_figure in range(n_figures):
        n_plots_remaining = n_plots - (i_figure * n_plots_per_figure)
        n_plots_this_figure = min(n_plots_remaining, n_plots_per_figure)
        if n_plots_this_figure >= 1:
            # Start new Figure:
            fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(11, 8.5))  # was 15, 9
            fig.tight_layout(rect=(0, 0, 1, 0.925))  # rect=(left, bottom, right, top) for entire fig
            fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85, wspace=0.25, hspace=0.325)
            fig.suptitle('MP ' + mp_string + '   AN ' + an_string + '     ::     Comp Variability Page ' +
                         str(i_figure + 1) + ' of ' + str(n_figures),
                         color='darkblue', fontsize=20)
            fig.canvas.set_window_title(
                'Comp Variability Plots: ' + 'MP ' + mp_string + '   AN ' + an_string)
            subplot_text = str(n_comps) + ' comps    ' + \
                           'sigma=' + '{0:.0f}'.format(1000.0 * sigma) + ' mMag' + (12 * ' ') + \
                           ' rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
            fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center', fontsize=12,
                     color='dimgray')
            for i_plot in range(n_plots_this_figure):
                first_index_in_plot = i_figure * n_plots_per_figure + i_plot
                i_col = i_plot % n_cols
                i_row = int(floor(i_plot / n_cols))
                ax = axes[i_row, i_col]
                # Make a frame and axes only (x and y vectors are None):
                make_9_subplot(ax, 'Comp Variability plot', xlabel_jd, 'mMag', '', True, None, None)
                # Make a scatter plot for each chosen comp:
                scatterplots = []
                legend_labels = []
                for i_plot_comp in range(n_comps_per_plot):
                    i_plot_index = first_index_in_plot + i_plot_comp * n_plots
                    if i_plot_index <= n_comps - 1:
                        is_this_plot = (df_comp_offsets['PlotIndex'] == i_plot_index)
                        x = df_comp_offsets.loc[is_this_plot, 'JD_fract']
                        y = 1000.0 * df_comp_offsets.loc[is_this_plot, 'NormalizedOffset']
                        ax.plot(x, y, linewidth=2, alpha=0.8, color=plot_colors[i_plot_comp])
                        sc = ax.scatter(x=x, y=y, s=24, alpha=0.8, color=plot_colors[i_plot_comp])
                        scatterplots.append(sc)
                        this_comp_id = (df_comp_offsets.loc[is_this_plot, 'CompID']).iloc[0]
                        # print('i_figure ' + str(i_figure) +\
                        #       '  i_plot ' + str(i_plot) +\
                        #       '  i_plot_comp ' + str(i_plot_comp) +\
                        #       '  i_plot_index ' + str(i_plot_index) +\
                        #       '  this_comp_id ' + this_comp_id)  # (debug)
                        legend_labels.append(this_comp_id)
                        plotted_comp_ids.append(this_comp_id)
                ax.set_xlim(jd_low_limit, jd_high_limit)
                ax.set_ylim(offset_low_limit, offset_high_limit)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
                ax.xaxis.set_minor_locator(ticker.MaxNLocator(20))
                ax.legend(scatterplots, legend_labels, loc='upper right')
            # Remove any empty subplots from this (last) Figure:
            for i_plot in range(n_plots_this_figure, n_plots_per_figure):
                i_col = i_plot % n_cols
                i_row = int(floor(i_plot / n_cols))
                ax = axes[i_row, i_col]
                ax.remove()
        plt.show()
        plt.savefig(image_prefix + '5_Comp Variability_' + '{:02d}'.format(i_figure + 1) + '.png')

    # Verify that all comps were plotted exactly once (debug):
    all_comps_plotted_once = (sorted(comp_ids) == sorted(plotted_comp_ids))
    if not all_comps_plotted_once:
        print('comp ids plotted more than once',
              [item for item, count in Counter(plotted_comp_ids).items() if count > 1])


def draw_x_line(ax, x_value, color='lightgray'):
    """ Draw vertical line, typically to mark MP transit time. """
    ax.axvline(x=x_value, color=color, linewidth=1, zorder=-100)



_____SUPPORT_FUNCTIONS_and_CLASSES________________________________________ = 0


def make_fits_objects(this_directory, fits_filenames):
    """ Assemble valid, time-sorted FITS objects into (1) a list and (2) an OrderedDict by filename:
    :param this_directory:
    :param fits_filenames:
    :return:
    """
    all_fits_objects = [FITS(this_directory, '', fn) for fn in fits_filenames]
    invalid_fits_filenames = [fo.filename for fo in all_fits_objects if not fo.is_valid]
    if invalid_fits_filenames:
        for fn in invalid_fits_filenames:
            print(' >>>>> WARNING: Invalid FITS filename ' + fn +
                  ' is skipped. User should explicitly exclude it (e.g., move to Exclude subdir.')
    fits_objects = [fo for fo in all_fits_objects if fo.is_valid]
    fits_objects = sorted(fits_objects, key=lambda fo: fo.utc_mid)
    fits_object_dict = OrderedDict((fo.filename, fo) for fo in fits_objects)
    return fits_objects, fits_object_dict


def get_refcat2_comp_stars(fits_objects):
    aggr_ra_deg_min, aggr_ra_deg_max, aggr_dec_deg_min, aggr_dec_deg_max = \
        aggregate_bounding_ra_dec(fits_objects, extension_percent=3)
    refcat2 = Refcat2(ra_deg_range=(aggr_ra_deg_min, aggr_ra_deg_max),
                      dec_deg_range=(aggr_dec_deg_min, aggr_dec_deg_max))
    utc_mids = [fo.utc_mid for fo in fits_objects]
    utc_mid_session = min(utc_mids) + (max(utc_mids) - min(utc_mids)) / 2
    refcat2.update_epoch(utc_mid_session)
    return refcat2


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
    mp_location_lines = [
        '[MP Location]',
        '# Exactly 2 MP XY, one per line (typically earliest and latest FITS):',
        'MP XY = ' + filename_earliest + ' 000.0  000.0',
        '        ' + filename_latest + ' 000.0  000.0',
        '']
    # bulldozer_lines = [
    #     '[Bulldozer]',
    #     '# At least 3 ref star XY, one per line, all from one FITS only if at all possible:',
    #     'Ref Star XY = ' + filename_earliest + ' 000.0  000.0',
    #     '              ' + filename_earliest + ' 000.0  000.0',
    #     '              ' + filename_earliest + ' 000.0  000.0',
    #     '']
    selection_criteria_lines = [
        '[Selection Criteria]',
        'Omit Comps =',
        'Omit Obs =',
        '# One image only per line, with or without .fts:',
        'Omit Images =',
        'Min Catalog r mag = 12',
        'Max Catalog r mag = 16',
        'Max Catalog dr mmag = 16',
        'Max Catalog di mmag = 16',
        'Min Catalog ri color = 0.10',
        'Max Catalog ri color = 0.34',
        '']
    regression_lines = [
        '[Regression]',
        'MP ri color = +0.220',
        'MP ri color origin = Default MP color',
        '# Fit Transform, one of: Fit=1, Fit=2, Use [val1], Use [val1] [val2]:',
        'Fit Transform = Use +0.4 -0.6',
        '# Fit Extinction, one of: Yes, Use [val]:',
        'Fit Extinction = Use +0.16',
        'Fit Vignette = Yes',
        'Fit XY = No',
        'Fit JD = No']
    raw_lines = header_lines + ini_lines + mp_location_lines + selection_criteria_lines + regression_lines
    ready_lines = [line + '\n' for line in raw_lines]
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(ready_lines)
    # print('New ' + session_ini_filename + ' file written.\n')


def _session_setup(calling_function_name='[FUNCTION NAME NOT GIVEN]'):
    """ Typically called at the top of lightcurve workflow functions, to collect commonly required data.
    :return: tuple of data elements: context [tuple], defaults_dict [py dict], log_file [file object].
    """
    context = _get_session_context()
    if context is None:
        return
    this_directory, mp_string, an_string, filter_string = context
    defaults_dict = ini.make_defaults_dict()
    session_dict = ini.make_session_dict(defaults_dict, this_directory)
    log_filename = defaults_dict['session log filename']
    log_file = open(log_filename, mode='a')  # set up append to log file.
    log_file.write('\n===== ' + calling_function_name + '()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    return context, defaults_dict, session_dict, log_file


def initial_screen_comps(refcat2):
    """ Screen ATLAS refcat2 stars for quality before aperture photometry.
        This is a wide selection, to be tightened during regression (do_phot()).
        Screens are performed IN-PLACE on refcat2 catalog object. Returns text lines documenting progress.
    :param refcat2: ATLAS refcat2 catalog from astropak.catalog.py. [Refcat2 object]
    :return info: text documenting actions taken. [list of strings]
    """
    info = []
    info.append('Begin with ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_min_r_mag(INITIAL_MIN_R_MAG)
    info.append('Refcat2: min(r) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_max_r_mag(INITIAL_MAX_R_MAG)
    info.append('Refcat2: max(r) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_max_r_uncert(INITIAL_MAX_DR_MMAG)
    info.append('Refcat2: max(dr) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_max_i_uncert(INITIAL_MAX_DI_MMAG)
    info.append('Refcat2: max(di) screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.select_sloan_ri_color(INITIAL_MIN_RI_COLOR, INITIAL_MAX_RI_COLOR)
    info.append('Refcat2: Sloan ri color screened to ' + str(len(refcat2.df_selected)) + ' stars.')
    refcat2.remove_overlapping()
    info.append('Refcat2: overlaps removed to ' + str(len(refcat2.df_selected)) + ' stars.')
    return info


def saturation_sat_at_xy1024(x1024, y1024, vignette_at_1024, adu_saturation):
    """ Return estimated saturation ADU limit from aperture's distances from image center."""
    dist2 = x1024 ** 2 + y1024 ** 2
    fraction_decreased = vignette_at_1024 * (dist2 / 1024) ** 2
    return adu_saturation * (1.0 - fraction_decreased)


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


def _write_df_images_csv(df_images, this_directory, defaults_dict, log_file):
    filename_df_images = defaults_dict['df_images filename']
    fullpath_df_images = os.path.join(this_directory, filename_df_images)
    df_images.to_csv(fullpath_df_images, sep=';', quotechar='"',
                     quoting=2, index=True)  # quoting=2 means quotes around non-numerics.
    print('images written to', fullpath_df_images)
    log_file.write(filename_df_images + ' written: ' + str(len(df_images)) + ' images.\n')


def _write_df_comps_csv(df_comps, this_directory, defaults_dict, log_file):
    filename_df_comps = defaults_dict['df_comps filename']
    fullpath_df_comps = os.path.join(this_directory, filename_df_comps)
    df_comps.to_csv(fullpath_df_comps, sep=';', quotechar='"',
                    quoting=2, index=True)  # quoting=2 means quotes around non-numerics.
    print('comps written to', fullpath_df_comps)
    log_file.write(filename_df_comps + ' written: ' + str(len(df_comps)) + ' comp stars.\n')


def _write_df_comp_obs_csv(df_comp_obs, this_directory, defaults_dict, log_file):
    filename_df_comp_obs = defaults_dict['df_comp_obs filename']
    fullpath_df_comp_obs = os.path.join(this_directory, filename_df_comp_obs)
    df_comp_obs.to_csv(fullpath_df_comp_obs, sep=';', quotechar='"',
                       quoting=2, index=True)  # quoting=2 means quotes around non-numerics.
    print('comp star observations written to', fullpath_df_comp_obs)
    log_file.write(filename_df_comp_obs + ' written: '
                   + str(len(df_comp_obs)) + ' comp star observations.\n')


def _write_df_mp_obs_csv(df_mp_obs, this_directory, defaults_dict, log_file):
    filename_df_mp_obs = defaults_dict['df_mp_obs filename']
    fullpath_df_mp_obs = os.path.join(this_directory, filename_df_mp_obs)
    df_mp_obs.to_csv(fullpath_df_mp_obs, sep=';', quotechar='"',
                     quoting=2, index=True)  # quoting=2 means quotes around non-numerics.
    print('MP observations written to', fullpath_df_mp_obs)
    log_file.write(filename_df_mp_obs + ' written: ' + str(len(df_mp_obs)) + ' MP observations.\n')


def write_text_file(this_directory, filename, lines):
    fullpath = os.path.join(this_directory, filename)
    with open(fullpath, 'w') as f:
        f.write(lines)
