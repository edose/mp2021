__author__ = "Eric Dose, Albuquerque"

from astroplan import Observer
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from astropak.reference import DEGREES_PER_RADIAN

""" This module: 
      
"""

# Python core:
import os
from collections import Counter, OrderedDict
from math import floor, log10, log, sin

# External packages:
import numpy as np
import pandas as pd
from astropy.io import fits as apyfits

# Author's packages:
from astropak.catalogs import Refcat2
from astropak.image import PointSourceAp, MovingSourceAp, FITS, aggregate_bounding_ra_dec
from astropak.util import jd_from_datetime_utc, RaDec
from mp2021 import util as util, ini as ini

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')

FOCAL_LENGTH_MAX_PCT_DEVIATION = 1.0

# The following apply only to initial screening of comps from catalog to dataframe(s):
INITIAL_MIN_R_MAG = 10
INITIAL_MAX_R_MAG = 16
INITIAL_MAX_DR_MMAG = 20
INITIAL_MAX_DI_MMAG = 20
INITIAL_MIN_RI_COLOR = 0.00
INITIAL_MAX_RI_COLOR = 0.44


_____SUPPORT_for_assess___________________________________________________ = 0


def do_fits_assessments(defaults_dict, this_directory):
    return_dict = {
        'file not read': [],  # list of filenames
        'filter not read': [],  # "
        'file count by filter': [],  # list of tuples (filter, file count)
        'warning count': 0,  # total count of all warnings.
        'not platesolved': [],  # list of filenames
        'not calibrated': [],  # "
        'unusual fwhm': [],  # list of tuples (filename, fwhm)
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
        df.loc[filename, 'JD_mid'] = jd_mid  # needed only to write control.ini stub (1st & last FITS).
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
    return df, return_dict


_____SUPPORT_for_make_dfs_____________________________________________ = 0


def validate_mp_xy(fits_filenames, control_dict):
    """ Quick check for frequently made errors in MP XY entries in control dict.
    :param fits_filenames:
    :param control_dict:
    :return: 2-tuple, (mp_xy_files_found, mp_xy_values_ok). [2-tuple of booleans, both True if OK]
    """
    mp_location_filenames = [mp_xy_entry[0] for mp_xy_entry in control_dict['mp xy']]
    mp_xy_files_found = all([fn in fits_filenames for fn in mp_location_filenames])
    mp_location_xy = [(item[1], item[2]) for item in control_dict['mp xy']][:2]
    flatten_mp_location_xy = [y for x in mp_location_xy for y in x]
    mp_xy_values_ok = all([x > 0.0 for x in flatten_mp_location_xy])
    return mp_xy_files_found, mp_xy_values_ok


def make_df_images(fits_objects):
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


def make_df_comps(refcat2):
    df_comps = refcat2.selected_columns(['RA_deg', 'Dec_deg', 'RP1', 'R1', 'R10',
                                         'g', 'dg', 'r', 'dr', 'i', 'di', 'z', 'dz',
                                         'BminusV', 'APASS_R', 'T_eff', 'CatalogID'])
    comp_ids = [str(i + 1) for i in range(len(df_comps))]
    df_comps.index = comp_ids
    df_comps.insert(0, 'CompID', comp_ids)
    return df_comps


def make_comp_apertures(fits_objects, df_comps, disc_radius, gap, background_width):
    comp_radec_dict = {comp_id: RaDec(df_comps.loc[comp_id, 'RA_deg'], df_comps.loc[comp_id, 'Dec_deg'])
                       for comp_id in df_comps.index}
    comp_apertures_dict = OrderedDict()
    obs_id = 1
    for fo in fits_objects:
        print('\n' + fo.filename + ': ', end='', flush=True)
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
                    # if obs_id % 1000 == 0:
                    #     print('{:6d} raw apertures done'.format(obs_id), flush=True)
        comp_apertures_dict[fo.filename] = ap_list
    print()
    return comp_apertures_dict


def make_df_comp_obs(comp_apertures, df_comps, instrument, df_images):
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


def make_mp_apertures(fits_object_dict, mp_string, control_dict,
                      disc_radius, gap, background_width, log_file, starting_obs_id,
                      print_ap_details=False):
    """ Make mp_apertures, one row per MP (thus one row per FITS file).
    :param fits_object_dict:
    :param mp_string:
    :param control_dict:
    :param disc_radius:
    :param gap:
    :param background_width:
    :param log_file:
    :param starting_obs_id: [int]
    :param print_ap_details: True iff user wants aperture details for each MP. [boolean]
    :return:
    """
    utc0, ra0, dec0, ra_per_second, dec_per_second = \
        calc_mp_motion(control_dict, fits_object_dict, log_file)
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
        if print_ap_details:
            print('\n' + filename, 'raw:', 'start={:.1f} {:.1f}'.format(xy_start[0], xy_start[1]),
                  'end={:.1f} {:.1f}'.format(xy_end[0], xy_end[1]),
                  'flux={:.1f}'.format(raw_ap.net_flux))
            print(filename, 'bkgd level:', str(int(raw_ap.background_level)))
            print(filename, 'pixel counts:', raw_ap.foreground_pixel_count, raw_ap.background_pixel_count)
        if raw_ap.is_valid and raw_ap.all_inside_image:  # severest screen available from PointSourceAp.
            ap = raw_ap.recenter(max_iterations=3)
            if print_ap_details:
                print(filename, 'recentered.')
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
                if print_ap_details:
                    print(filename, 'recentered:',
                          'start={:.1f} {:.1f}'.format(ap.xy_start[0], ap.xy_start[1]),
                          'end={:.1f} {:.1f}'.format(ap.xy_end[0], ap.xy_end[1]),
                          'flux={:.1f}'.format(ap.net_flux))
    return mp_apertures_dict, mp_mid_radec_dict


def make_df_mp_obs(mp_apertures, mp_mid_radec_dict, instrument, df_images):
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
    # ***** We have moved the following test higher in function heirarchy (to avoid passing in exception).
    # flatten_mp_location_xy = [y for x in mp_location_xy for y in x]
    # if any([x <= 0.0 for x in flatten_mp_location_xy]):
    #     raise SessionIniFileError('MP XY invalid -- did you enter values and save session.ini?')
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
    ra_per_second = (mp_ra_deg[1] - ra0) / span_seconds  # deg RA per second
    dec_per_second = (mp_dec_deg[1] - dec0) / span_seconds  # deg Dec per second
    if log_file is not None:
        log_file.write('MP at JD ' + '{0:.5f}'.format(jd_from_datetime_utc(utc0)) + ':  RA,Dec='
                       + '{0:.5f}'.format(ra0) + u'\N{DEGREE SIGN}' + ', '
                       + '{0:.5f}'.format(dec0) + u'\N{DEGREE SIGN}' + ',  d(RA,Dec)/hour='
                       + '{0:.6f}'.format(ra_per_second * 3600.0) + ', '
                       + '{0:.6f}'.format(dec_per_second * 3600.0) + '\n')
    return utc0, ra0, dec0, ra_per_second, dec_per_second


def add_obsairmass_df_comp_obs(df_comp_obs, site_dict, df_comps, df_images):
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


def add_obsairmass_df_mp_obs(df_mp_obs, site_dict, df_images):
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


def add_gr_color_df_comps(df_comps):
    """ Simply add one color column to df_comps. """
    df_comps['gr_color'] = df_comps['g'] - df_comps['r']


def add_ri_color_df_comps(df_comps):
    """ Simply add one color column to df_comps. """
    df_comps['ri_color'] = df_comps['r'] - df_comps['i']


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


def write_df_images_csv(df_images, this_directory, defaults_dict, log_file):
    filename_df_images = defaults_dict['df_images filename']
    fullpath_df_images = os.path.join(this_directory, filename_df_images)
    df_images.to_csv(fullpath_df_images, sep=';', quotechar='"',
                     quoting=2, index=True)  # quoting=2 means quotes around non-numerics.
    print('images written to', fullpath_df_images)
    log_file.write(filename_df_images + ' written: ' + str(len(df_images)) + ' images.\n')


def write_df_comps_csv(df_comps, this_directory, defaults_dict, log_file):
    filename_df_comps = defaults_dict['df_comps filename']
    fullpath_df_comps = os.path.join(this_directory, filename_df_comps)
    df_comps.to_csv(fullpath_df_comps, sep=';', quotechar='"',
                    quoting=2, index=True)  # quoting=2 means quotes around non-numerics.
    print('comps written to', fullpath_df_comps)
    log_file.write(filename_df_comps + ' written: ' + str(len(df_comps)) + ' comp stars.\n')


def write_df_comp_obs_csv(df_comp_obs, this_directory, defaults_dict, log_file):
    filename_df_comp_obs = defaults_dict['df_comp_obs filename']
    fullpath_df_comp_obs = os.path.join(this_directory, filename_df_comp_obs)
    df_comp_obs.to_csv(fullpath_df_comp_obs, sep=';', quotechar='"',
                       quoting=2, index=True)  # quoting=2 means quotes around non-numerics.
    print('comp star observations written to', fullpath_df_comp_obs)
    log_file.write(filename_df_comp_obs + ' written: '
                   + str(len(df_comp_obs)) + ' comp star observations.\n')


def write_df_mp_obs_csv(df_mp_obs, this_directory, defaults_dict, log_file):
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


def read_mp2021_csv(this_directory, filename, dtype_dict=None):
    """ Simple utility to read specified CSV file into a pandas Dataframe. """
    fullpath = os.path.join(this_directory, filename)
    df = pd.read_csv(fullpath, sep=';', index_col=0, header=0, dtype=dtype_dict)
    df.index = df.index.astype(str)
    return df


def make_df_masters(this_directory, defaults_dict, filters_to_include=None, require_mp_obs_each_image=True,
                    data_error_exception_type=ValueError):
    """ Get, screen and merge dataframes df_images_all, df_comps_all, df_comp_obs_all, and df_mp_obs_all
        into two master dataframes dataframe df_comp_master and df_mp_master.
        USAGE (typical):
    :param this_directory:
    :param defaults_dict:
    :param filters_to_include: either one filter name, or a list of filters.
               Only observations in that filter or filters will be retained.
               None includes ALL filters given in input dataframes [None, or string, or list of strings]
    :param require_mp_obs_each_image: True to remove all obs from images without MP observation. [boolean]
    :param data_error_exception_type: Exception type to raise if there is any problem with data read in.
    :return: df_comp_master, df_mp_master, the two master tables of data, one row per retained observation.
                 [2-tuple of pandas DataFrames]
"""
    if isinstance(filters_to_include, str):
        filters_to_include = [filters_to_include]
    # context, defaults_dict, session_dict, log_file = _session_setup('do_session()')
    # this_directory, mp_string, an_string, filter_string = context

    df_images_all = read_mp2021_csv(this_directory, defaults_dict['df_images filename'])
    df_comps_all = read_mp2021_csv(this_directory, defaults_dict['df_comps filename'],
                                   dtype_dict={'CompID': str})
    df_comp_obs_all = read_mp2021_csv(this_directory, defaults_dict['df_comp_obs filename'],
                                      dtype_dict={'CompID': str, 'ObsID': str})
    df_mp_obs_all = read_mp2021_csv(this_directory, defaults_dict['df_mp_obs filename'],
                                    dtype_dict={'ObsID': str})

    # Keep only rows in specified filters:
    image_rows_to_keep = df_images_all['Filter'].isin(filters_to_include)
    df_images = df_images_all.loc[image_rows_to_keep, :]
    filters_string = str(filters_to_include)
    if len(df_images) <= 0:
        raise data_error_exception_type('No images found in specified filter(s): ' + filters_string)
    comp_obs_rows_to_keep = df_comp_obs_all['FITSfile'].isin(df_images['FITSfile'])
    df_comp_obs = df_comp_obs_all.loc[comp_obs_rows_to_keep, :]
    if len(df_comp_obs) <= 0:
        raise data_error_exception_type('No comp obs found in specified filters(s)' + filters_string)
    mp_obs_rows_to_keep = df_mp_obs_all['FITSfile'].isin(df_images['FITSfile'])
    df_mp_obs = df_mp_obs_all.loc[mp_obs_rows_to_keep, :]
    if len(df_mp_obs) <= 0:
        raise data_error_exception_type('No MP obs found in specified filters(s)' + filters_string)

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


def make_df_model_raw(df_comp_master):
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


def mark_user_selections(df_model, session_dict):
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
    df_model = df_model.copy(deep=True)  # to shut up pandas and its fake warnings.
    df_model.loc[:, 'UseInModel'] = ~ obs_to_deselect
    return df_model
