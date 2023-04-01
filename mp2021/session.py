""" This module: Workflow for MP (minor planet) photometry.
    The workflow is applied to a "session" = one MP's images from one imaging night.
    Intended for lightcurves in support of determining MP rotation rates.
"""
__author__ = "Eric Dose, Albuquerque"


# Python core:
import os
from datetime import datetime, timezone
from collections import Counter
from math import floor, ceil
import copy
from typing import Tuple

# External packages:
import numpy as np
import pandas as pd
# import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm


# Author's packages:
import mp2021.util as util
import mp2021.ini as ini
import mp2021.common as common
# from mp2021.common import do_fits_assessments, make_df_images, make_df_comps,
#     make_comp_apertures, make_df_comp_obs, make_mp_apertures,
#     make_fits_objects, get_refcat2_comp_stars,
#     initial_screen_comps, \
#     make_df_mp_obs, write_df_images_csv, write_df_comps_csv, write_df_comp_obs_csv, \
#     write_df_mp_obs_csv, write_text_file, validate_mp_xy,
#     add_obsairmass_df_comp_obs, \
#     add_obsairmass_df_mp_obs, add_ri_color_df_comps, make_df_masters

from astropak.util import datetime_utc_from_jd, ra_as_hours, dec_as_hex
from astropak.stats import MixedModelFit


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname\
    (os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')

MINIMUM_COMP_OBS_COUNT = 5

ALCDEF_BASE_DATA = {'contactname': 'Eric V. Dose',
                    'contactinfo': 'MP@ericdose.com',
                    'observers': 'Dose, E.V.',
                    'filter': 'C',
                    'magband': 'SR'}
SESSION_PLOT_FILE_PREFIX = 'Image_Session_'
PASSBAND_NAMES = {'SG': 'g', 'SR': 'r', 'SI': 'i'}  # local name: catalog name.


class SessionIniFileError(Exception):
    """ Raised on any fatal problem with session ini file."""


class SessionLogFileError(Exception):
    """ Raised on any fatal problem with session log file."""


class SessionDataError(Exception):
    """ Raised on any fatal problem with data,
        esp. with contents of FITS files or missing data."""


class SessionSpecificationError(Exception):
    """ Raised on any fatal problem in specifying the session or
        processing, esp. in _make_df_all(). """


def start(session_top_directory=None, mp_id=None, an_date=None, filter=None):
    # Adapted from mp_phot workflow_session.start().
    """ Launch one session of MP photometry workflow.
        Adapted from package mp_phot, workflow_session.py.start().
        Example usage: session.start('C:/Astro/MP Photometry/', 1111, 20200617, 'Clear')
    :param session_top_directory: path of lowest directory common to
        all MP lightcurve FITS, e.g., 'C:/Astro/MP Photometry'.
        None will use .ini file default (normal case). [string]
    :param mp_id: either a MP number, e.g., 1602 for Indiana [integer or string],
        or for an id string for unnumbered MPs only, e.g., ''. [string only]
    :param an_date: Astronight date representation, e.g., '20191106'. [int or string]
    :param filter: name of filter for this session, or
        None to use default from instrument file. [string]
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
    mp_directory = os.path.join(session_top_directory, 'MP_' + mp_string,
                                'AN' + an_string)
    os.chdir(mp_directory)
    print('Working directory set to:', mp_directory)

    # Initiate (or overwrite) log file:
    log_filename = defaults_dict['session log filename']
    with open(log_filename, mode='w') as log_file:
        log_file.write('Session Log File.' + '\n')
        log_file.write(mp_directory + '\n')
        log_file.write('MP: ' + mp_string + '\n')
        log_file.write('AN: ' + an_string + '\n')
        log_file.write('FILTER: ' + filter + '\n')
        log_file.write('This log started: ' +
                       '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n\n')
    print('Log file started.')
    print('Next: assess()')


def resume(session_top_directory=None, mp_id=None, an_date=None, filter=None):
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
    # TODO: add check for coherent RA,Dec positions; a jackknife test? (or at least a listing of RADecs).
    try:
        context = _get_session_context()
    except SessionLogFileError as e:
        print(' >>>>> ERROR: ' + str(e))
        return
    this_directory, mp_string, an_string, filter_string = context
    defaults_dict = ini.make_defaults_dict()
    df, return_dict = common.do_fits_assessments(defaults_dict, this_directory)

    # Summarize and write instructions for user's next steps:
    session_ini_filename = defaults_dict['session control filename']
    session_log_filename = defaults_dict['session log filename']
    session_log_fullpath = os.path.join(this_directory, session_log_filename)
    with open(session_log_fullpath, mode='a') as log_file:
        if return_dict['warning count'] == 0:
            print('\n >>>>> ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.')
            print('Next: (1) enter MP pixel positions in', session_ini_filename,
                  'AND SAVE it,\n      (2) make_dfs()')
            log_file.write('assess(): ALL ' + str(len(df)) +
                           ' FITS FILES APPEAR OK.' + '\n')
        else:
            print('\n >>>>> ' + str(return_dict['warning count']) +
                  ' warnings (see listing above).')
            print('        Correct these and rerun assess() until no warnings remain.')
            log_file.write('assess(): ' + str(return_dict['warning count']) +
                           ' warnings.' + '\n')

    df_temporal = df.loc[:, ['Filename', 'JD_mid']].sort_values(by=['JD_mid'])
    filenames_temporal_order = df_temporal['Filename']
    _write_session_ini_stub(this_directory, filenames_temporal_order)
    if return_results:
        return return_dict


def make_dfs(print_ap_details=False):
    """ Perform aperture photometry for one session of lightcurve photometry only.
        For color index determination, see color.make_dfs().
        :param print_ap_details: True if user wants aperture details for each MP. [bool]
        """
    context, defaults_dict, session_dict, log_file = _session_setup('make_dfs')
    this_directory, mp_string, an_string, filter_string = context
    instrument_dict = ini.make_instrument_dict(defaults_dict)
    instrument = util.Instrument(instrument_dict)
    disc_radius, gap, background_width = instrument.nominal_ap_profile
    site_dict = ini.make_site_dict(defaults_dict)

    fits_filenames = util.get_mp_filenames(this_directory)
    if not fits_filenames:
        raise SessionDataError(f'No FITS files found in '
                               f'session directory {this_directory}')

    # Quick validation of MP XY filenames & values:
    mp_xy_files_found, mp_xy_values_ok = \
        common.validate_mp_xy(fits_filenames, session_dict)
    if not mp_xy_files_found:
        raise SessionIniFileError(f'At least 1 MP XY file not found in '
                                  f'session directory {this_directory}')
    if not mp_xy_values_ok:
        raise SessionIniFileError('MP XY invalid -- did you enter values '
                                  'and save session.ini?')

    fits_objects, fits_object_dict = \
        common.make_fits_objects(this_directory, fits_filenames)
    df_images = common.make_df_images(fits_objects)

    # Get and screen catalog entries for comp stars:
    refcat2 = common.get_refcat2_comp_stars(fits_objects)
    info_lines = common.initial_screen_comps(refcat2)  # in-place screening.
    print('\n'.join(info_lines), '\n')
    log_file.write('\n'.join(info_lines) + '\n')

    # Make comp-star apertures, comps dataframe, and comp obs dataframe:
    df_comps = common.make_df_comps(refcat2)
    comp_apertures_dict = \
        common.make_comp_apertures(fits_objects, df_comps, disc_radius, gap,
                                   background_width)
    df_comp_obs = common.make_df_comp_obs(comp_apertures_dict, df_comps,
                                          instrument, df_images)

    # Make MP apertures and MP obs dataframe:
    starting_mp_obs_id = max(df_comp_obs['ObsID'].astype(int)) + 1
    mp_apertures_dict, mp_mid_radec_dict = \
        common.make_mp_apertures(fits_object_dict, mp_string,
                                 session_dict, disc_radius,
                                 gap, background_width, log_file,
                                 starting_obs_id=starting_mp_obs_id,
                                 print_ap_details=print_ap_details)
    df_mp_obs = common.make_df_mp_obs(mp_apertures_dict, mp_mid_radec_dict,
                                      instrument, df_images)

    # Post-process dataframes:
    _remove_images_without_mp_obs(fits_object_dict, df_images, df_comp_obs, df_mp_obs)
    common.add_obsairmass_df_comp_obs(df_comp_obs, site_dict, df_comps, df_images)
    common.add_obsairmass_df_mp_obs(df_mp_obs, site_dict, df_images)
    common.add_gr_color_df_comps(df_comps)
    common.add_ri_color_df_comps(df_comps)

    # Write dataframes to CSV files:
    common.write_df_images_csv(df_images, this_directory, defaults_dict, log_file)
    common.write_df_comps_csv(df_comps, this_directory, defaults_dict, log_file)
    common.write_df_comp_obs_csv(df_comp_obs, this_directory, defaults_dict, log_file)
    common.write_df_mp_obs_csv(df_mp_obs, this_directory, defaults_dict, log_file)

    log_file.close()
    print('\nNext: (1) enter comp selection limits and model options in ' +
          defaults_dict['session control filename'],
          '\n      (2) run do_session()\n')


def do_session():
    """ Primary lightcurve photometry for one session.
    Takes all data incl. color index, generates:
    Takes the 4 CSV files from make_dfs().
    Generates:
    * diagnostic plots for iterative regression refinement,
    * results in Canopus-import format,
    * ALCDEF-format file.
    Typically iterated, pruning comp-star ranges and outliers, until converged
        and then simply stop.
    NB: One may choose the FITS files by filter (typically 'Clear' or 'BB'), but
        * output lightcurve passband is fixed as Sloan 'r', and
        * color index is fixed as Sloan (r-i).
    :returns None. Writes all info to files.
    USAGE: do_session()   [no return value]
    """
    context, defaults_dict, session_dict, log_file = _session_setup('do_session')
    this_directory, mp_string, an_string, filter_string = context
    # log_filename = defaults_dict['session log filename']
    # log_file = open(log_filename, mode='a')
    mp_int = int(mp_string)  # put this in try/catch block.
    mp_string = str(mp_int)
    site_dict = ini.make_site_dict(defaults_dict)

    df_comp_master, df_mp_master = \
        common.make_df_masters(this_directory, defaults_dict,
                               filters_to_include=filter_string,
                               require_mp_obs_each_image=True,
                               data_error_exception_type=SessionDataError)
    df_model_raw = common.make_df_model_raw(df_comp_master)  # comps only.
    df_model = common.mark_user_selections(df_model_raw, session_dict)
    model = SessionModel(df_model, filter_string, session_dict, df_mp_master,
                         this_directory)

    _write_mpfile_line(mp_string, an_string, model)
    _write_canopus_file(mp_string, an_string, this_directory, model)
    _write_alcdef_file(mp_string, an_string, defaults_dict, session_dict, site_dict,
                       this_directory, model)
    _make_session_diagnostic_plots(model, df_model)



_____SUPPORT_for_make_dfs_____________________________________________ = 0


def _remove_images_without_mp_obs(fits_object_dict, df_images, df_comp_obs, df_mp_obs):
    for filename in fits_object_dict.keys():
        mp_obs_count = sum(df_mp_obs['FITSfile'] == filename)
        if mp_obs_count != 1:
            df_images = df_images.loc[df_images['FITSfile'] != filename, :]
            df_comp_obs = df_comp_obs.loc[df_comp_obs['FITSfile'] != filename, :]
            df_mp_obs = df_mp_obs.loc[df_mp_obs['FITSfile'] != filename, :]


_____SUPPORT_for_do_session______________________________________________ = 0


class SessionModel:
    """ Generates and holds mixed-model regression model, affords prediction for MP magnitudes. """
    def __init__(self, df_model, filter_string, session_dict, df_mp_master, this_directory):
        self.df_model = df_model
        self.filter_string = filter_string
        self.session_dict = session_dict
        self.df_used_comps_obs = \
            self.df_model.copy().loc[self.df_model['UseInModel'], :]
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
        if transform_option == ('fit=1',):
            fixed_effect_var_list.append('CI')
            msg = ' Transform first-order fit to Color Index.'
        elif transform_option == ('fit=2',):
            fixed_effect_var_list.extend(['CI', 'CI2'])
            msg = ' Transform second-order fit to Color Index.'
        elif transform_option[0] == 'use':
            if len(transform_option) == 2:
                transform_offset = float(transform_option[1]) * self.df_used_comps_obs['CI']
                msg = ' Transform (Color Index) not fit: 1st-order value fixed at' + \
                      transform_option[1]
            elif len(transform_option) == 3:
                transform_offset = float(transform_option[1]) * self.df_used_comps_obs['CI'] +\
                                   float(transform_option[2]) * self.df_used_comps_obs['CI2']
                msg = ' Transform (Color Index) not fit: 1st, 2nd order values fixed at' + \
                      transform_option[1] + transform_option[2]
            else:
                raise SessionSpecificationError('Invalid \'Fit Transform\' option in session.ini')
            dep_var_offset += transform_offset

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
        print('mean ri color =', '{0:.3f}'.format(self.df_used_comps_obs['ri_color'].mean()))
        if not self.mm_fit.converged:
            msg = ' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.'
            print(msg)
            fit_summary_lines.append(msg)

        common.write_text_file(self.this_directory, 'fit_summary.txt',
                               'Regression (mp2021) for: ' + self.this_directory + '\n\n' +
                               '\n'.join(fit_summary_lines) + '\n\n' +
                               self.mm_fit.statsmodels_object.summary().as_text() +
                               '\ncomps = ' + str(n_comps_used) + ' used' +
                               '\nsigma = ' + '{0:.1f}'.format(1000.0 * self.mm_fit.sigma) + ' mMag.')

    def _calc_mp_mags(self):
        """ Use model and MP instrument magnitudes to get best estimates of MP absolute magnitudes."""
        bogus_cat_mag = 0.0  # we'll need this below, to correct raw predictions.
        self.df_used_mp_obs = self.df_used_mp_obs.copy(deep=True)  # to shut up pandas & its fake warnings.
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
                   x_data=df_plot_comp_obs['r'], y_data=1000.0 * df_plot_comp_obs['InstMagSigma'])
    ax.scatter(x=model.df_mp_mags['MP_Mags'], y=1000.0 * model.df_mp_mags['InstMagSigma'],
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


def _write_alcdef_file(mp_string, an_string, defaults_dict, session_dict, site_dict, this_directory, model):
    """ Write ALCDEF file for one MP photometry (lightcurve) session.
        Service function called by do_session()--separate invocation not needed.
        Corrected for ALCDEF format clarifications by Brian Warner, via e-mails of June-July 2021.
    :return: [none]
    """
    mpfile_names = util.all_mpfile_names(defaults_dict['mpfile directory'])
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

    lines.append('COMMENT=ALCDEF file for MP ' + mp_string + '  AN ' + an_string + ' (mp2021 workflow)')
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
    lines.append('EQUINOX=J2000.0')  # instead of EPOCH, per Warner e-mail of 2021-07-02.
    lines.append('PHASE=+' + '{0:.1f}'.format(abs(session_eph_dict['Phase'])))
    lines.append('PABL=' + '{0:.1f}'.format(abs(session_eph_dict['PAB_longitude'])))
    lines.append('PABB=' + '{0:.1f}'.format(abs(session_eph_dict['PAB_latitude'])))
    lines.append('COMMENT=These results from submitter\'s ATLAS-refcat2 based workflow')
    lines.append('COMMENT=as described in the authors presentations for SAS Symposium 2020 and 2021,')
    # (For next line: ALCDEF does not allow web addresses even in comments.)
    lines.append('COMMENT=using code publicly available at: github website, user=edose, repo=mp2021.')
    lines.append('COMMENT=This session used ' +
                 str(len(model.df_used_comps_obs['CompID'].drop_duplicates())) +
                 ' comp stars. COMPNAME etc lines are omitted.')
    lines.append('DIFFERMAGS=FALSE')
    lines.append('STANDARD=TRANSFORMED')
    lines.append('CICORRECTION=TRUE')
    lines.append('CIBAND=SRI')
    lines.append('CITARGET=' + '{0:+.3f}'.format(session_dict['mp ri color']))
    lines.append('COMMENT=CITARGET is from : ' + session_dict['mp ri color origin'])
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


def combine_alcdef_files(mp, apparition_year, top_directory):
    """ Append into one file: all ALCDEF files in one MP Campaign directory, then write to MP directory.
        Text-combining function only; does not change any data.
    :param mp: MP number. A subdirectory 'MP_[mpnumber]' must exist in mp_phot_top_directory. [int or str]
    :param apparition_year: used to name output file. [string or int]
    :param top_directory: directory just above the MP_nnnn directory containing alcdef files.
           Is usually the default, or like 'C:/Astro/MP Photometry/MPBs In Press/For MPB 48-1/'. [string]
    Before use: Be VERY sure that unused session directories are removed to 'Exclude' subdir.
    Usage: combine_alcdef_files(1604, 2020) or
           combine_alcdef_files(1604, 2020, 'C:/Astro/MP Photometry'
    After use: Verify combined file with ALCDEF Verify web-based checker (on alcdef.org).
    :return: None. Writes new file to MP's directory.
    """
    mpdir = os.path.join(top_directory, 'MP_' + str(mp))
    an_subdirs = [f.path for f in os.scandir(mpdir)
                  if (f.is_dir() and f.name.startswith('AN') and len(f.name) == 10)]
    all_lines = []
    n_files_read = 0
    for sd in an_subdirs:
        filenames = [f.path for f in os.scandir(sd)
                     if (f.name.startswith('alcdef_MP_') and f.name.endswith('.txt'))]
        for filename in filenames:
            with open(filename, 'r') as f:
                lines = f.readlines()
            # all_lines.append('COMMENT=################################'
            #                  '######################################\n')
            all_lines.extend(lines)
            print('{0:6d}'.format(len(lines)), 'lines read from', filename)
            n_files_read += 1
    if n_files_read <= 0:
        print(' >>>>> WARNING: No ALCDEF files were found.')
        exit(0)

    # Write combined ALCDEF file to MP directory:
    combined_filename = 'alcdef_combined_MP_' + str(mp) + '_' + str(apparition_year) + '.txt'
    fullpath = os.path.join(mpdir, combined_filename)
    with open(fullpath, 'w') as f:
        f.writelines(all_lines)
    print(str(len(all_lines)), 'total lines from', str(len(an_subdirs)),
          'files written to top_directory/' + combined_filename + '.')


_____SUPPORT_for_do_transform______________________________________________ = 0


def do_transform(filter_string: str = 'BB',
                 target_passband: str = 'SG',
                 color_index: Tuple[str, str] = ('SG', 'SR'),
                 fit_order: int = 1,
                 ri_color_range: Tuple[float, float] = (0, 0.5)):
    # Setup, about the same as for do_session():
    context, defaults_dict, session_dict, _ = _session_setup('do_session')
    this_directory, mp_string, an_string, _ = context
    # log_filename = defaults_dict['session log filename']
    # log_file = open(log_filename, mode='a')
    # mp_int = int(mp_string)  # put this in try/catch block.
    # mp_string = str(mp_int)
    # site_dict = ini.make_site_dict(defaults_dict)
    transform_dict = copy.deepcopy(session_dict)

    # Modify parameters for transform extraction:
    if isinstance(ri_color_range, tuple) and len(ri_color_range) == 2:
        transform_dict['min catalog ri color'] = ri_color_range[0]
        transform_dict['max catalog ri color'] = ri_color_range[1]
    print(f"Using ri color range = "
          f"{transform_dict['min catalog ri color']} - "
          f"{transform_dict['max catalog ri color']}")

    # Make master dataframes:
    df_comp_master, df_mp_master = \
        common.make_df_masters(this_directory, defaults_dict,
                               filters_to_include=filter_string,
                               require_mp_obs_each_image=False,
                               data_error_exception_type=SessionDataError)
    df_model_raw = common.make_df_model_raw(df_comp_master)  # comps only.
    df_model = common.mark_user_selections(df_model_raw, transform_dict)

    model = TransformModel(df_model, filter_string, target_passband, color_index,
                           fit_order, transform_dict, df_mp_master, this_directory)

    id_string = f'{filter_string}_{target_passband}_'\
                f'{color_index[0]}-{color_index[1]}'
    _write_canopus_file(id_string, an_string, this_directory, model)
    _make_session_diagnostic_plots(model, df_model)


class TransformModel:
    """ Generates and holds mixed-model regression model, yields transform estimate. """
    def __init__(self, df_model: pd.DataFrame,
                 filter_string: str = 'BB',
                 passband: str = 'SR',
                 color_index: Tuple[str, str] = ('SR', 'SI'),
                 fit_order: int = 1,
                 transform_dict: dict = None,
                 df_mp_master: pd.DataFrame = None,
                 this_directory: str = None):
        self.df_model = df_model
        self.filter = filter_string
        self.passband = passband
        self.color_index = color_index
        self.fit_order = fit_order
        self.transform_dict = transform_dict
        self.df_used_comps_obs = \
            self.df_model.copy().loc[self.df_model['UseInModel'], :]
        images_in_used_comps = self.df_used_comps_obs['FITSfile'].drop_duplicates()
        mp_rows_to_use = df_mp_master['FITSfile'].isin(images_in_used_comps)
        self.df_used_mp_obs = df_mp_master.loc[mp_rows_to_use, :]
        self.this_directory = this_directory

        self.dep_var_name = 'InstMag_with_offsets'
        self.mm_fit = None      # placeholder for the fit result [MixedModelFit object].
        self.transform = None   # placeholder for this fit parameter result [scalar].
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
            Model's .predict() is not really used for transform estimation.
        :return: [None]
        """
        fit_summary_lines = []

        # Handle passband option (from do_transform() parameter):
        catalog_passband = PASSBAND_NAMES[self.passband]
        dep_var_offset = self.df_used_comps_obs[catalog_passband].copy()
        fixed_effect_var_list = []

        # Handle transform (Color Index) option (from do_transform() parameter):
        if self.color_index == ('SG', 'SR'):
            ci = 'gr_color'
        elif self.color_index == ('SR', 'SI'):
            ci = 'ri_color'
        else:
            raise ValueError(f'The given color index of '
                             f'\'{self.color_index}\' not allowed.')

        # Handle fit order option from do_transform() parameter,
        # overriding anything in the session.ini file:
        if self.fit_order not in [1, 2]:
            raise ValueError(f'Fit order is {self.fit_order} but must be 1 or 2.')
        self.df_used_comps_obs['CI'] = self.df_used_comps_obs[ci]
        fixed_effect_var_list.append('CI')
        if self.fit_order == 2:
            self.df_used_comps_obs['CI2'] = \
                [ci ** 2 for ci in self.df_used_comps_obs['CI']]
            fixed_effect_var_list.append(['CI2'])

        # Handle extinction (ObsAirmass) option (from session.ini file):
        # Option chosen from: 'yes', ('use', [ext]).
        extinction_option = self.transform_dict['fit extinction']
        if extinction_option == 'yes':
            fixed_effect_var_list.append('ObsAirmass')
            msg = ' Extinction fit on ObsAirmass data.'
        elif isinstance(extinction_option, tuple) and len(extinction_option) == 2 and \
            (extinction_option[0] == 'use'):
            extinction_offset = float(extinction_option[1]) * \
                                self.df_used_comps_obs['ObsAirmass']
            dep_var_offset += extinction_offset
            msg = ' Extinction (ObsAirmass) not fit: value fixed at default of' + \
                  ' {0:.3f}'.format(float(extinction_option[1]))
        else:
            raise SessionSpecificationError('Invalid \'Fit Extinction\' '
                                            'option in session.ini')

        # Build all other fixed-effect (x) variable lists and dep-var offsets:
        if self.transform_dict['fit vignette']:
            fixed_effect_var_list.append('Vignette')
        if self.transform_dict['fit xy']:
            fixed_effect_var_list.extend(['X1024', 'Y1024'])
        if self.transform_dict['fit jd']:
            fixed_effect_var_list.append('JD_fract')

        # Build 'random-effect' and dependent (y) variables:
        random_effect_var_name = 'FITSfile'  # cirrus effect is per-image
        self.df_used_comps_obs[self.dep_var_name] = \
            self.df_used_comps_obs['InstMag'] - dep_var_offset

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
        print('mean ri color =', '{0:.3f}'
              .format(self.df_used_comps_obs['ri_color'].mean()))
        if not self.mm_fit.converged:
            msg = ' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.'
            print(msg)
            fit_summary_lines.append(msg)

        common.write_text_file(self.this_directory, 'fit_summary.txt',
                               'Regression (mp2021) for: ' + self.this_directory + '\n\n' +
                               '\n'.join(fit_summary_lines) + '\n\n' +
                               self.mm_fit.statsmodels_object.summary().as_text() +
                               '\ncomps = ' + str(n_comps_used) + ' used' +
                               '\nsigma = ' + '{0:.1f}'.format(1000.0 * self.mm_fit.sigma) + ' mMag.')

    def _calc_mp_mags(self):
        """ Use model and MP instrument magnitudes to get best estimates of MP absolute magnitudes."""
        bogus_cat_mag = 0.0  # we'll need this below, to correct raw predictions.
        self.df_used_mp_obs = \
            self.df_used_mp_obs.copy(deep=True)  # shut up pandas & its fake warnings.
        self.df_used_mp_obs['CatMag'] = \
            bogus_cat_mag  # totally bogus local value, corrected for later.
        best_mp_ri_color = self.transform_dict['mp ri color']
        self.df_used_mp_obs['CI'] = best_mp_ri_color
        self.df_used_mp_obs['CI2'] = best_mp_ri_color ** 2
        raw_predictions = self.mm_fit.predict(self.df_used_mp_obs,
                                              include_random_effect=True)
        dep_var_offset = pd.Series(len(self.df_used_mp_obs) * [0.0],
                                   index=raw_predictions.index)

        # No handling needed for transform (is always a term in this regression).

        # Handle extinction offsets for MPs:
        extinction_option = self.transform_dict['fit extinction']
        if isinstance(extinction_option, tuple):
            extinction_offset = float(extinction_option[1]) * \
                                self.df_used_mp_obs['ObsAirmass']
            dep_var_offset += extinction_offset

        # Calculate best MP magnitudes, incl. effect of assumed bogus_cat_mag:
        mp_mags = self.df_used_mp_obs['InstMag'] - dep_var_offset - \
            raw_predictions + bogus_cat_mag
        df_mp_mags = pd.DataFrame(data={'MP_Mags': mp_mags}, index=list(mp_mags.index))
        df_mp_mags = \
            pd.merge(left=df_mp_mags,
                     right=self.df_used_mp_obs.loc[:, ['JD_mid', 'FITSfile',
                                                       'InstMag', 'InstMagSigma']],
                     how='left', left_index=True, right_index=True, sort=False)
        return df_mp_mags




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
    ax.set_ylabel(y_label, labelpad=0)  # "
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
    df_comp_offsets = df_plot_comp_obs.loc[:, ['CompID', 'ObsID', 'JD_fract', 'InstMag_with_offsets']]

    # Get normalized offsets (jackknife differences) (code simplified vs. mpc):
    all_jd_fracts = df_comp_offsets['JD_fract'].copy().drop_duplicates().sort_values()
    df_comp_offsets['NormalizedOffset'] = None
    df_comp_offsets['LatestNormalizedOffset'] = None
    for jd_fract in all_jd_fracts:
        is_jd_fract = (df_comp_offsets['JD_fract'] == jd_fract)
        obs_ids = df_comp_offsets.loc[is_jd_fract, 'ObsID']
        inst_mag_count = sum(is_jd_fract == True)
        inst_mag_sum = sum(df_comp_offsets.loc[is_jd_fract, 'InstMag_with_offsets'])
        for obs_id in obs_ids:
            this_inst_mag = df_comp_offsets.loc[obs_id, 'InstMag_with_offsets']
            mean_other_inst_mags = (inst_mag_sum - this_inst_mag) / (inst_mag_count - 1)
            df_comp_offsets.loc[obs_id, 'NormalizedOffset'] = this_inst_mag - mean_other_inst_mags

    # Get latest normalized offset for each comp (plotting aid):
    all_comp_ids = df_comp_offsets['CompID'].copy().drop_duplicates()
    for comp_id in all_comp_ids:
        is_comp_id = (df_comp_offsets['CompID'] == comp_id)
        latest_jd_fract = max(df_comp_offsets.loc[is_comp_id, 'JD_fract'])
        is_latest_jd_fract = (df_comp_offsets['JD_fract'] == latest_jd_fract)
        latest_normalized_offset = df_comp_offsets.loc[is_comp_id & is_latest_jd_fract, 'NormalizedOffset']
        df_comp_offsets.loc[is_comp_id, 'LatestNormalizedOffset'] = latest_normalized_offset.iloc[0]

    # Add plot_index, so that comps are plotted in decreasing order of offsets:
    df_comp_offsets = df_comp_offsets.sort_values(by=['LatestNormalizedOffset', 'CompID', 'JD_fract'],
                                                  ascending=[False, True, True])
    df_plot_index = df_comp_offsets[['CompID']].drop_duplicates()
    df_plot_index['PlotIndex'] = range(len(df_plot_index))
    df_plot_index.index = list(df_plot_index['PlotIndex'])
    df_comp_offsets = pd.merge(left=df_comp_offsets, right=df_plot_index,
                               how='left', on='CompID', sort=False)

    # Plot the normalized offsets vs JD for each comp_id, 4 comp_ids to a subplot:
    n_cols, n_rows = 3, 3
    n_plots_per_figure = n_cols * n_rows
    n_comps_per_plot = 4
    n_comps = len(df_plot_index)
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
                        this_comp_id = df_plot_index.loc[i_plot_index, 'CompID']
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
    all_comp_ids = df_plot_index['CompID']
    all_comps_plotted_once = (sorted(plotted_comp_ids) == sorted(all_comp_ids))
    if not all_comps_plotted_once:
        print('comp ids plotted more than once',
              [item for item, count in Counter(plotted_comp_ids).items() if count > 1])


def draw_x_line(ax, x_value, color='lightgray'):
    """ Draw vertical line, typically to mark MP transit time. """
    ax.axvline(x=x_value, color=color, linewidth=1, zorder=-100)


_____SUPPORT_FUNCTIONS_and_CLASSES________________________________________ = 0


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
        'Omit Comps = ',
        'Omit Obs = ',
        '# One image only per line, with or without .fts:',
        'Omit Images = ',
        'Min Catalog r mag = 11.5',
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
        '# Clear:\'Use +0.39 -0.71\'    BB:\'Use -0.121\'',
        'Fit Transform = Use -0.135',
        '# Fit Extinction, one of: Yes, Use [val]:',
        'Fit Extinction = Use +0.13',
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


# def update_alcdef_files(mp):
#     """ Updates ALCDEF txt file for each AN directory in MP directory.
#         To produce updates of July 2021, for ALCDEF requirements.
#     """
#     # Make list of AN directories:
#     mp_dir = 'C:/Astro/MP Photometry/For MPB 48-4 July 15/MP_' + str(mp)
#     dir_list = []
#     for file in os.listdir(mp_dir):
#         d = os.path.join(mp_dir, file)
#         if os.path.isdir(d):
#             if (d[-10:])[:5] == 'AN202':  # must be a bona fide AN subdir.
#                 dir_list.append(d)
#                 # print(' >>>>>' + d + '<')
#
#     # For each AN directory: read ALCDEF file, update it, write it back out:
#     for d in dir_list:
#         an_str = d[-8:]
#         fullpath = os.path.join(d, 'alcdef_MP_' + str(mp) + '_' + an_str + '.txt')
#         print('>' + fullpath + '<')
#         with open(fullpath, mode='r') as f:
#             lines = f.readlines()
#
#         # 1. Insert DIFFERMAGS= and STANDARD= lines:
#         line_num = min([n for (n, line) in enumerate(lines) if line.startswith('CICORRECTION=')])
#         lines[line_num:line_num] = ['DIFFERMAGS=FALSE\n', 'STANDARD=TRANSFORMED\n']
#
#         # 2. Replace 3 COMMENT= lines with new COMMENT= lines:
#         line_num = min([n for (n, line) in enumerate(lines) if line.startswith('COMMENT=')])
#         lines[line_num:line_num + 3] = [
#             'COMMENT=These results from submitter\'s ATLAS-refcat2 based workflow\n',
#             'COMMENT=as described in the author\'s presentations for SAS Symposium 2020 and 2021,\n',
#             'COMMENT=using code publicly available at: github website, user=edose, repo=mp2021.\n',
#             'COMMENT=This session used 25 comp stars. COMPNAME etc lines are omitted.\n']
#
#         # 3. Add Comment (hashed/ignore) line:
#         lines[1:1] = ['# As modified 2021-07-15, using update_alcdef_files().\n']
#
#         # 4. Insert new EQUINOX= line:
#         line_num = min([n for (n, line) in enumerate(lines) if line.startswith('OBJECTDEC=')])
#         lines[line_num+1:line_num+1] = ['EQUINOX=J2000.0\n']
#
#         with open(fullpath, mode='w') as f:
#             f.writelines(lines)
