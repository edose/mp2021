""" This module:
"""

__author__ = "Eric Dose, Albuquerque"

# Python core:
import os
from datetime import datetime, timezone, timedelta
from math import pi, cos, floor
from copy import deepcopy
from typing import Tuple, List

# External packages:
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

# Author's packages:
import astropak.util
from astropak.stats import MixedModelFit
import mp2021.util as util
import mp2021.ini as ini
import mp2021.common as common
from mp2021.session import make_qq_plot_fullpage, make_9_subplot, draw_x_line, \
    make_comp_variability_plots

THIS_PACKAGE_ROOT_DIRECTORY = \
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')
COLOR_SUMMARY_FILENAME = 'color_summary.txt'
COLOR_PLOT_FILE_PREFIX = 'Image_Color_'


class ColorIniFileError(Exception):
    """ Raised on any fatal problem with color ini file."""


class ColorLogFileError(Exception):
    """ Raised on any fatal problem with color log file."""


class ColorDataError(Exception):
    """ Raised on any fatal problem with data, esp. with contents of FITS files."""


class ColorSpecificationError(Exception):
    """ Raised on any fatal problem in specifying the color processing. """


def start(color_top_directory: str = None, mp_id: int = None, an_date: int = None,
          color_def_filename: str = 'ColorDef_Sloan2colors_from_Sloan.ini') -> None:
    """ Launch one MP color workflow (one color subdirectory).
        Adapted from package mp2021, session.start().
        Example usage: color.start('C:/Astro/MP Color/', 1111, 20210617)
    :param color_top_directory: path of lowest directory common to all MP color FITS,
        e.g., 'C:/Astro/MP Color'. None -> .ini file default (normal case). [string]
    :param mp_id: either a MP number, e.g., 1602 for Indiana [integer or string], or
        for an id string for unnumbered MPs only, e.g., ''. [string only]
    :param an_date: Astronight date representation, e.g., '20191106'. [int or string]
    :param color_def_filename: file containing color workflow definition. [string]
    :return: [None]
    """
    defaults_dict = ini.make_defaults_dict()
    if color_top_directory is None:
        color_top_directory = defaults_dict['color top directory']
    if mp_id is None or an_date is None:
        print(' >>>>> Usage: start(color_top_directory, mp_id, an_date)')
        return
    mp_string = util.parse_mp_id(mp_id)
    an_string = util.parse_an_date(an_date)

    # Construct directory path, and make it the working directory:
    mp_directory = os.path.join(color_top_directory, 'MP_' + mp_string,
                                'AN' + an_string)
    os.chdir(mp_directory)
    print(f'Working directory set to: {mp_directory}')

    # Initiate (or overwrite) log file:
    log_filename = defaults_dict['color log filename']
    with open(log_filename, mode='w') as log_file:
        log_file.write('Color Log File.\n')
        log_file.write(f'MP directory: {mp_directory}\n')
        log_file.write(f'MP: {mp_string}\n')
        log_file.write(f'AN: {an_string}\n')
        log_file.write(f'Color definition file: {color_def_filename}\n')
        log_file.write('This log started: ' +
                       '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) +
                       '\n\n')
    print('Log file started.')
    print('Next: assess()')


def resume(color_top_directory: str = None, mp_id: int = None, an_date: int = None) \
        -> None:
    """ Restart a color workflow in its correct working directory,
        but *keep* the previous log file--DO NOT overwrite it.
        Adapted from package mp2021, session.resume().
        Example usage: color.resume('C:/Astro/MP Photometry/', 1111, 20200617)
    Parameters are exactly as for color.start().
    :return: [None]
    """
    defaults_dict = ini.make_defaults_dict()
    if color_top_directory is None:
        color_top_directory = defaults_dict['color top directory']
    if mp_id is None or an_date is None:
        print(' >>>>> Usage: resume(top_directory, mp_id, an_date)')
        return
    mp_string = util.parse_mp_id(mp_id).upper()
    an_string = util.parse_an_date(an_date)

    # Construct directory path and make it the working directory:
    this_directory = os.path.join(color_top_directory, 'MP_' + mp_string,
                                  'AN' + an_string)
    os.chdir(this_directory)
    print('Working directory set to:', this_directory)
    log_this_directory, log_mp_string, log_an_string, _ = _get_color_context()
    if log_mp_string.upper() != mp_string:
        raise ColorLogFileError(' '.join(['MP string does not match that of color log.',
                                          log_mp_string, mp_string]))
    if log_an_string != an_string:
        raise ColorLogFileError(' '.join(['AN string does not match that '
                                          'of session log.', log_an_string, an_string]))
    print('Resuming in', this_directory)


def assess(return_results: bool = False) -> None:
    """ First, verify that all required files are in this color directory or
            otherwise accessible.
        Then, perform checks on FITS files in this directory before performing
            color photometry proper.
         Modeled after and extended from assess() found in variable-star photometry
             package 'photrix'.
        :return: [None], or dict of summary info and warnings. [py dict]
    """
    context = _get_color_context()
    this_directory, mp_string, an_string, color_definition_filename = context
    defaults_dict = ini.make_defaults_dict()
    df, return_dict = common.do_fits_assessments(defaults_dict, this_directory)

    # Summarize and write instructions for user's next steps:
    color_ini_filename = defaults_dict['color control filename']
    color_log_filename = defaults_dict['color log filename']
    color_log_fullpath = os.path.join(this_directory, color_log_filename)
    with open(color_log_fullpath, mode='a') as log_file:
        if return_dict['warning count'] == 0:
            print('\n >>>>> ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.')
            print('Next: (1) enter MP pixel positions in', color_ini_filename,
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
    mean_datetime = astropak.util.datetime_utc_from_jd(df['JD_mid'].mean())
    _write_color_ini_stub(this_directory, filenames_temporal_order, mean_datetime,
                          color_definition_filename)
    if return_results:
        return return_dict


def make_dfs(print_ap_details: bool = False) -> None:
    """ Perform aperture photometry for one subdirectory of color photometry.
        Makes 4 dataframes.
        :param print_ap_details: True if user wants aperture details for each MP. [bool]
        """
    context, defaults_dict, color_dict, color_def_dict, log_file = \
        _color_setup('make_dfs')
    this_directory, mp_string, an_string, color_defintion_filename = context
    instrument_dict = ini.make_instrument_dict(defaults_dict)
    instrument = util.Instrument(instrument_dict)
    disc_radius, gap, background_width = instrument.nominal_ap_profile
    site_dict = ini.make_site_dict(defaults_dict)

    fits_filenames = util.get_mp_filenames(this_directory)
    if not fits_filenames:
        raise ColorDataError('No FITS files found in color directory ' + this_directory)

    # Quick validation of MP XY filenames & values:
    mp_xy_files_found, mp_xy_values_ok = \
        common.validate_mp_xy(fits_filenames, color_dict)
    if not mp_xy_files_found:
        raise ColorIniFileError('At least 1 MP XY file not found in color directory ' +
                                this_directory)
    if not mp_xy_values_ok:
        raise ColorIniFileError('MP XY invalid -- did you enter '
                                'values and save color.ini?')

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
                                 color_dict, disc_radius, gap,
                                 background_width, log_file,
                                 starting_obs_id=starting_mp_obs_id,
                                 print_ap_details=print_ap_details)
    df_mp_obs = common.make_df_mp_obs(mp_apertures_dict, mp_mid_radec_dict,
                                      instrument, df_images)

    # WANTED??: _remove_images_without_mp_obs(fits_object_dict, df_images,
    #     df_comp_obs, df_mp_obs)
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
    print('\nNext: (1) ' + defaults_dict['color control filename'] +
          ': enter comp selection limits and regression model options.',
          '\n      (2) run do_color_stepwise()\n')


# TODO: verify that the two regressions here agree with March 2023 re-derivations.
def do_color(requested_comp_id: str = None) -> None:
    """ Primary color photometry for one color subdirectory.
    Takes the 4 CSV files from make_dfs().
    Gets MP colors in two steps:
        1. color_model_1: get (untransformed) MP mags for all passbands.
        2. color_model_2: get MP colors using simple regression on MP mags.
    Generates diagnostic plots for iterative regression refinement.
    Typically iterated, pruning comp-star ranges and outliers, until converged and
        then simply stop.
    :param requested_comp_id: A comp star id (from ATLAS), given only if user wants
        to use a comp star place of the minor planet. For testing only. [str]
    :returns None. Writes all info to files.
    USAGE: do_color()   [no return value]
    """
    context, defaults_dict, color_dict, color_def_dict, log_file = \
        _color_setup('do_color')
    this_directory, mp_string, an_string, _ = context
    filters_to_include = list(color_def_dict['filters'].keys())
    # instrument_dict = ini.make_instrument_dict(defaults_dict)
    # instrument = Instrument(instrument_dict)
    # disc_radius, gap, background_width = instrument.nominal_ap_profile
    # site_dict = ini.make_site_dict(defaults_dict)

    df_comp_master, df_mp_master = \
        common.make_df_masters(this_directory, defaults_dict,
                               filters_to_include=filters_to_include,
                               require_mp_obs_each_image=True,
                               data_error_exception_type=ColorDataError)
    # Get comps-only data from all used filters:
    df_model_raw = common.make_df_model_raw(df_comp_master)
    df_model = common.mark_user_selections(df_model_raw, color_dict)
    with open(COLOR_SUMMARY_FILENAME, mode='a') as sf:
        sf.write('Color Determination (mp2021) for: MP ' + mp_string +
                 '  AN ' + an_string + '\n')
        sf.flush()

    # #################### TEST ONLY:
    if requested_comp_id is not None:
        color_dict, df_mp_master = _replace_mp_with_a_comp(color_dict, df_comp_master,
                                                           df_mp_master,
                                                           requested_comp_id)
    # #################### End TEST ONLY code.

    color_model_1 = ColorModel_1(df_model, color_dict, color_def_dict,
                                 df_mp_master, this_directory)
    color_model_2 = ColorModel_2(color_model_1.df_untransformed_mp_mags)
    _make_color_diagnostic_plots(df_model, color_model_1, color_model_2)

    # #################### TEST ONLY:
    if requested_comp_id is not None:
        with open(COLOR_SUMMARY_FILENAME, mode='a') as sf:
            sf.write(color_model_1.return_text)
            sf.write(color_model_2.return_text + '\n\n\n')
            sf.flush()
    # #################### End TEST ONLY code.


__________SUPPORT_for_do_color____________________________________________ = 0


class ColorModel_1:
    """ Generate and hold mixed-model regression model suitable to
            MP color index determination.
        Make estimates ("predictions") for partial MP magnitudes (partial,
            because still need to be adjusted for MP color index,
            which we don't know until we've run ColorModel_2.
    """
    def __init__(self, df_model: pd.DataFrame, color_dict: dict, color_def_dict: dict,
                 df_mp_master: pd.DataFrame, this_directory: str) -> None:
        self.df_model = df_model
        self.color_dict = color_dict
        self.color_def_dict = color_def_dict
        self.df_used_comp_obs = self.df_model.copy().loc[self.df_model['UseInModel'], :]
        images_in_used_comps = self.df_used_comp_obs['FITSfile'].drop_duplicates()
        mp_rows_to_use = df_mp_master['FITSfile'].isin(images_in_used_comps)
        self.df_used_mp_obs = df_mp_master.loc[mp_rows_to_use, :]
        self.this_directory = this_directory

        self.dep_var_name = 'InstMag_with_offsets'
        self.mm_fit = None      # placeholder for fit result [a MixedModelFit object].
        self.vignette = None    # placeholder for this fit parameter results [a scalar].

        self.return_text = ''
        self._prep_and_do_regression()
        self.df_untransformed_mp_mags = self._calc_mp_mags()

    def _prep_and_do_regression(self) -> None:
        """ Using MixedModelFit class (which wraps statsmodels.MixedLM.from_formula()).
            Uses ONLY selected comp data in this model.
            (Use model's .predict() to calculate best MP mags from model and MP obs.)
        :return: [None]
        """
        fit_summary_lines = []
        # fixed_effect_var_list = []
        filters_to_include = list(self.color_def_dict['filters'].keys())

        # Prepare catmag offset to dep var:
        self.df_used_comp_obs['CatMag'] = 0.0
        for f in filters_to_include:
            catmag_passband = self.color_def_dict['filters'][f]['target passband']
            catmag_passband_column_name = common.PASSBAND_COLUMN_LOOKUP[catmag_passband]
            is_in_filter = [filt == f for filt in self.df_used_comp_obs['Filter']]
            self.df_used_comp_obs.loc[is_in_filter, 'CatMag'] = \
                self.df_used_comp_obs.loc[is_in_filter, catmag_passband_column_name]
        catmag_offset = self.df_used_comp_obs['CatMag']

        # Prepare transform*color offset to dep var, from instrument ini file:
        # Do not use None in the next lines, as None puts Object type into DataFrame.
        self.df_used_comp_obs['TransformValue'] = np.NAN
        self.df_used_comp_obs['TransformColor'] = np.NAN
        self.df_used_mp_obs['TransformValue'] = np.NAN
        for f in filters_to_include:
            transform_value = self.color_dict['transforms'][f]
            transform_ci = self.color_def_dict['filters'][f]['transform ci']
            transform_ci_column_name = common.COLOR_COLUMN_LOOKUP[transform_ci]
            is_in_filter = [filt == f for filt in self.df_used_comp_obs['Filter']]
            self.df_used_comp_obs.loc[is_in_filter, 'TransformValue'] = transform_value
            self.df_used_comp_obs.loc[is_in_filter, 'TransformColor'] = \
                self.df_used_comp_obs.loc[is_in_filter, transform_ci_column_name]
            is_in_filter = [filt == f for filt in self.df_used_mp_obs['Filter']]
            self.df_used_mp_obs.loc[is_in_filter, 'TransformValue'] = transform_value
        transform_offset = (self.df_used_comp_obs['TransformValue'] *
                            self.df_used_comp_obs['TransformColor'])

        # Prepare extinction*airmass offset to dep var:
        self.df_used_comp_obs['ExtinctionValue'] = np.NAN
        self.df_used_mp_obs['ExtinctionValue'] = np.NAN
        for f in filters_to_include:
            extinction_value = self.color_dict['extinctions'][f]
            is_in_filter = [filt == f for filt in self.df_used_comp_obs['Filter']]
            self.df_used_comp_obs.loc[is_in_filter, 'ExtinctionValue'] = \
                extinction_value
            is_in_filter = [filt == f for filt in self.df_used_mp_obs['Filter']]
            self.df_used_mp_obs.loc[is_in_filter, 'ExtinctionValue'] = extinction_value
        extinction_offset = (self.df_used_comp_obs['ExtinctionValue'] *
                             self.df_used_comp_obs['ObsAirmass'])

        # Prepare offsets with all components except zero-point offsets (not known yet):
        partial_offset = catmag_offset + transform_offset + extinction_offset
        instmag_partially_offset = self.df_used_comp_obs['InstMag'] - partial_offset

        # Prepare *approximate* per-filter Zero-pt (from main filter) offset to dep var,
        #     so that mixed-model regression's random errors are indeed close to random:
        # Get mean offsets for each filter:
        mean_inst_mags = {}
        for f in filters_to_include:
            is_in_filter = [filt == f for filt in self.df_used_comp_obs['Filter']]
            mean_inst_mags[f] = instmag_partially_offset[is_in_filter].mean()
        ref_filter = self.color_def_dict['reference filter']
        self.df_used_comp_obs['ZeroPointOffset'] = np.NAN
        self.df_used_mp_obs['ZeroPointOffset'] = np.NAN
        for f in filters_to_include:
            instmag_fully_offset = mean_inst_mags[f] - mean_inst_mags[ref_filter]
            is_in_filter = [filt == f for filt in self.df_used_comp_obs['Filter']]
            self.df_used_comp_obs.loc[is_in_filter, 'ZeroPointOffset'] = \
                instmag_fully_offset
            is_in_filter = [filt == f for filt in self.df_used_mp_obs['Filter']]
            self.df_used_mp_obs.loc[is_in_filter, 'ZeroPointOffset'] = \
                instmag_fully_offset
        zeropoint_offset = self.df_used_comp_obs['ZeroPointOffset']
        total_offset = partial_offset + zeropoint_offset
        self.df_used_comp_obs[self.dep_var_name] = \
            (self.df_used_comp_obs['InstMag'] - total_offset)

        # Build fixed effect variables and values, and random effect variable:
        fixed_effect_var_list = []
        fixed_effect_var_list.append('Vignette')  # mandatory fixed-effect (indep) var.
        random_effect_var_name = 'FITSfile'  # cirrus effect is per-image

        # Execute regression:
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)
        self.mm_fit = MixedModelFit(data=self.df_used_comp_obs,
                                    dep_var=self.dep_var_name,
                                    fixed_vars=fixed_effect_var_list,
                                    group_var=random_effect_var_name)
        n_comps_used = len(self.df_used_comp_obs['CompID'].drop_duplicates())
        print(self.mm_fit.statsmodels_object.summary())
        print('comps =', str(n_comps_used), ' used.')
        print('sigma =', '{0:.1f}'.format(1000.0 * self.mm_fit.sigma), 'mMag.')
        if not self.mm_fit.converged:
            msg = ' >>>>> WARNING: Regression (mixed-model) DID NOT CONVERGE.'
            print(msg)
            fit_summary_lines.append(msg)
        self.return_text += (66 * '=') + '\n' +\
                            self.mm_fit.statsmodels_object.summary().as_text() +\
                            '\n'.join(fit_summary_lines) +\
                            'comps = ' + str(n_comps_used) + ' used' +\
                            '\nsigma = ' + \
                            '{0:.1f}'.format(1000.0 * self.mm_fit.sigma) + ' mMag.'
        common.write_text_file(self.this_directory, 'fit_summary.txt', self.return_text)

    def _calc_mp_mags(self) -> pd.DataFrame:
        """ Use model and MP instrument magnitudes to get best estimates of
            MP absolute magnitudes."""
        self.df_used_mp_obs['CatMag'] = 0  # totally bogus local value.

        raw_predictions = self.mm_fit.predict(self.df_used_mp_obs,
                                              include_random_effect=True)
        instrument_mag_observed = self.df_used_mp_obs['InstMag']
        extinction_offset = (self.df_used_mp_obs['ExtinctionValue'] *
                             self.df_used_mp_obs['ObsAirmass'])
        zeropoint_offset = self.df_used_mp_obs['ZeroPointOffset']
        untransformed_mp_best_mags = instrument_mag_observed \
            - extinction_offset - zeropoint_offset - raw_predictions

        # Return value is pandas series: best_mp_mag + transform*color_index.
        df_untransformed_mp_mags = pd.DataFrame(
            data={'Untransformed_MP_Mags': untransformed_mp_best_mags},
            index=list(untransformed_mp_best_mags.index))
        df_untransformed_mp_mags = pd.merge(left=df_untransformed_mp_mags,
                                            right=self.df_used_mp_obs.loc[:,
                                                  ['JD_mid', 'JD_fract', 'FITSfile',
                                                   'InstMag', 'InstMagSigma', 'Filter',
                                                   'TransformValue']],
                                            how='left',
                                            left_index=True, right_index=True,
                                            sort=False)
        return df_untransformed_mp_mags


class ColorModel_2:
    """ Accepts adjusted MP magnitudes from ColorModel_1 and related data,
        effectively solves simultaneous equations to yield
        MP colors and other less important results."""
    def __init__(self, df_untransformed_mp_mags: pd.DataFrame) -> None:
        """ Organize data, perform second regression, store MP colors & other data.
        :param df_untransformed_mp_mags:
        :return: [None] See object attribute variables for results.
        """
        context, defaults_dict, color_dict, color_def_dict, log_file = \
            _color_setup('ColorModel_2')
        self.df_model = df_untransformed_mp_mags
        self.color_def_dict = color_def_dict
        self.results = None
        self.return_text = ''
        self._prep_and_do_regression(include_dt2=color_dict['fit dt2'])
        print(self.return_text + '\n')

    def _prep_and_do_regression(self, include_dt2: bool = False) -> None:
        """ Using ordinary least squares regression to extract color values for MP.
        :param include_dt2: True iff quadratic time parm should be included,
            else False. [boolean]
        :return: [None]
        """
        dep_var_name = 'Untransformed_MP_Mags'
        indep_var_names = []

        # Make time columns:
        self.df_model['DT'] = self.df_model['JD_fract'] - \
                              self.df_model['JD_fract'].mean()
        indep_var_names.append('DT')
        if include_dt2:
            self.df_model['DT2'] = (self.df_model['DT'] ** 2).astype(float)
            indep_var_names.append('DT2')

        # Make Color Index Factor column(s):
        filters_to_include = self.df_model['Filter'].drop_duplicates()
        reference_filter = self.color_def_dict['reference filter']
        self.df_model['CI Factor'] = 0
        color_column_names = []
        for f in filters_to_include:
            if f != reference_filter:
                pb = self.color_def_dict['filters'][f]['target passband']
                ci_pbs = self.color_def_dict['filters'][f]['transform ci']
                if pb == ci_pbs[0]:
                    ci_factor = 1
                elif pb == ci_pbs[1]:
                    ci_factor = -1
                else:
                    raise ini.ColorDefinitionError\
                        ('Passband ' + pb + ' (from filter ' + f +
                         'not valid for its transform ci ' + '-'.join(ci_pbs))
                is_in_filter = list(self.df_model['Filter'] == f)
                self.df_model.loc[is_in_filter, 'CI Factor'] = ci_factor
                color_column_name = 'Color_' + '_'.join(ci_pbs)
                self.df_model[color_column_name] = 0
                self.df_model.loc[is_in_filter, color_column_name] = ci_factor
                indep_var_names.append(color_column_name)

        # Make TransformOffsets, add to proper Color_ column:
        for f in filters_to_include:
            # pb = self.color_def_dict['filters'][f]['target passband']
            ci_pbs = self.color_def_dict['filters'][f]['transform ci']
            is_in_filter = list(self.df_model['Filter'] == f)
            color_column_name = 'Color_' + '_'.join(ci_pbs)
            self.df_model.loc[is_in_filter, color_column_name] += \
                self.df_model.loc[is_in_filter, 'TransformValue']

        # Arrange data and execute Ordinary Least Squares multiple regression:
        df_y = self.df_model[dep_var_name].copy()
        df_x = self.df_model[indep_var_names].copy()
        df_x = sm.add_constant(df_x)
        import warnings
        from statsmodels.tools.sm_exceptions import ValueWarning
        warnings.simplefilter('ignore', ValueWarning)
        model = sm.OLS(df_y, df_x)
        self.results = model.fit()
        self.return_text += '\n\n\n' + (81 * '=') + '\n' + str(self.results.summary())
        self.return_text += '\nResidual = ' + \
                            '{0:0.1f}'.format(1000 * self.results.mse_resid**0.5) + \
                            ' mMag.'


def _make_color_diagnostic_plots(df_model: pd.DataFrame, color_model_1: ColorModel_1,
                                 color_model_2: ColorModel_2) -> None:
    """  Display and write to file several diagnostic plots, to:
     * decide which obs, comps, images might need removal by editing color.ini, and
     * possibly adjust regression parameters, also by editing color.ini (unusual). """
    context, defaults_dict, color_dict, color_def_dict, log_file = \
        _color_setup('_make_color_diagnostic_plots')
    this_directory, mp_string, an_string, _ = context

    # Delete any previous plot files from current directory:
    color_plot_filenames = [f for f in os.listdir('.')
                            if f.startswith(COLOR_PLOT_FILE_PREFIX)
                            and f.endswith('.png')]
    for f in color_plot_filenames:
        os.remove(f)

    # Wrangle needed data into convenient forms (adapted from session.py):
    df_plot_comp_obs = \
        pd.merge(left=df_model.loc[df_model['UseInModel'], :].copy(),
                 right=color_model_1.mm_fit.df_observations,
                 how='left', left_index=True, right_index=True, sort=False)
    df_plot_comp_obs = \
        pd.merge(left=df_plot_comp_obs,
                 right=color_model_1.df_used_comp_obs['InstMag_with_offsets'],
                 how='left', left_index=True, right_index=True, sort=False)
    df_image_effect = color_model_1.mm_fit.df_random_effects
    df_image_effect.rename(columns={"GroupName": "FITSfile", "Group": "ImageEffect"},
                           inplace=True)
    sigma = color_model_1.mm_fit.sigma
    comp_ids = df_plot_comp_obs['CompID'].drop_duplicates()
    n_comps = len(comp_ids)
    jd_floor = floor(min(df_model['JD_mid']))
    xlabel_jd = 'JD(mid)-' + str(jd_floor)
    mp_mag_sr = color_model_2.results.params.const
    color_sr_si = color_model_2.results.params.Color_SR_SI
    n_color_points = len(color_model_2.df_model)

    # ################ SESSION FIGURE 1: Q-Q plot of mean comp effects
    #     (1 pt per comp star used in model).
    window_title = 'Color Q-Q (by comp):  MP ' + mp_string + '   AN ' + an_string
    page_title = 'Color: MP ' + mp_string + '   AN ' + an_string + \
                 '   ::   Q-Q by comp (mean residual)'
    plot_annotation = f'{n_comps} comps used in color model.\n(tags: comp star ID)'
    df_y = df_plot_comp_obs.loc[:, ['CompID', 'Residual']].groupby(['CompID']).mean()
    df_y = df_y.sort_values(by='Residual')
    y_data = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y.index.values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
                          COLOR_PLOT_FILE_PREFIX + '1_QQ_comps.png')

    # ################ SESSION FIGURE 2: Q-Q plot of comp residuals
    #     (one point per comp obs).
    window_title = 'Color Q-Q (by comp observation):  MP ' + mp_string + \
                   '   AN ' + an_string
    page_title = 'Color: MP ' + mp_string + '   AN ' + an_string + \
                 '   ::   Q-Q by comp observation'
    plot_annotation = str(len(df_plot_comp_obs)) + ' observations of ' + \
                      str(n_comps) + ' comps used in model.' + \
                      '\n (tags: observation ID)'
    df_y = df_plot_comp_obs.loc[:, ['ObsID', 'Residual']]
    df_y = df_y.sort_values(by='Residual')
    y_data = df_y['Residual'] * 1000.0  # for millimags
    y_labels = df_y['ObsID'].values
    make_qq_plot_fullpage(window_title, page_title, plot_annotation, y_data, y_labels,
                          COLOR_PLOT_FILE_PREFIX + '2_QQ_obs.png')

    # ################ SESSION FIGURE 3: Catalog and Time plots:
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(11, 8.5))  # (w, h) in "inches"
    fig.tight_layout(rect=(0, 0, 1, 0.925))  # (left, bottom, right, top) for entire fig
    fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85,
                        wspace=0.25, hspace=0.325)
    fig.suptitle('Color: MP ' + mp_string + '   AN ' + an_string +
                 '   ::    catalog and time plots', color='darkblue', fontsize=20)
    fig.canvas.set_window_title('Color: Catalog and Time Plots: ' + 'MP ' +
                                mp_string + '   AN ' + an_string)
    subplot_text = 'rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
    fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center',
             fontsize=12, color='dimgray')

    # Catalog mag uncertainty plot (comps only, one point per comp,
    #     x=cat r mag, y=cat r uncertainty):
    ax = axes[0, 0]
    make_9_subplot(ax, 'Catalog Mag Uncertainty (dr)', 'Catalog Mag (r)',
                   'mMag', '', False,
                   x_data=df_plot_comp_obs['r'], y_data=df_plot_comp_obs['dr'])

    # Catalog color plot (comps only, one pt per comp, x=cat r mag, y=cat color (r-i)):
    ax = axes[0, 1]
    make_9_subplot(ax, 'Catalog Color Index', 'Catalog Mag (r)', 'CI Mag', '',
                   zero_line=False, x_data=df_plot_comp_obs['r'],
                   y_data=(df_plot_comp_obs['r'] - df_plot_comp_obs['i']))
    ax.scatter(x=n_color_points * [mp_mag_sr], y=n_color_points * [color_sr_si],
               s=24, alpha=1, color='orange',
               edgecolors='red', zorder=+100)  # add MP points.

    # Inst Mag plot (comps only, one point per obs, x=cat r mag, y=InstMagSigma):
    ax = axes[0, 2]
    make_9_subplot(ax, 'Instrument Magnitude Uncertainty', 'Catalog Mag (r)',
                   'mMag', '', True, x_data=df_plot_comp_obs['r'],
                   y_data=1000.0 * df_plot_comp_obs['InstMagSigma'])
    ax.scatter(x=n_color_points * [mp_mag_sr],
               y=1000.0 * color_model_1.df_used_mp_obs['InstMagSigma'],
               s=24, alpha=1, color='orange',
               edgecolors='red', zorder=+100)  # add MP points.

    # Cirrus plot (comps only, one point per image, x=JD_fract, y=Image Effect):
    ax = axes[1, 0]
    df_this_plot = pd.merge(df_image_effect, df_plot_comp_obs.loc[:,
                                             ['FITSfile', 'JD_fract']],
                            how='left', on='FITSfile', sort=False).drop_duplicates()
    make_9_subplot(ax, 'Image effect (cirrus plot)', xlabel_jd, 'mMag', '', False,
                   x_data=df_this_plot['JD_fract'],
                   y_data=1000.0 * df_this_plot['ImageEffect'],
                   alpha=1.0, jd_locators=True)
    ax.invert_yaxis()  # per custom of plotting magnitudes brighter=upward

    # SkyADU plot (comps only, one point per obs: x=JD_fract, y=SkyADU):
    ax = axes[1, 1]
    make_9_subplot(ax, 'SkyADU vs time', xlabel_jd, 'ADU', '', False,
                   x_data=df_plot_comp_obs['JD_fract'],
                   y_data=df_plot_comp_obs['SkyADU'],
                   jd_locators=True)

    # FWHM plot (comps only, one point per obs: x=JD_fract, y=FWHM):
    ax = axes[1, 2]
    make_9_subplot(ax, 'FWHM vs time', xlabel_jd, 'FWHM (pixels)', '', False,
                   x_data=df_plot_comp_obs['JD_fract'],
                   y_data=df_plot_comp_obs['FWHM'],
                   jd_locators=True)

    # InstMagSigma plot (comps only, one point per obs; x=JD_fract, y=InstMagSigma):
    ax = axes[2, 0]
    make_9_subplot(ax, 'Inst Mag Sigma vs time', xlabel_jd, 'mMag', '', False,
                   x_data=df_plot_comp_obs['JD_fract'],
                   y_data=1000.0 * df_plot_comp_obs['InstMagSigma'],
                   jd_locators=True)

    # Obs.Airmass plot (comps only, one point per obs; x=JD_fract, y=ObsAirmass):
    ax = axes[2, 1]
    make_9_subplot(ax, 'Obs.Airmass vs time', xlabel_jd, 'Obs.Airmass', '', False,
                   x_data=df_plot_comp_obs['JD_fract'],
                   y_data=df_plot_comp_obs['ObsAirmass'],
                   jd_locators=True)

    # Skip Sloan r' vs time (lightcurve) as meaningless for color determination,
    #     and remove empty plot:
    axes[2, 2].remove()

    plt.show()
    fig.savefig(COLOR_PLOT_FILE_PREFIX + '3_Catalog_and_Time.png')

    # ################ SESSION FIGURE 4: Residual plots:
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(11, 8.5))  # (w, h) in "inches"
    fig.tight_layout(rect=(0, 0, 1, 0.925))  # (left, bottom, right, top) for entire fig
    fig.subplots_adjust(left=0.06, bottom=0.06, right=0.94, top=0.85,
                        wspace=0.25, hspace=0.325)
    fig.suptitle('Color: MP ' + mp_string + '   AN ' + an_string +
                 '    ::    residual plots', color='darkblue', fontsize=20)
    fig.canvas.set_window_title('Color: Residual Plots: ' + 'MP ' +
                                mp_string + '   AN ' + an_string)
    subplot_text = str(len(df_plot_comp_obs)) + ' obs   ' + \
                   str(n_comps) + ' comps    ' + \
                   'sigma=' + '{0:.0f}'.format(1000.0 * sigma) + ' mMag' + \
                   (12 * ' ') + \
                   ' rendered {:%Y-%m-%d  %H:%M UTC}'.format(datetime.now(timezone.utc))
    fig.text(s=subplot_text, x=0.5, y=0.92, horizontalalignment='center',
             fontsize=12, color='dimgray')

    # Comp residual plot (comps only, one point per obs: x=catalog r mag, y=model residual):
    ax = axes[0, 0]
    make_9_subplot(ax, 'Model residual vs r (catalog)', 'Catalog Mag (r)',
                   'mMag', '', True,
                   x_data=df_plot_comp_obs['r'],
                   y_data=1000.0 * df_plot_comp_obs['Residual'])
    # TODO: put derived r' mag in place of 'MP_Mags' (for spec of every relevant plot).
    ax.scatter(x=n_color_points * [mp_mag_sr], y=n_color_points * [0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)
    draw_x_line(ax, color_dict['min catalog r mag'])
    draw_x_line(ax, color_dict['max catalog r mag'])

    # Comp residual plot (comps only, one point per obs: x=raw Instrument Mag,
    #     y=model residual):
    ax = axes[0, 1]
    make_9_subplot(ax, 'Model residual vs raw Instrument Mag',
                   'Raw instrument mag', 'mMag', '', True,
                   x_data=df_plot_comp_obs['InstMag'],
                   y_data=1000.0 * df_plot_comp_obs['Residual'])
    ax.scatter(x=color_model_1.df_used_mp_obs['InstMag'],
               y=len(color_model_1.df_used_mp_obs) * [0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)

    # Comp residual plot (comps only, one point per obs:
    #     x=catalog r-i color, y=model residual):
    ax = axes[0, 2]
    make_9_subplot(ax, 'Model residual vs Color Index (cat)',
                   'Catalog Color (r-i)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['r'] - df_plot_comp_obs['i'],
                   y_data=1000.0 * df_plot_comp_obs['Residual'])
    ax.scatter([color_sr_si], y=[0.0],
               s=24, alpha=1, color='orange', edgecolors='red', zorder=+100)

    # Comp residual plot (comps only, one point per obs:
    #     x=Julian Date fraction, y=model residual):
    ax = axes[1, 0]
    make_9_subplot(ax, 'Model residual vs JD', xlabel_jd, 'mMag', '', True,
                   x_data=df_plot_comp_obs['JD_fract'],
                   y_data=1000.0 * df_plot_comp_obs['Residual'],
                   jd_locators=True)

    # Comp residual plot (comps only, one point per obs:
    #     x=ObsAirmass, y=model residual):
    ax = axes[1, 1]
    make_9_subplot(ax, 'Model residual vs Obs.Airmass', 'ObsAirmass',
                   'mMag', '', True,
                   x_data=df_plot_comp_obs['ObsAirmass'],
                   y_data=1000.0 * df_plot_comp_obs['Residual'])

    # Comp residual plot (comps only, one point per obs:
    #     x=Sky Flux (ADUs), y=model residual):
    ax = axes[1, 2]
    make_9_subplot(ax, 'Model residual vs Sky Flux', 'Sky Flux (ADU)',
                   'mMag', '', True,
                   x_data=df_plot_comp_obs['SkyADU'],
                   y_data=1000.0 * df_plot_comp_obs['Residual'])

    # Comp residual plot (comps only, one point per obs:
    #     x=X in images, y=model residual):
    ax = axes[2, 0]
    make_9_subplot(ax, 'Model residual vs X in image', 'X from center (pixels)',
                   'mMag', '', True,
                   x_data=df_plot_comp_obs['X1024'] * 1024.0,
                   y_data=1000.0 * df_plot_comp_obs['Residual'])
    draw_x_line(ax, 0.0)

    # Comp residual plot (comps only, one point per obs:
    #     x=Y in images, y=model residual):
    ax = axes[2, 1]
    make_9_subplot(ax, 'Model residual vs Y in image',
                   'Y from center (pixels)', 'mMag', '', True,
                   x_data=df_plot_comp_obs['Y1024'] * 1024.0,
                   y_data=1000.0 * df_plot_comp_obs['Residual'])
    draw_x_line(ax, 0.0)

    # Comp residual plot (comps only, one point per obs:
    #     x=vignette (dist from center), y=model residual):
    ax = axes[2, 2]
    make_9_subplot(ax, 'Model residual vs distance from center',
                   'dist from center (pixels)', 'mMag', '', True,
                   x_data=1024*np.sqrt(df_plot_comp_obs['Vignette']),
                   y_data=1000.0 * df_plot_comp_obs['Residual'])

    plt.show()
    fig.savefig(COLOR_PLOT_FILE_PREFIX + '4_Residuals.png')

    # ################ FIGURE(S) 5: Variability plots:
    # Several comps on a subplot, vs JD, normalized by (minus) the mean
    #     of all other comps' responses.
    # Make df_offsets (one row per obs, at first with only raw offsets):
    # TODO: make df_plot_comp_obs_filter_corrected from df_plot_comp_obs,
    #       using corrections for (1) color as determined, and (2) transforms as given.
    df_plot_comp_obs_filter_corrected = df_plot_comp_obs.copy()
    # TODO: Make corrections here.
    iiii = 4

    make_comp_variability_plots(df_plot_comp_obs_filter_corrected, mp_string,
                                an_string, xlabel_jd, sigma, COLOR_PLOT_FILE_PREFIX)


__________SUPPORT_FUNCTIONS_and_CLASSES_________________________ = 0


def _get_color_context() -> Tuple[str, str, str, str]:
    """ Run at beginning of color workflow functions (ex start() or resume())
        to orient the function. Assumes python current working directory =
        the relevant AN subdirectory with session.log in place.
        Adapted from package mp_phot, workflow_session.py._get_session_context().
        Required for .resume().
        TESTED OK 2021-01-08.
    :return: 3-tuple: (this_directory, mp_string, an_string) [3 strings]
    """
    # Open and validate color log file:
    this_directory = os.getcwd()
    defaults_dict = ini.make_defaults_dict()
    color_log_filename = defaults_dict['color log filename']
    color_log_fullpath = os.path.join(this_directory, color_log_filename)
    if not os.path.isfile(color_log_fullpath):
        raise ColorLogFileError(f'No color log file found at {color_log_fullpath}.\n'
                                f'You probably need to run start() or resume().')
    with open(color_log_fullpath, mode='r') as log_file:
        lines = log_file.readlines()
    if len(lines) < 6:
        raise ColorLogFileError(f'Too few lines in file: {color_log_fullpath}.')
    if not lines[0].lower().startswith('color log file'):
        raise ColorLogFileError(f'Header line \'{lines[0]}\' cannot be parsed.')

    # Check consistency of working directory and parse it to string of normal format:
    directory_from_color_log = lines[1].split(':', maxsplit=1)[1].strip().lower()\
        .replace('\\', '/').replace('//', '/')
    directory_from_cwd = this_directory.strip().lower()\
        .replace('\\', '/').replace('//', '/')
    if directory_from_color_log != directory_from_cwd:
        print()
        print(directory_from_color_log, directory_from_cwd)
        raise ColorLogFileError('Header line does not match current working directory.')
    this_directory = directory_from_cwd

    # Extract context data from color log file:
    mp_string = lines[2][3:].strip().upper()
    an_string = lines[3][3:].strip()
    color_definition_filename = lines[4][len('Color definition file:'):].strip()
    return this_directory, mp_string, an_string, color_definition_filename


def _write_color_ini_stub(this_directory: str, filenames_temporal_order: List[str],
                          mean_datetime: datetime, color_definition_filename: str):
    """ Write color subdir's initial control (.ini) file, later to be edited by user.
        Called only by (at the end of) .assess().
        DO NOT overwrite if color.ini already exists.
    :param this_directory:
    :param filenames_temporal_order: FITS filenames (all used filters)
        in ascending time order. [list of strings]
    :param mean_datetime: averate datetime of exposures. [py datetime object]
    :return: [None]
    """
    # Load default data, and ensure that we do not overwrite existing color ini file:
    defaults_dict = ini.make_defaults_dict()
    color_ini_filename = defaults_dict['color control filename']
    fullpath = os.path.join(this_directory, color_ini_filename)
    if os.path.exists(fullpath):
        return

    # Establish some basic data:
    filename_earliest = filenames_temporal_order[0]
    filename_latest = filenames_temporal_order[-1]
    year_an = mean_datetime.year
    site_dict = ini.make_site_dict(defaults_dict)
    coldest_date = site_dict['coldest date'].split('-')
    coldest_datetime = datetime(year_an, int(coldest_date[0]),
                                int(coldest_date[1])).replace(tzinfo=timezone.utc)
    season_phase = (mean_datetime - coldest_datetime).total_seconds() / \
                   (365.25 * 24 * 3600)

    # Get color definition file and parse it:
    color_definition_fullpath = os.path.join(INI_DIRECTORY, 'color',
                                             color_definition_filename)
    color_def_dict = ini.make_color_def_dict(color_definition_fullpath)
    extinction_lines = []
    for f in color_def_dict['filters']:
        extinctions = site_dict['extinctions'][f]
        mean_ext = (extinctions[0] + extinctions[1]) / 2.0
        half_amplitude = (extinctions[0] - extinctions[1]) / 2.0
        extinction_an = mean_ext - half_amplitude * cos(2.0 * pi * season_phase)
        extinction_lines.append((' '.ljust(15) + f.ljust(8) + ' ' +
                                 '{:6.4f}'.format(extinction_an)))
    extinction_lines[0] = 'Extinctions ='.ljust(15) + extinction_lines[0][15:]

    # Get instrument file and parse it:
    inst_dict = ini.make_instrument_dict(defaults_dict)
    transform_lines = []
    for f in color_def_dict['filters']:
        f_def = color_def_dict['filters'][f]
        transform_key = (f, f_def['target passband'], f_def['transform ci'][0],
                         f_def['transform ci'][1])
        transform = inst_dict['transforms'][transform_key][0]
        transform_lines.append((' '.ljust(15) + f.ljust(8) + str(transform)))
    transform_lines[0] = 'Transforms ='.ljust(15) + transform_lines[0][15:]

    header_lines = [
        f'; This is {fullpath}.',
        '']
    ini_lines = [
        '[Ini Template]',
        'Filename = color.template',
        '']
    color_definition_lines = [
        '[Color Definition]',
        f'Color Definition Filename = {color_definition_filename}',
        '']
    mp_location_lines = [
        '[MP Location]',
        '; Exactly 2 MP XY, one per line (typically earliest and latest FITS):',
        f'MP XY = {filename_earliest} 000.0  000.0',
        f'        {filename_latest} 000.0  000.0',
        '']
    selection_criteria_lines = [
        '[Selection Criteria]',
        'Omit Comps = ',
        'Omit Obs = ',
        '; One image only per line, with or without .fts:',
        'Omit Images = ',
        'Min Catalog r mag = 11',
        'Max Catalog r mag = 16',
        'Max Catalog dr mmag = 16',
        'Max Catalog di mmag = 16',
        'Min Catalog ri color = -0.20',
        'Max Catalog ri color = +0.64',
        '']
    regression_lines = [
        '[Regression]',
        f'; Extinctions adapted from Site file \'{defaults_dict["site ini"]}\', '
        f'adjusted for observing night.',
        '; Overwrite if needed (rare).'] + \
        extinction_lines + \
        [f'; Transforms copied from Instrument file'
         f' \'{defaults_dict["instrument ini"]}\'.'] +\
        transform_lines + \
        ['; Set Fit DT2 = Yes to fit quadratic time-dependence.',
         'Fit DT2 = No']
    raw_lines = header_lines + ini_lines + color_definition_lines +\
        mp_location_lines + selection_criteria_lines + regression_lines
    ready_lines = [line + '\n' for line in raw_lines]
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(ready_lines)


def _color_setup(calling_function_name='[FUNCTION NAME NOT GIVEN]'):
    """ Typically called at the top of color workflow functions,
        to collect commonly required data.
    :return: tuple of data elements: context [tuple], defaults_dict [py dict],
        log_file [file object].
    """
    context = _get_color_context()
    if context is None:
        return
    this_directory, mp_string, an_string, color_definition_filename = context
    defaults_dict = ini.make_defaults_dict()
    color_dict = ini.make_color_dict(defaults_dict, this_directory)
    color_definition_fullpath = os.path.join(INI_DIRECTORY, 'color',
                                             color_definition_filename)
    color_def_dict = ini.make_color_def_dict(color_definition_fullpath)
    log_filename = defaults_dict['color log filename']
    log_file = open(log_filename, mode='a')  # set up append to log file.
    log_file.write('\n===== ' + calling_function_name + '()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) +
                   '\n')
    return context, defaults_dict, color_dict, color_def_dict, log_file


__________TESTING_FUNCTIONS_and_CLASSES__________________________ = 0


def test_do_color_with_comps():
    # Make list of comp_ids, write them to file, one per line:
    context, defaults_dict, color_dict, color_def_dict, log_file = \
        _color_setup('do_color_stepwise')
    this_directory, mp_string, an_string = context
    filters_to_include = list(color_def_dict['filters'].keys())
    df_comp_master, _ = common.make_df_masters(this_directory, defaults_dict,
                                               filters_to_include=filters_to_include,
                                               require_mp_obs_each_image=True,
                                               data_error_exception_type=ColorDataError)
    comp_ids = df_comp_master.loc[:, 'CompID'].drop_duplicates()
    if os.path.exists(COLOR_SUMMARY_FILENAME):
        os.remove(COLOR_SUMMARY_FILENAME)
    for comp_id in comp_ids:
        do_color(requested_comp_id=comp_id)


def get_df_from_sloan_moc(top_mp_number=5000):
    """ Read and parse SDSS MOC file, produce relatively small pandas DataFrame.
    :param top_mp_number: highest asteroid number to include. [int]
    :return: dataframe of results including mags and color.
    """
    fullpath = 'D:/Astro/Catalogs/SDSS MOC/gbo.sdss-moc.phot/data/sdssmocadr4.tab'
    if os.path.exists(fullpath):
        with open(fullpath, 'r') as f:
            lines = f.readlines()
    else:
        print(' >>>>> Cannot find SDSS MOC at:', fullpath)
        return
    dict_list = []
    for line in lines:
        if line[249:251] != ' 0':
            if line[242:243] == '1':
                mp_number = int(line[244:251])
                if mp_number <= top_mp_number:
                    # Put this catalog line in the dataframe:
                    g = float(line[174:179])  # magnitude
                    r = float(line[185:190])  # "
                    i = float(line[196:201])  # "
                    sgr = g - r  # color
                    sri = r - i  # "
                    line_dict = {'MP': mp_number, 'SG': g, 'SR': r, 'SI': i, 'SGR': sgr, 'SRI': sri}
                    dict_list.append(line_dict)
    df = pd.DataFrame(data=dict_list)
    if len(df) >= 1:
        df.index = list(df['MP'])
    return df


def _replace_mp_with_requested_comp(df_comp_master, comp_id,
                                    n_required_comp_obs, color_dict):
    """ Make all the new data needed to substitute a comp star (with known color)
        for a minor planet. A quick means to test do_color() with targets of
        *known* color.
    :param df_comp_master:
    :param comp_id:
    :param n_required_comp_obs:
    :param color_dict:
    :return: 2-tuple:
        (0): new df_mp_master w/ data of one comp. [pandas Dataframe]
        (1): new color_dict w/pixel positions of one comp. [python OrderedDict]
    """
    # Make new dataframe with only comp observations:
    is_comp_id = (df_comp_master.loc[:, 'CompID'] == comp_id)
    new_df = df_comp_master.loc[is_comp_id, :].copy()
    if len(new_df) != n_required_comp_obs:
        return None, None

    # Also, update color_dict with comp pixel positions,
    #     in place of normal MP pixel positions.
    new_xy_list = []
    for i, item in enumerate(color_dict['mp xy']):
        filename = item[0]
        x = new_df.loc[new_df['FITSfile'] == filename, 'Xcentroid']
        y = new_df.loc[new_df['FITSfile'] == filename, 'Ycentroid']
        new_xy_list.append(tuple([filename, x, y]))
    new_color_dict = deepcopy(color_dict)
    new_color_dict['mp xy'] = new_xy_list

    return new_df, new_color_dict


def _replace_mp_with_a_comp(color_dict: dict, df_comp_master: pd.DataFrame,
                            df_mp_master: pd.DataFrame, requested_comp_id: str) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    n_images_required = 7  # based on sequence defined in Color Def .ini file.
    df_mp_master, color_dict = \
        _replace_mp_with_requested_comp(df_comp_master, requested_comp_id,
                                        n_images_required, color_dict)
    with open(COLOR_SUMMARY_FILENAME, mode='a') as sf:
        sf.write('REQUESTED COMP ID = ' + requested_comp_id + '\n')
        ra = (df_mp_master.loc[:, 'RA_deg'])[0]
        dec = (df_mp_master.loc[:, 'Dec_deg'])[0]
        sf.write('RA, Dec = ' + '{0:.4f}'.format(ra) +
                 ' {0:.5f}'.format(dec) + '\n')
        instmag_0 = (df_mp_master.loc[:, 'InstMag'])[0]
        sf.write('InstMag_0 = ' + '{0:.5f}'.format(instmag_0) + '\n')
        sf.flush()
    return color_dict, df_mp_master


# def make_test_df_untransformed():
# THIS WORKED VERY WELL, JULY 2, 2021.
#     catmag_SR = 14.1
#     color_SG_SR = +0.225
#     color_SR_SI = +0.207
#     catmag_SG = catmag_SR + color_SG_SR
#     catmag_SI = catmag_SR - color_SR_SI
#
#     jd_fracts = [0.3 + 0.03 * i for i in range(7)]
#     jd_mids = [1234567 + jd for jd in jd_fracts]
#     FITSfile = ['File_' + str(i + 1) for i in range(7)]
#
#     t_V, t_R, t_I = -0.51, -0.194, -0.216
#     untr_SG = catmag_SG - t_V * color_SG_SR
#     untr_SR = catmag_SR - t_R * color_SR_SI
#     untr_SI = catmag_SI - t_I * color_SR_SI
#
#     row_V = {'Untransformed_MP_Mags': untr_SG, 'Filter': 'V', 'TransformValue': t_V}
#     row_R = {'Untransformed_MP_Mags': untr_SR, 'Filter': 'R', 'TransformValue': t_R}
#     row_I = {'Untransformed_MP_Mags': untr_SI, 'Filter': 'I', 'TransformValue': t_I}
#     dict_list = [row_V, row_R, row_I, row_V, row_I, row_R, row_V]
#     df = pd.DataFrame(data=dict_list)
#     df.loc[:, 'JD_mid'] = jd_mids
#     df.loc[:, 'JD_fract'] = jd_fracts
#     df.loc[:, 'FITSfile'] = FITSfile
#     error = [i * 0.0001 for i in [1, -1, 1, -1, 1, -1, 0]]
#     df.loc[:, 'Untransformed_MP_Mags'] = \
#         [u + e for (u, e) in zip(df.loc[:, 'Untransformed_MP_Mags'], error)]
#     return df

__________MAJOR_TEST_of_do_color__________________________ = 0


def make_artificial_data_to_test_do_color(rms_mags: float=0.000):
    """ Just what the name says. """
    test_data_directory = 'C:/Astro/MP Color/MP_10/AN20220513'

    # Make the dataframes:
    print('Making 4 dataframes...')
    df_images_all = make_df_images_all()
    df_comps_all = make_df_comps_all()
    df_comp_obs_all = make_df_comp_obs_all(df_images_all, df_comps_all, rms_mags)
    df_mp_obs_all = make_df_mp_obs_all(df_images_all, rms_mags)

    # Write the dataframes as .csv files:
    print(f'Writing 4 .csv files to {test_data_directory}.')
    df_images_all.to_csv(os.path.join(test_data_directory, 'df_images_all.csv'),
                         sep=';', quotechar='"', quoting=2, index=True)
    df_comps_all.to_csv(os.path.join(test_data_directory, 'df_comps_all.csv'),
                        sep=';', quotechar='"', quoting=2, index=True)
    df_comp_obs_all.to_csv(os.path.join(test_data_directory, 'df_comp_obs_all.csv'),
                           sep=';', quotechar='"', quoting=2, index=True)
    df_mp_obs_all.to_csv(os.path.join(test_data_directory, 'df_mp_obs_all.csv'),
                         sep=';', quotechar='"', quoting=2, index=True)
    print('Done.')


def _set_key_parameters() -> Tuple:
    import random
    # random.seed(234)
    zero_points = {'SG': -22, 'SR': -21, 'SI': -21.4}
    extinctions = {'SG': 0.185, 'SR': 0.144, 'SI': 0.107}
    transforms = {'SG': -0.07, 'SR': -0.12, 'SI': 0.0633}
    catmag_passbands = {'SG': 'g', 'SR': 'r', 'SI': 'i'}
    color_indices = {'SG': 'gr_color', 'SR': 'ri_color', 'SI': 'ri_color'}
    # cirrus_effect_raw = [random.uniform(-0.02, 0.02)
    # for _ in range(12)]  # TODO: revert to this.
    cirrus_effect_raw = [0 for _ in range(12)]
    cer_mean = sum(cirrus_effect_raw) / len(cirrus_effect_raw)
    cirrus_effect = [cer - cer_mean for cer in cirrus_effect_raw]
    vignette_coeff = 0.008  # TODO: revert to this.
    # vignette_coeff = 0
    return zero_points, extinctions, transforms, catmag_passbands, \
        color_indices, cirrus_effect, vignette_coeff


def make_df_images_all() -> pd.DataFrame:
    """ Make IMAGES dataframe for 12 FITS files (images).
    :return: df_images_all: suitable for rigorous testing of do_color()."""
    #
    fits_filters = ['SR', 'SG', 'SI', 'SG', 'SI', 'SR',
                    'SR', 'SI', 'SG', 'SI', 'SG', 'SR']
    fits_files = [f'MP_10-{i:04d}-{f}.fts'
                  for (i, f) in zip(list(range(1, 12 + 1)), fits_filters)]
    first_utc_mid = datetime(2022, 5, 14, 2, 45, 5).replace(tzinfo=timezone.utc)
    fits_utc_mids = [first_utc_mid + i * timedelta(seconds=864) for i in range(12)]
    exposures = {'SR': 180, 'SG': 400, 'SI': 360}
    fits_exposures = [exposures[f] for f in fits_filters]
    fits_utc_starts = [start - timedelta(seconds=exp / 2)
                       for (start, exp) in zip(fits_utc_mids, fits_exposures)]
    fits_jd_starts = [astropak.util.jd_from_datetime_utc(dt) for dt in fits_utc_starts]
    fits_jd_mids = [astropak.util.jd_from_datetime_utc(dt) for dt in fits_utc_mids]
    jd_floor = floor(min(fits_jd_mids))
    fits_airmasses = [1.4 + 0.03 * i for i in range(12)]
    fits_jd_fracts = [mid - jd_floor for mid in fits_jd_mids]
    d = {'FITSfile': fits_files,
         'JD_mid': fits_jd_mids,
         'Filter': fits_filters,
         'Exposure': fits_exposures,
         'Airmass': fits_airmasses,
         'JD_start': fits_jd_starts,
         'UTC_start': fits_utc_starts,
         'UTC_mid': fits_utc_mids,
         'JD_fract': fits_jd_fracts}
    df_images_all = pd.DataFrame(data=d, index=fits_files)
    return df_images_all


def make_df_comps_all() -> pd.DataFrame:
    """ Make COMPS dataframe for 6 comparison stars.
    :return: df_comps_all: suitable for rigorous testing of do_color()."""
    import random
    comp_ids = [i + 1 for i in range(6)]
    # random.seed(234)
    ra_deg = [151.0 + random.uniform(-0.1, 0.1) for _ in range(6)]
    dec_deg = [27.0 + random.uniform(-0.1, 0.1) for _ in range(6)]
    g_mag = [12.0 + i * 0.3 for i in range(6)]
    dg = [8 + random.randrange(-4, 4) for _ in range(6)]
    r_mag = [gg - 0.4 + 0.03 * i for (gg, i) in zip(g_mag, list(range(6)))]
    dr = [8 + random.randrange(-4, 4) for _ in range(6)]
    i_mag = [rr - 0.2 + 0.02 * i for (rr, i) in zip(r_mag, list(range(6)))]
    di = [8 + random.randrange(-3, 3) for _ in range(6)]
    z_mag = [ii - 0.1 + 0.022 * i for (ii, i) in zip(i_mag, list(range(6)))]
    dz = [10 + random.randrange(-5, 5) for _ in range(6)]
    bmv = [0.75 + random.uniform(-0.25, 0.25) for _ in range(6)]
    apass_r = [13.0 + random.uniform(-1, 1) for _ in range(6)]
    t_eff = [5500 + random.randrange(-500, 500) for _ in range(6)]
    catalog_id = [f'123+26_{i + 100:06d}' for i in range(6)]
    gr_color = [gg - rr for (gg, rr) in zip(g_mag, r_mag)]
    ri_color = [rr - ii for (rr, ii) in zip(r_mag, i_mag)]
    d = {'CompID': comp_ids,
         'RA_deg': ra_deg,
         'Dec_deg': dec_deg,
         'g': g_mag,
         'dg': dg,
         'r': r_mag,
         'dr': dr,
         'i': i_mag,
         'di': di,
         'z': z_mag,
         'dz': dz,
         'BminusV': bmv,
         'APASS_R': apass_r,
         'T_eff': t_eff,
         'CatalogID': catalog_id,
         'gr_color': gr_color,
         'ri_color': ri_color}
    df_comps_all = pd.DataFrame(data=d, index=comp_ids)
    return df_comps_all


def make_df_comp_obs_all(df_images_all: pd.DataFrame, df_comps_all: pd.DataFrame,
                         rms_mags: float=0) -> pd.DataFrame:
    """ Make COMPS OBSERVATION dataframe for 12 images, 6 comparison stars.
    :param df_images_all: previously constructed dataframe.
    :param df_comps_all: previously constructed dataframe.
    :param rms_mags: std dev of Gaussian error RMS added to each InstMag.
    :return: df_comp_obs_all: suitable for rigorous testing of do_color()."""
    import random
    # random.seed(234)
    n_comp_obs = 6 * 12
    zero_points, extinctions, transforms, catmag_passbands, \
        color_indices, cirrus_effect, vignette_coeff \
        = _set_key_parameters()
    fits_files = [ff for ff in df_images_all['FITSfile'] for _ in range(6)]
    comp_id = [ci for _ in df_images_all['FITSfile'] for ci in df_comps_all['CompID']]
    obs_id = [i + 1 for i in range(n_comp_obs)]
    type = ['Comp'] * n_comp_obs
    disc_radius = [12.6] * n_comp_obs
    sky_radius_inner = [19.6] * n_comp_obs
    sky_radius_outer = [19.6] * n_comp_obs
    fwhm = [7 + random.uniform(-1, 1) for _ in range(n_comp_obs)]
    elongation = [1.1 + random.uniform(-0.05, 0.05) for _ in range(n_comp_obs)]
    sky_adu = [785 + random.randrange(-15, 15) for _ in range(n_comp_obs)]
    xcentroid = [1536 + random.uniform(-1000, 1000) for _ in range(n_comp_obs)]
    ycentroid = [1024 + random.uniform(-700, 700) for _ in range(n_comp_obs)]
    x1024 = [(xc - 1536) / 1024 for xc in xcentroid]
    y1024 = [(yc - 1024) / 1024 for yc in ycentroid]
    vignette = [x**2 + y**2 for (x, y) in zip(x1024, y1024)]
    ra_deg = [df_comps_all.loc[cid, 'RA_deg'] for cid in comp_id]
    dec_deg = [df_comps_all.loc[cid, 'Dec_deg'] for cid in comp_id]
    airmass = [a for a in df_images_all['Airmass'] for _ in range(6)]

    # Build InstMag for each observation:
    filters = [df_images_all.loc[ff, 'Filter'] for ff in fits_files]
    zero_point_values = [zero_points[f] for f in filters]
    ci_names = [color_indices[f] for f in filters]
    ci_values = [df_comps_all.loc[id, ci]
                 for (id, ci) in zip(comp_id, ci_names)]
    transform_values = [transforms[f] for f in filters]
    transform_terms = [t * ci for (t, ci) in zip(transform_values, ci_values)]
    extinction_values = [extinctions[f] for f in filters]
    extinction_terms = [ext * am for (ext, am) in zip(extinction_values, airmass)]
    vignette_terms = [vignette_coeff * v for v in vignette]
    cirrus_effect_values = [ce for ce in cirrus_effect for _ in range(6)]
    catmag_passband_names = [catmag_passbands[f] for f in filters]
    catmag_values = [df_comps_all.loc[cid, cpn]
                     for (cid, cpn) in zip(comp_id, catmag_passband_names)]
    random_err_mags = [random.gauss(mu=0.0, sigma=rms_mags) for _ in obs_id]
    inst_mag_values = [zpv + catmag + tt + et + vt + cev + rand
                       for (zpv, catmag, tt, et, vt, cev, rand)
                       in zip(zero_point_values, catmag_values, transform_terms,
                              extinction_terms, vignette_terms, cirrus_effect_values,
                              random_err_mags)]
    inst_mag_sigma = [max(rms_mags, 0.005)] * n_comp_obs
    d = {'FITSfile': fits_files,
         'CompID': comp_id,
         'ObsID': obs_id,
         'Type': type,
         'InstMag': inst_mag_values,
         'InstMagSigma': inst_mag_sigma,
         'DiscRadius': disc_radius,
         'SkyRadiusInner': sky_radius_inner,
         'SkyRadiusOuter': sky_radius_outer,
         'FWHM': fwhm,
         'Elongation': elongation,
         'SkyADU': sky_adu,
         'Vignette': vignette,
         'X1024': x1024,
         'Y1024': y1024,
         'Xcentroid': xcentroid,
         'Ycentroid': ycentroid,
         'RA_deg': ra_deg,
         'Dec_deg': dec_deg,
         'ObsAirmass': airmass}
    df_comp_obs_all = pd.DataFrame(data=d, index=obs_id)
    return df_comp_obs_all


def make_df_mp_obs_all(df_images_all: pd.DataFrame, rms_mags: float=0) -> pd.DataFrame:
    """ Make MP OBSERVATION dataframe for 12 images, one Minor Planet target.
    :param df_images_all: previously constructed dataframe.
    :param rms_mags: std dev of Gaussian error RMS added to each InstMag.
    :return: df_mp_obs_all: suitable for rigorous testing of do_color()."""
    import random
    random.seed(234)
    n_mp_obs = 12
    zero_points, extinctions, transforms, catmag_passbands, \
        color_indices, cirrus_effect, vignette_coeff \
        = _set_key_parameters()
    fits_file = list(df_images_all['FITSfile'])
    filters = [df_images_all.loc[ff, 'Filter'] for ff in fits_file]

    # Compute MP light curve (catalog magnitudes in passbands) for session:
    mp_mag_mean_t = 14.5  # in SR
    dt = 0.2  # d(mag)/d(day), in SR
    dt2 = 5
    mean_t = sum(df_images_all['JD_fract']) / len(df_images_all['JD_fract'])
    catmag_sr = [mp_mag_mean_t + dt * (t - mean_t) + dt2 * (t - mean_t)**2
                 for t in df_images_all['JD_fract']]
    transform_colors = {'SG': 0.234, 'SR': 0.179, 'SI': 0.179}  # for T * CI terms
    fit_color_values = {'SG': 0.234, 'SR': 0, 'SI': -0.179}     # for pb catmags vs SR
    catmag_passband_names = [catmag_passbands[f] for f in filters]
    catmag_values = [mag_sr + fit_color_values[f]
                     for (f, mag_sr) in zip(filters, catmag_sr)]

    # Make data for most dataframe columns:
    mp_id = ['MP_10'] * n_mp_obs
    obs_id = [str(372 + i) for i in range(n_mp_obs)]
    type = ['MP'] * n_mp_obs
    disc_radius = [12.6] * n_mp_obs
    sky_radius_inner = [19.6] * n_mp_obs
    sky_radius_outer = [19.6] * n_mp_obs
    fwhm = [7 + random.uniform(-1, 1) for _ in range(n_mp_obs)]
    elongation = [1.1 + random.uniform(-0.05, 0.05) for _ in range(n_mp_obs)]
    sky_adu = [785 + random.randrange(-15, 15) for _ in range(n_mp_obs)]
    sky_sigma = [20 + random.randrange(-10, 10) for _ in range(n_mp_obs)]
    xcentroid = [1536 + i * 10 for i in range(n_mp_obs)]
    ycentroid = [1024 + i * 10 for i in range(n_mp_obs)]
    xstart = [xc - 1 for xc in xcentroid]
    ystart = [yc - 1 for yc in ycentroid]
    xend = [xc + 1 for xc in xcentroid]
    yend = [yc + 1 for yc in ycentroid]
    x1024 = [(xc - 1536) / 1024 for xc in xcentroid]
    y1024 = [(yc - 1024) / 1024 for yc in ycentroid]
    vignette = [x**2 + y**2 for (x, y) in zip(x1024, y1024)]
    ra_deg = [151.27 + 0.001 * i for i in range(n_mp_obs)]
    dec_deg = [26.93 + 0.001 * i for i in range(n_mp_obs)]
    airmass = list(df_images_all['Airmass'])

    # Build InstMag for each observation:
    zero_point_values = [zero_points[f] for f in filters]
    transform_color_values = [transform_colors[f] for f in filters]
    transform_values = [transforms[f] for f in filters]
    transform_terms = [t * ci
                       for (t, ci) in zip(transform_values, transform_color_values)]
    extinction_values = [extinctions[f] for f in filters]
    extinction_terms = [ext * am for (ext, am) in zip(extinction_values, airmass)]
    vignette_terms = [vignette_coeff * v for v in vignette]
    cirrus_effect_values = cirrus_effect
    random_err_mags = [random.gauss(mu=0.0, sigma=rms_mags) for _ in obs_id]
    inst_mag_values = [zpv + catmag + tt + et + vt + cev + rand
                       for (zpv, catmag, tt, et, vt, cev, rand)
                       in zip(zero_point_values, catmag_values, transform_terms,
                              extinction_terms, vignette_terms, cirrus_effect_values,
                              random_err_mags)]
    inst_mag_sigma = [max(rms_mags, 0.005)] * n_mp_obs
    d = {'FITSfile': df_images_all['FITSfile'],
         'MP_ID': mp_id,
         'ObsID': obs_id,
         'Type': type,
         'InstMag': inst_mag_values,
         'InstMagSigma': inst_mag_sigma,
         'SkyADU': sky_adu,
         'SkySigma': sky_sigma,
         'DiscRadius': disc_radius,
         'SkyRadiusInner': sky_radius_inner,
         'SkyRadiusOuter': sky_radius_outer,
         'FWHM': fwhm,
         'Elongation': elongation,
         'Vignette': vignette,
         'X1024': x1024,
         'Y1024': y1024,
         'Xcentroid': xcentroid,
         'Ycentroid': ycentroid,
         'Xstart': xstart,
         'Ystart': ystart,
         'Xend': xend,
         'Yend': yend,
         'RA_deg_mid': ra_deg,
         'Dec_deg_mid': dec_deg,
         'ObsAirmass': airmass}
    df_mp_obs_all = pd.DataFrame(data=d, index=list(df_images_all['FITSfile']))
    return df_mp_obs_all