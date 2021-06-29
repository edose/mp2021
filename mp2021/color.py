__author__ = "Eric Dose, Albuquerque"

# Python core:
import os
from datetime import datetime, timezone
from math import pi, cos

# External packages:
import pandas as pd

# Author's packages:
import astropak.util
from astropak.stats import MixedModelFit
import mp2021.util as util
import mp2021.ini as ini
import mp2021.common as common
# from mp2021.common import do_fits_assessments, make_df_images, make_df_comps, make_comp_apertures, \
#     make_df_comp_obs, make_mp_apertures, make_fits_objects, get_refcat2_comp_stars, \
#     initial_screen_comps, make_df_mp_obs, validate_mp_xy, add_obsairmass_df_comp_obs, \
#     add_obsairmass_df_mp_obs, add_gr_color_df_comps, add_ri_color_df_comps, write_df_images_csv, \
#     write_df_comps_csv, write_df_comp_obs_csv, write_df_mp_obs_csv, make_df_masters

THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')


class ColorIniFileError(Exception):
    """ Raised on any fatal problem with color ini file."""


class ColorLogFileError(Exception):
    """ Raised on any fatal problem with color log file."""


class ColorDataError(Exception):
    """ Raised on any fatal problem with data, esp. with contents of FITS files or missing data."""


class ColorSpecificationError(Exception):
    """ Raised on any fatal problem in specifying the color processing, esp. in _make_df_all(). """


def start(color_top_directory=None, mp_id=None, an_date=None,
          color_def_filename='ColorDef_Sloan2colors_fromVRI.ini'):
    """ Launch one MP color workflow (one color subdirectory).
        Adapted from package mp2021, session.start().
        Example usage: color.start('C:/Astro/MP Color/', 1111, 20210617)
    :param color_top_directory: path of lowest directory common to all MP color FITS, e.g.,
               'C:/Astro/MP Color'. None will use .ini file default (normal case). [string]
    :param mp_id: either a MP number, e.g., 1602 for Indiana [integer or string], or for an id string
               for unnumbered MPs only, e.g., ''. [string only]
    :param an_date: Astronight date representation, e.g., '20191106'. [integer or string]
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
    mp_directory = os.path.join(color_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    os.chdir(mp_directory)
    print('Working directory set to:', mp_directory)

    # Initiate (or overwrite) log file:
    log_filename = defaults_dict['color log filename']
    with open(log_filename, mode='w') as log_file:
        log_file.write('Color Log File.' + '\n')
        log_file.write(mp_directory + '\n')
        log_file.write('MP: ' + mp_string + '\n')
        log_file.write('AN: ' + an_string + '\n')
        log_file.write('Definition: ' + color_def_filename + '\n')
        log_file.write('This log started: ' +
                       '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n\n')
    print('Log file started.')
    print('Next: assess()')


def resume(color_top_directory=None, mp_id=None, an_date=None,
           color_def_filename='ColorDef_Sloan2colors_fromVRI.ini'):
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
    this_directory = os.path.join(color_top_directory, 'MP_' + mp_string, 'AN' + an_string)
    os.chdir(this_directory)
    print('Working directory set to:', this_directory)
    log_this_directory, log_mp_string, log_an_string = _get_color_context()
    if log_mp_string.upper() != mp_string:
        raise ColorLogFileError(' '.join(['MP string does not match that of color log.',
                                          log_mp_string, mp_string]))
    if log_an_string != an_string:
        raise ColorLogFileError(' '.join(['AN string does not match that of session log.',
                                          log_an_string, an_string]))
    print('Resuming in', this_directory)


def assess(return_results=False):
    """  First, verify that all required files are in this color directory or otherwise accessible.
     Then, perform checks on FITS files in this directory before performing color photometry proper.
     Modeled after and extended from assess() found in variable-star photometry package 'photrix'.
    :return: [None], or dict of summary info and warnings. [py dict]
    """
    try:
        context = _get_color_context()
    except ColorLogFileError as e:
        print(' >>>>> ERROR: ' + str(e))
        return
    this_directory, mp_string, an_string = context
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
            log_file.write('assess(): ALL ' + str(len(df)) + ' FITS FILES APPEAR OK.' + '\n')
        else:
            print('\n >>>>> ' + str(return_dict['warning count']) + ' warnings (see listing above).')
            print('        Correct these and rerun assess() until no warnings remain.')
            log_file.write('assess(): ' + str(return_dict['warning count']) + ' warnings.' + '\n')
    df_temporal = df.loc[:, ['Filename', 'JD_mid']].sort_values(by=['JD_mid'])
    filenames_temporal_order = df_temporal['Filename']
    mean_datetime = astropak.util.datetime_utc_from_jd(df['JD_mid'].mean())
    _write_color_ini_stub(this_directory, filenames_temporal_order, mean_datetime)  # if not already exists.
    if return_results:
        return return_dict


def make_dfs(print_ap_details=False):
    """ Perform aperture photometry for one subdirectory of color photometry. Makes 4 dataframes.
        :param print_ap_details: True if user wants aperture details for each MP. [boolean]
        """
    context, defaults_dict, color_dict, color_def_dict, log_file = _color_setup('make_dfs')
    this_directory, mp_string, an_string = context
    instrument_dict = ini.make_instrument_dict(defaults_dict)
    instrument = util.Instrument(instrument_dict)
    disc_radius, gap, background_width = instrument.nominal_ap_profile
    site_dict = ini.make_site_dict(defaults_dict)

    fits_filenames = util.get_mp_filenames(this_directory)
    if not fits_filenames:
        raise ColorDataError('No FITS files found in color directory ' + this_directory)

    # Quick validation of MP XY filenames & values:
    mp_xy_files_found, mp_xy_values_ok = common.validate_mp_xy(fits_filenames, color_dict)
    if not mp_xy_files_found:
        raise ColorIniFileError('At least 1 MP XY file not found in color directory ' + this_directory)
    if not mp_xy_values_ok:
        raise ColorIniFileError('MP XY invalid -- did you enter values and save color.ini?')

    fits_objects, fits_object_dict = common.make_fits_objects(this_directory, fits_filenames)
    df_images = common.make_df_images(fits_objects)

    # Get and screen catalog entries for comp stars:
    refcat2 = common.get_refcat2_comp_stars(fits_objects)
    info_lines = common.initial_screen_comps(refcat2)  # in-place screening.
    print('\n'.join(info_lines), '\n')
    log_file.write('\n'.join(info_lines) + '\n')

    # Make comp-star apertures, comps dataframe, and comp obs dataframe:
    df_comps = common.make_df_comps(refcat2)
    comp_apertures_dict = common.make_comp_apertures(fits_objects, df_comps, disc_radius, gap,
                                                     background_width)
    df_comp_obs = common.make_df_comp_obs(comp_apertures_dict, df_comps, instrument, df_images)

    # Make MP apertures and MP obs dataframe:
    starting_mp_obs_id = max(df_comp_obs['ObsID'].astype(int)) + 1
    mp_apertures_dict, mp_mid_radec_dict = common.make_mp_apertures(fits_object_dict, mp_string,
                                                                    color_dict, disc_radius, gap,
                                                                    background_width, log_file,
                                                                    starting_obs_id=starting_mp_obs_id,
                                                                    print_ap_details=print_ap_details)
    df_mp_obs = common.make_df_mp_obs(mp_apertures_dict, mp_mid_radec_dict, instrument, df_images)

    # WANTED??: _remove_images_without_mp_obs(fits_object_dict, df_images, df_comp_obs, df_mp_obs)
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


def do_color():
    """ Primary color photometry for one color subdirectory.
    Takes the 4 CSV files from make_dfs().
    Gets MP colors in two steps:
        1. color_model_1: get (untransformed) MP mags for all passbands.
        2. color_model_2: get MP colors using simple regression on MP mags.
    Generates diagnostic plots for iterative regression refinement.
    Typically iterated, pruning comp-star ranges and outliers, until converged and then simply stop.
    :returns None. Writes all info to files.
    USAGE: do_color()   [no return value]
    """
    context, defaults_dict, color_dict, color_def_dict, log_file = _color_setup('do_color_stepwise')
    this_directory, mp_string, an_string = context
    filters_to_include = list(color_def_dict['filters'].keys())
    # instrument_dict = ini.make_instrument_dict(defaults_dict)
    # instrument = Instrument(instrument_dict)
    # disc_radius, gap, background_width = instrument.nominal_ap_profile
    # site_dict = ini.make_site_dict(defaults_dict)

    df_comp_master, df_mp_master = common.make_df_masters(this_directory, defaults_dict,
                                                          filters_to_include=filters_to_include,
                                                          require_mp_obs_each_image=True,
                                                          data_error_exception_type=ColorDataError)
    df_model_raw = common.make_df_model_raw(df_comp_master)  # comps-only data from all used filters.
    df_model = common.mark_user_selections(df_model_raw, color_dict)
    color_model_1 = ColorModel_1(df_model, color_dict, color_def_dict, df_mp_master, this_directory)
    color_model_2 = ColorModel_2(color_model_1)


__________SUPPORT_for_do_color____________________________________________ = 0


# TODO: (1) alter ColorModel_1 to use approximate zero-point offsets, allow random vars to take up residual.
# TODO: (2) decide whether ColorModel_2 should fit to (untransf) MP mags, or to MP InstMags.
class ColorModel_1:
    """ Generates and holds mixed-model regression model suitable to MP color index determination.
        Makes estimates ("predictions") for partial MP magnitudes (partial, because still need to be
           adjusted for MP color index, which we don't know until we've run ColorModel_2.
    """
    def __init__(self, df_model, color_dict, color_def_dict, df_mp_master, this_directory):
        self.df_model = df_model
        self.color_dict = color_dict
        self.color_def_dict = color_def_dict
        self.df_used_comp_obs = self.df_model.copy().loc[self.df_model['UseInModel'], :]
        images_in_used_comps = self.df_used_comp_obs['FITSfile'].drop_duplicates()
        mp_rows_to_use = df_mp_master['FITSfile'].isin(images_in_used_comps)
        self.df_used_mp_obs = df_mp_master.loc[mp_rows_to_use, :]
        self.this_directory = this_directory

        self.dep_var_name = 'InstMag_with_offsets'
        self.mm_fit = None      # placeholder for the fit result [a MixedModelFit object].
        self.vignette = None    # placeholder for this fit parameter results [a scalar].

        self._prep_and_do_regression()
        self.df_mp_mags = self._calc_mp_mags()

    def _prep_and_do_regression(self):
        """ Using MixedModelFit class (which wraps statsmodels.MixedLM.from_formula()).
            Uses ONLY selected comp data in this model.
            (Use model's .predict() to calculate best MP mags from model and MP observations.)
        :return: [None]
        """
        fit_summary_lines = []
        # fixed_effect_var_list = []
        filters_to_include = list(self.color_def_dict['filters'].keys())

        # Prepare catmag offset to dep var:
        self.df_used_comp_obs['CatMag'] = 0.0
        for f in filters_to_include:
            catmag_passband = self.color_def_dict['filters'][f]['target passband']
            catmag_passband_column_name = common.CATMAG_PASSBAND_COLUMN_LOOKUP[catmag_passband]
            is_in_filter = list(self.df_used_comp_obs['Filter'] == f)
            self.df_used_comp_obs.loc[is_in_filter, 'CatMag'] = \
                self.df_used_comp_obs.loc[is_in_filter, catmag_passband_column_name]
        catmag_offset = self.df_used_comp_obs['CatMag'].astype(float)

        # Prepare transform*color offset to dep var, from instrument ini file:
        self.df_used_comp_obs['TransformValue'] = None
        self.df_used_comp_obs['TransformColor'] = None
        for f in filters_to_include:
            transform_value = self.color_dict['transforms'][f]
            transform_ci = self.color_def_dict['filters'][f]['transform ci']
            transform_ci_column_name = common.TRANSFORM_COLUMN_LOOKUP[transform_ci]
            is_in_filter = (self.df_used_comp_obs['Filter'] == f)
            self.df_used_comp_obs.loc[is_in_filter, 'TransformValue'] = transform_value
            self.df_used_comp_obs.loc[is_in_filter, 'TransformColor'] = \
                self.df_used_comp_obs.loc[is_in_filter, transform_ci_column_name]
        transform_offset = (self.df_used_comp_obs['TransformValue'] *
                            self.df_used_comp_obs['TransformColor']).astype(float)

        # Prepare extinction*airmass offset to dep var, from site ini file:
        self.df_used_comp_obs['ExtinctionValue'] = None
        for f in filters_to_include:
            extinction_value = self.color_dict['extinctions'][f]
            is_in_filter = (self.df_used_comp_obs['Filter'] == f)
            self.df_used_comp_obs.loc[is_in_filter, 'ExtinctionValue'] = extinction_value
        extinction_offset = (self.df_used_comp_obs['ExtinctionValue'] *
                             self.df_used_comp_obs['ObsAirmass']).astype(float)

        # Prepare offsets with all components except zero-point offsets (which are not known yet):
        partial_offset = catmag_offset + transform_offset + extinction_offset
        instmag_partially_offset = self.df_used_comp_obs['InstMag'] - partial_offset

        # Prepare *approximate* per-filter Zero-point (from main filter) offset to dep var:
        # Get mean offsets for each filter:
        mean_inst_mags = {}
        for f in filters_to_include:
            is_in_filter = (self.df_used_comp_obs['Filter'] == f)
            mean_inst_mags[f] = instmag_partially_offset[is_in_filter].mean()
        ref_filter = self.color_def_dict['reference filter']
        self.df_used_comp_obs['ZeroPointOffset'] = None
        for f in filters_to_include:
            instmag_fully_offset = mean_inst_mags[f] - mean_inst_mags[ref_filter]
            is_in_filter = (self.df_used_comp_obs['Filter'] == f)
            self.df_used_comp_obs.loc[is_in_filter, 'ZeroPointOffset'] = instmag_fully_offset
        zeropoint_offset = self.df_used_comp_obs['ZeroPointOffset'].astype(float)
        total_offset = partial_offset + zeropoint_offset
        self.df_used_comp_obs[self.dep_var_name] = (self.df_used_comp_obs['InstMag'] -
                                                    total_offset).astype(float)

        # Build fixed effect variables and values:
        fixed_effect_var_list = []
        fixed_effect_var_list.append('Vignette')  # mandatory fixed-effect (independent) variable.

        # for f in filters_to_include[1:]:
        #     is_in_filter = (self.df_used_comp_obs['Filter'] == f)
        #     column_name = 'dZ_' + f
        #     self.df_used_comp_obs[column_name] = 0.0
        #     self.df_used_comp_obs.loc[is_in_filter, column_name] = 1.0
        #     fixed_effect_var_list.append(column_name)

        # Complete regression preparations:
        random_effect_var_name = 'FITSfile'  # cirrus effect is per-image
        # dep_var_offset = catmag_offset + transform_offset + extinction_offset + zeropoint_offset
        # self.df_used_comp_obs[self.dep_var_name] = self.df_used_comp_obs['InstMag'] - dep_var_offset

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
        bogus_mp_ri_color = 0.0  # we'll need this below, to correct raw predictions.

        # Add required columns to df, make raw predictions:
        filter_fixed_effects = [fe for fe in self.mm_fit.fixed_vars if fe.startswith('dZ_')]
        for ffe in filter_fixed_effects:
            filter_name = ffe[3:]
            self.df_used_mp_obs[ffe] = [1 if f == filter_name else 0
                                        for f in self.df_used_mp_obs['Filter']]
        raw_predictions = self.mm_fit.predict(self.df_used_mp_obs, include_random_effect=True)

        instrument_mag_offset = self.df_used_mp_obs['InstMag'] * -1.0

        # Make extinction * Airmass offset:
        filters = self.df_used_mp_obs['Filter'].drop_duplicates()
        for f in filters:
            extinction_value = self.color_dict['extinctions'][f]
            is_in_filter = (self.df_used_mp_obs['Filter'] == f)
            self.df_used_mp_obs.loc[is_in_filter, 'ExtinctionValue'] = extinction_value
        extinction_offset = (self.df_used_mp_obs['ExtinctionValue'] *
                             self.df_used_mp_obs['ObsAirmass']).astype(float)

        # Return value is pandas series: best_mp_mag + transform*color_index.
        adjusted_mp_mags = -1.0 * (raw_predictions + instrument_mag_offset + extinction_offset)
        df_mp_mags = pd.DataFrame(data={'Adj_MP_Mags': adjusted_mp_mags},
                                  index=list(adjusted_mp_mags.index))
        df_mp_mags = pd.merge(left=df_mp_mags,
                              right=self.df_used_mp_obs.loc[:, ['JD_mid', 'JD_fract', 'FITSfile',
                                                                'InstMag', 'InstMagSigma', 'Filter']],
                              how='left', left_index=True, right_index=True, sort=False)
        return df_mp_mags


class ColorModel_2:
    """ Accepts adjusted MP magnitudes from ColorModel_1 and related data, effectives solves simultaneous
        equations to yield MP colors and other less-important results."""
    def __init__(self, color_model_1):
        """ Organize data, perform second regression, store MP colors & a bit of other data.
        :param color_model_1:
        :return: [None] See object attribute variables for results.
        """
        context, defaults_dict, color_dict, color_def_dict, log_file = _color_setup('ColorModel_2')
        df = color_model_1.df_mp_mags
        dep_var_name = 'Adj_MP_Mags'
        random_effect_name = 'FITSfile'

        # Make time columns:
        df['DT'] = df['JD_fract'] - df['JD_fract'].mean()
        df['DT2'] = df['DT'] ** 2

        # Make TransformValue column:
        df['TransformValue'] = None
        filters_to_include = df['Filter'].drop_duplicates()
        for f in filters_to_include:
            transform_value = color_dict['transforms'][f]
            is_in_filter = list(df['Filter'] == f)
            df.loc[is_in_filter, 'TransformValue'] = transform_value

        # Make ColorIndexFactor columns:
        reference_passband = color_def_dict['Reference']['Passband']


__________SUPPORT_FUNCTIONS_and_CLASSES = 0


def _get_color_context():
    """ Run at beginning of color workflow functions (ex start() or resume()) to orient the function.
        Assumes python current working directory = the relevant AN subdirectory with session.log in place.
        Adapted from package mp_phot, workflow_session.py._get_session_context(). Required for .resume().
        TESTED OK 2021-01-08.
    :return: 3-tuple: (this_directory, mp_string, an_string) [3 strings]
    """
    this_directory = os.getcwd()
    defaults_dict = ini.make_defaults_dict()
    color_log_filename = defaults_dict['color log filename']
    color_log_fullpath = os.path.join(this_directory, color_log_filename)
    if not os.path.isfile(color_log_fullpath):
        raise ColorLogFileError('No color log file found. You probably need to run start() or resume().')
    with open(color_log_fullpath, mode='r') as log_file:
        lines = log_file.readlines()
    if len(lines) < 5:
        raise ColorLogFileError('Too few lines.')
    if not lines[0].lower().startswith('color log file'):
        raise ColorLogFileError('Header line cannot be parsed.')
    directory_from_color_log = lines[1].strip().lower().replace('\\', '/').replace('//', '/')
    directory_from_cwd = this_directory.strip().lower().replace('\\', '/').replace('//', '/')
    if directory_from_color_log != directory_from_cwd:
        print()
        print(directory_from_color_log, directory_from_cwd)
        raise ColorLogFileError('Header line does not match current working directory.')
    mp_string = lines[2][3:].strip().upper()
    an_string = lines[3][3:].strip()
    # definition_string = lines[4][len('Definition:'):].strip()
    return this_directory, mp_string, an_string


def _write_color_ini_stub(this_directory, filenames_temporal_order, mean_datetime):
    """ Write color subdir's initial control (.ini) file, later to be edited by user.
        Called only by (at the end of) .assess().  DO NOT overwrite if color.ini already exists.
    :param this_directory:
    :param filenames_temporal_order: FITS filenames (all used filters) in ascending time order.
               [list of strings]
    :param mean_datetime: averate datetime of exposures. [py datetime object]
    :return: [None]
    """
    # Do not overwrite existing session ini file:
    defaults_dict = ini.make_defaults_dict()
    color_ini_filename = defaults_dict['color control filename']
    fullpath = os.path.join(this_directory, color_ini_filename)
    if os.path.exists(fullpath):
        return

    filename_earliest = filenames_temporal_order[0]
    filename_latest = filenames_temporal_order[-1]
    year_an = mean_datetime.year
    site_dict = ini.make_site_dict(defaults_dict)
    coldest_date = site_dict['coldest date'].split('-')
    coldest_datetime = datetime(year_an, int(coldest_date[0]),
                                int(coldest_date[1])).replace(tzinfo=timezone.utc)
    season_phase = (mean_datetime - coldest_datetime).total_seconds() / (365.25 * 24 * 3600)
    color_def_dict = ini.make_color_def_dict(defaults_dict)
    extinction_lines = []
    for f in color_def_dict['filters']:
        extinctions = site_dict['extinctions'][f]
        mean_ext = (extinctions[0] + extinctions[1]) / 2.0
        half_amplitude = (extinctions[0] - extinctions[1]) / 2.0
        extinction_an = mean_ext - half_amplitude * cos(2.0 * pi * season_phase)
        extinction_lines.append((' '.ljust(15) + f.ljust(8) + ' ' + '{:6.4f}'.format(extinction_an)))
    extinction_lines[0] = 'Extinctions ='.ljust(15) + extinction_lines[0][15:]

    inst_dict = ini.make_instrument_dict(defaults_dict)
    transform_lines = []
    for f in color_def_dict['filters']:
        f_def = color_def_dict['filters'][f]
        transform_key = (f, f_def['target passband'], f_def['transform ci'][0], f_def['transform ci'][1])
        transform = inst_dict['transforms'][transform_key][0]
        transform_lines.append((' '.ljust(15) + f.ljust(8) + str(transform)))
    transform_lines[0] = 'Transforms ='.ljust(15) + transform_lines[0][15:]

    header_lines = [
        '# This is ' + fullpath + '.',
        '']
    ini_lines = [
        '[Ini Template]',
        'Filename = color.template',
        '']
    color_definition_lines = [
        '[Color Definition]',
        'Color Definition Filename = ' + defaults_dict['color definition filename'],
        '']
    mp_location_lines = [
        '[MP Location]',
        '# Exactly 2 MP XY, one per line (typically earliest and latest FITS):',
        'MP XY = ' + filename_earliest + ' 000.0  000.0',
        '        ' + filename_latest + ' 000.0  000.0',
        '']
    selection_criteria_lines = [
        '[Selection Criteria]',
        'Omit Comps = ',
        'Omit Obs = ',
        '# One image only per line, with or without .fts:',
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
        '# Extinctions adapted from Site file \'' + defaults_dict['site ini'] +
        '\', adjusted for observing night.',
        '# Overwrite if needed (rare).'] + \
        extinction_lines + \
        ['# Transforms copied from Instrument file \'' + defaults_dict['instrument ini'] + '\'.'] +\
        transform_lines + \
        ['Fit Vignette = Yes']
    raw_lines = header_lines + ini_lines + color_definition_lines +\
        mp_location_lines + selection_criteria_lines + regression_lines
    ready_lines = [line + '\n' for line in raw_lines]
    if not os.path.exists(fullpath):
        with open(fullpath, 'w') as f:
            f.writelines(ready_lines)


def _color_setup(calling_function_name='[FUNCTION NAME NOT GIVEN]'):
    """ Typically called at the top of color workflow functions, to collect commonly required data.
    :return: tuple of data elements: context [tuple], defaults_dict [py dict], log_file [file object].
    """
    context = _get_color_context()
    if context is None:
        return
    this_directory, mp_string, an_string = context
    defaults_dict = ini.make_defaults_dict()
    color_dict = ini.make_color_dict(defaults_dict, this_directory)
    color_def_dict = ini.make_color_def_dict(color_dict)
    log_filename = defaults_dict['color log filename']
    log_file = open(log_filename, mode='a')  # set up append to log file.
    log_file.write('\n===== ' + calling_function_name + '()  ' +
                   '{:%Y-%m-%d  %H:%M:%S utc}'.format(datetime.now(timezone.utc)) + '\n')
    return context, defaults_dict, color_dict, color_def_dict, log_file
