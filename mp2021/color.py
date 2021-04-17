__author__ = "Eric Dose, Albuquerque"

# Python core:
import os
from datetime import datetime, timezone

# External packages:

# Author's packages:
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
    _write_color_ini_stub(this_directory, filenames_temporal_order)  # if it doesn't already exist.
    if return_results:
        return return_dict


def make_dfs(print_ap_details=False):
    """ Perform aperture photometry for one subdirectory of color photometry.
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
    comp_apertures_dict = common.make_comp_apertures(fits_objects, df_comps, disc_radius, gap, background_width)
    df_comp_obs = common.make_df_comp_obs(comp_apertures_dict, df_comps, instrument, df_images)

    # Make MP apertures and MP obs dataframe:
    starting_mp_obs_id = max(df_comp_obs['ObsID'].astype(int)) + 1
    mp_apertures_dict, mp_mid_radec_dict =common.make_mp_apertures(fits_object_dict, mp_string,
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
          '\n      (2) run do_color()\n')


def do_color():
    """ Primary color photometry for one color subdirectory.
    Takes the 4 CSV files from make_dfs().
    Generates diagnostic plots for iterative regression refinement.
    Typically iterated, pruning comp-star ranges and outliers, until converged and then simply stop.
    :returns None. Writes all info to files.
    USAGE: do_color()   [no return value]
    """
    context, defaults_dict, color_dict, color_def_dict, log_file = _color_setup('make_dfs')
    this_directory, mp_string, an_string = context
    filters_to_include = color_def_dict['filters']
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
    model = ColorModel_1(df_model, color_dict, color_def_dict, df_mp_master, this_directory)






__________SUPPORT_for_do_color____________________________________________ = 0


class ColorModel_1:
    """ Generates and holds mixed-model regression model suitable to MP color index determination.
        Affords prediction for MP magnitudes. """
    def __init__(self, df_model, color_dict, color_def_dict, df_mp_master, this_directory):
        self.df_model = df_model
        self.color_dict = color_dict
        self.color_def_dict = color_def_dict
        self.df_used_comps_obs = self.df_model.copy().loc[self.df_model['UseInModel'], :]
        images_in_used_comps = self.df_used_comps_obs['FITSfile'].drop_duplicates()
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


    def _calc_mp_mags(self):
        """ Use model and MP instrument magnitudes to get best estimates of MP absolute magnitudes."""
        return 1111


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


def _write_color_ini_stub(this_directory, filenames_temporal_order):
    """ Write color subdir's initial control (.ini) file, later to be edited by user.
        Called only by (at the end of) .assess().  DO NOT overwrite if color.ini already exists.
    :param this_directory:
    :param filenames_temporal_order: FITS filenames (all used filters) in ascending time order.
               [list of strings]
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
        '# Extinctions from Site .ini file.',
        '# Transforms from Instrument .ini file',
        'Fit Vignette = Yes']
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


