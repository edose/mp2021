__author__ = "Eric Dose, Albuquerque"

""" This module: manages INI files for other modules. """

# Python core:
import os
import configparser
from collections import OrderedDict

# External packages:

# Author's packages:
import astropak.ini


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')
DEFAULTS_INI_FILENAME = 'defaults.ini'


class IniParseError(Exception):
    """ For parsing errors only; does not apply to float conversions etc. """


class ColorDefinitionError(Exception):
    """ For any problem with Color Definition File. """


def make_defaults_dict(ini_dir=INI_DIRECTORY, filename=DEFAULTS_INI_FILENAME):
    """ Reads .ini file, returns defaults_dict. TEST OK 2021-01-02.
        See defaults.template for value types and key names.
        :param ini_dir: the directory path where defaults ini file is found. [string]
        :param filename: defaults ini filename, typically 'defaults.ini'. [string]
    :return: the defaults_dict. [python dict object; All keys and values are strings]

    """
    fullpath = os.path.join(ini_dir, filename)
    defaults_ini = astropak.ini.IniFile(fullpath)
    return defaults_ini.value_dict


def make_instrument_dict(defaults_dict):
    """ Reads .ini file, returns instrument_dict. TESTS OK 2021-01-02.
        See instrument.template for value types and key names.
    :return: instrument_dict. [python dict object, some items nested dicts]
    """
    filename = defaults_dict['instrument ini']
    fullpath = os.path.join(INI_DIRECTORY, 'instrument', filename)
    instrument_ini = astropak.ini.IniFile(fullpath)
    instrument_dict = instrument_ini.value_dict

    # Parse and overwrite 'mag exposures':
    mag_exposure_dict = dict()
    mag_exposure_lines = [line.strip() for line in instrument_dict['mag exposures']]
    for line in mag_exposure_lines:
        mag_exposure_list = []
        filter_name, raw_value = tuple(line.split(maxsplit=1))
        pairs = raw_value.split(',')
        for pair in pairs:
            items = pair.split()
            if len(items) > 0:
                if len(items) != 2:
                    raise IniParseError('Instrument ini: mag exposures line: ' + line)
                try:
                    mag, exp = float(items[0]), float(items[1])
                except ValueError as e:
                    raise ValueError(' '.join([str(e), 'in mag exposures line:', line])) from None
                mag_exposure_list.append((mag, exp))
        mag_exposure_dict[filter_name] = tuple(mag_exposure_list)
    instrument_dict['mag exposures'] = mag_exposure_dict

    # Parse and overwrite 'transforms':
    transform_dict = dict()
    transform_lines = [line.strip() for line in instrument_dict['transforms']]
    for line in transform_lines:
        items = [item.strip() for item in line.replace(',', ' ').split()]
        if len(items) not in [5, 6]:
            raise IniParseError('Instrument ini: transforms line: ' + line)
        key = tuple(items[:4])
        try:
            values = tuple([float(item) for item in items[4:]])
        except ValueError as e:
            raise ValueError(' '.join([str(e), 'in transforms line', line])) from None
        transform_dict[key] = values
    instrument_dict['transforms'] = transform_dict

    # Parse and overwrite 'available filters', 'default color filters', 'default color index':
    instrument_dict['available filters'] = tuple(instrument_dict['available filters'].split())
    instrument_dict['default color filters'] = tuple(instrument_dict['default color filters'].split())
    instrument_dict['default color index'] = \
        tuple([s.strip() for s in instrument_dict['default color index'].split('-')])
    return instrument_dict


def make_observer_dict(defaults_dict):
    """ Reads .ini file, returns observer_dict. TESTS OK 2021-01-02.
        Used mostly for ALCDEF file generation.
    :return: observer_dict. [python dict object, all keys and values are strings]
    See observer.template for value types and key names.
    """
    filename = defaults_dict['observer ini']
    fullpath = os.path.join(INI_DIRECTORY, 'observer', filename)
    observer_ini = astropak.ini.IniFile(fullpath)
    observer_dict = observer_ini.value_dict
    return observer_dict


def make_site_dict(defaults_dict):
    """ Reads .ini file, returns site_dict. TESTS OK 2021-01-02.
    :return: site_dict. [python dict object, some values are nested dicts.]
    See site.template for value types and key names.
    """
    filename = defaults_dict['site ini']
    fullpath = os.path.join(INI_DIRECTORY, 'site', filename)
    site_ini = astropak.ini.IniFile(fullpath)
    site_dict = site_ini.value_dict

    # Parse and overwrite 'extinctions':
    extinction_dict = dict()
    for line in site_dict['extinctions']:
        items = line.replace(',', ' ').split()
        if len(items) != 3:
            raise IniParseError('Site ini: extinction line: ' + line)
        filter_name = items[0]
        try:
            summer_extinction, winter_extinction = float(items[1]), float(items[2])
        except ValueError:
            raise IniParseError('Site ini: bad extinction line:', line) from None
        extinction_dict[filter_name] = tuple([summer_extinction, winter_extinction])
    site_dict['extinctions'] = extinction_dict
    return site_dict


def make_session_dict(defaults_dict, session_directory):
    """ Read the session control file for this lightcurve session, return session dict.
    :param defaults_dict:
    :param session_directory:
    :return: session_dict [py dict].
    Structure of session_dict:
      { 'mp xy': [(filename1, x1, y1), (filename2, x2, y2)],
        'omit comps': ['22', '44', '4221'],
        'omit obs': ['4223', '429']
        'omit images': ['MP_1443_0001-Clear.fts'],
        'min catalog r mag': 12.5,
        'max catalog r mag': 16,
        'max catalog dr mmag': 15,
        'min catalog ri color': 0.10,
        'max catalog ri color': 0.34,
        'mp ri color': +0.22,
        'fit transform': ('use', '+0.4', '-0.16'), or ('use', '+0.4'), ('fit', '1'), ('fit', '2').
        'fit extinction': (use', '+0.16'), or 'yes' to fit extinction.
        'fit vignette': True,
        'fit xy': False,
        'fit jd': True }
    """
    session_ini_filename = defaults_dict['session control filename']
    fullpath = os.path.join(session_directory, session_ini_filename)
    session_ini = astropak.ini.IniFile(fullpath, template_directory_path=INI_DIRECTORY)
    session_dict = session_ini.value_dict  # raw values, a few to be reparsed just below:

    # ########## package mp2021 will have NO Ref Stars, as they are not needed until Bulldozer.
    # # Bulldozer section:
    # # Parse and overwrite 'ref star xy':
    # ref_star_xy_list = []
    # ref_star_xy_lines = [line.strip() for line in control_dict['ref star xy']]
    # for line in ref_star_xy_lines:
    #     items = line.replace(',', ' ').rsplit(maxsplit=2)  # for each line, items are: filename x y
    #     if len(items) == 3:
    #         filename = items[0]
    #         x = ini.float_or_warn(items[1], filename + 'Ref Star X' + items[1])
    #         y = ini.float_or_warn(items[2], filename + 'Ref Star Y' + items[2])
    #         ref_star_xy_list.append((filename, x, y))
    #     elif len(items >= 1):
    #         print(' >>>>> ERROR: ' + items[1] + ' Ref Star XY invalid: ' + line)
    #         return None
    # if len(ref_star_xy_list) < 2:
    #     print(' >>>>> ERROR: control \'ref star xy\' has fewer than 2 entries, not allowed.')
    # control_dict['ref star xy'] = ref_star_xy_list

    # Parse and overwrite 'mp xy':
    mp_xy_list = _extract_mp_xy_positions(session_dict, 'Session')
    session_dict['mp xy'] = mp_xy_list

    # Selection Criteria section, Omit elements:
    session_dict = _extract_omit_comps_obs_images(session_dict, 'Session')

    # Standardize remaining elements:
    try:
        _ = float(session_dict['mp ri color'])
    except ValueError:
        raise ValueError('Session ini: MP ri Color is not a float: ' +
                         session_dict['mp ri color']) from None
    session_dict['fit transform'] = tuple([item.lower()
                                           for item in session_dict['fit transform'].split()])
    session_dict['fit extinction'] = tuple([item.lower()
                                            for item in session_dict['fit extinction'].split()])
    if len(session_dict['fit extinction']) == 1:
        session_dict['fit extinction'] = session_dict['fit extinction'][0]
    return session_dict


def make_color_dict(defaults_dict, color_directory):
    """ Read the color control file for this color subdirectory, return color dict.
    :param defaults_dict:
    :param color_directory:
    :return: color_dict. [py dict]
    Structure of color_dict:
      { 'mp xy': [(filename1, x1, y1), (filename2, x2, y2)],
        'omit comps': ['22', '44', '4221'],
        'omit obs': ['4223', '429']
        'omit images': ['MP_1443_0001-Clear.fts'],
        'min catalog r mag': 12.5,
        'max catalog r mag': 16,
        'max catalog dr mmag': 15,
        'min catalog ri color': 0.10,
        'max catalog ri color': 0.34,
        'mp ri color': +0.22,
        'extinctions': {'V': 0.1615, 'R': 01292, 'I': 0.0869},
        'transforms': {'V': -0.015, 'R': -0.15, 'I': 0.11},
        'fit vignette': True }
    """
    color_ini_filename = defaults_dict['color control filename']
    fullpath = os.path.join(color_directory, color_ini_filename)
    color_ini = astropak.ini.IniFile(fullpath, template_directory_path=INI_DIRECTORY)
    color_dict = color_ini.value_dict  # raw values, a few to be reparsed just below:

    # Parse and overwrite 'mp xy':
    mp_xy_list = _extract_mp_xy_positions(color_dict, 'Color')
    color_dict['mp xy'] = mp_xy_list

    # Selection Criteria section, Omit elements:
    color_dict = _extract_omit_comps_obs_images(color_dict, 'Color')

    # Parse and replace 'extinctions':
    extinction_lines = [line.strip() for line in color_dict['extinctions']]
    extinction_dict = OrderedDict()
    for line in extinction_lines:
        items = line.replace(',', ' ').split()
        if len(items) != 2:
            raise IniParseError(color_ini_filename + ', Extinction line: ' + line)
        try:
            value = float(items[1])
        except ValueError:
            raise ValueError((color_ini_filename + ', Extinction value: ' + items[1]))
        extinction_dict[items[0]] = value
    color_dict['extinctions'] = extinction_dict

    # Parse and replace 'transforms':
    transform_lines = [line.strip() for line in color_dict['transforms']]
    transform_dict = OrderedDict()
    for line in transform_lines:
        items = line.replace(',', ' ').split()
        if len(items) != 2:
            raise IniParseError(color_ini_filename + ', Transform line: ' + line)
        try:
            value = float(items[1])
        except ValueError:
            raise ValueError((color_ini_filename + ', Transform value: ' + items[1]))
        transform_dict[items[0]] = value
    color_dict['transforms'] = transform_dict

    return color_dict


def make_color_def_dict(defaults_dict):
    """ Read the color definition .ini file for this color subdirectory, return color def dict.
    :param color_dict:
    :return: color_def_dict. [py dict]
    Structure of color_def_dict:
      { 'target colors': [('SG', 'SR'), ('SR', 'SI)], (for target colors SG-SR, SR-SI)
        'filters': {'V': {'name': 'Johnson-Cousins V',
                          'target passband': 'SG',
                          'transform ci': 'SG', 'SR')},
                   {'R': {'name': 'Johnson-Cousins R',
                          'target passband': 'SR',
                          'transform ci': 'SR', 'SI')},
                   {'I': {'name': 'Johnson-Cousins I',
                          'target passband': 'SI',
                          'transform ci': 'SR', 'SI')}
      }
    """
    color_def_filename = defaults_dict['color definition filename']
    color_def_fullpath = os.path.join(INI_DIRECTORY, 'color', color_def_filename)
    if not (os.path.exists(color_def_fullpath) and os.path.isfile(color_def_fullpath)):
        raise ColorDefinitionError('Requested file not found: ' + color_def_fullpath)
    ini_config = configparser.ConfigParser()
    ini_config.read(color_def_fullpath)
    color_def_dict = OrderedDict()

    target_color_strings = ini_config.get('Targets', 'Target Colors').split('\n')
    color_def_dict['target_colors'] = [tuple(i.strip() for i in s.split('-'))
                                       for s in target_color_strings]
    color_def_dict['reference filter'] = ini_config.get('Reference', 'Filter')
    color_def_dict['reference passband'] = ini_config.get('Reference', 'Passband')

    filter_sections = [fs for fs in ini_config.sections() if fs.lower().startswith('filter')]
    filters = [fs[6:].strip() for fs in filter_sections]
    filters_dict = OrderedDict()
    for f, fs in zip(filters, filter_sections):
        filters_dict[f] = {'name': ini_config.get(fs, 'Name'),
                           'target passband': ini_config.get(fs, 'Target Passband'),
                           'transform ci': tuple(ini_config.get(fs, 'Transform CI').split('-'))}
    color_def_dict['filters'] = filters_dict

    # Verify: reference filter and passband match in exactly one Filter section.
    n_filter_matches = sum([1 if f == color_def_dict['reference filter'] else 0 for f in filters_dict])
    if n_filter_matches != 1:
        raise ColorDefinitionError('Reference filter ' + color_def_dict['reference filter'] +
                                   ' must be present in exactly one Filter section, but is in ' +
                                   str(n_filter_matches))
    n_passband_matches = sum([1 if filters_dict[f]['target passband'] ==
                                   color_def_dict['reference passband']
                              else 0 for f in filters_dict])
    if n_passband_matches != 1:
        raise ColorDefinitionError('Reference passband ' + color_def_dict['reference passband'] +
                                   ' must be present in exactly one Filter section, but is in ' +
                                   str(n_passband_matches))

    # Verify: target colors eactly equals *set of* Transform CI colors.
    target_color_set = set(color_def_dict['target_colors'])
    transform_ci_set = set([filters_dict[f]['transform ci'] for f in filters_dict])
    if target_color_set != transform_ci_set:
        raise ColorDefinitionError('Target Colors must equal set of transform CI colors, but do not.')

    return color_def_dict


_____INI_UTILITIES______________________________________ = 0


def _extract_mp_xy_positions(control_dict, control_type):
    """ From session or color control dict, extract and return MP x,y positions given for 2 FITS files.
    :param control_dict: session or color control dict. [py dict]
    :param control_type: 'Session' or 'Color', as appropriate. [string]
    :return:
    """
    control_string = control_type.strip().title()
    mp_xy_list = []
    mp_xy_lines = [line.strip() for line in control_dict['mp xy']]
    for line in mp_xy_lines:
        items = line.replace(',', ' ').rsplit(maxsplit=2)  # for each line, items are: filename x y
        if len(items) != 3:
            raise IniParseError(control_string + ' ini: MP x,y line: ' + line)
        filename = items[0]
        try:
            x, y = float(items[1]), float(items[2])
        except ValueError:
            raise ValueError(control_string + ' ini: MP x or y is not a float.')
        mp_xy_list.append((filename, x, y))
    if len(mp_xy_list) != 2:
        raise IniParseError(control_string + ' ini: MP x,y lines must number exactly 2.')
    return mp_xy_list


def _extract_omit_comps_obs_images(control_dict, control_type):
    """ From session or color control dict, extract identifiers for comp stars, observations, and
        images to omit from regression, return an updated copy of the control_dict.
    :param control_dict: session or color control dict. [py dict]
    :param control_type: 'Session' or 'Color', as appropriate. [string]
    :return:
    """
    control_string = control_type.strip().title()
    control_dict['omit comps'] = _multiline_ini_value_to_items(' '.join(control_dict['omit comps']))
    control_dict['omit obs'] = _multiline_ini_value_to_items(' '.join(control_dict['omit obs']))
    control_dict['omit images'] = [s.strip() for s in control_dict['omit images']]
    try:
        omit_comps_as_ints = [int(comp) for comp in control_dict['omit comps']]
    except ValueError:
        error_string = control_string + ' ini: at least one Omit Comps entry is not an integer.'
        raise ValueError(error_string) from None
    if any([(i < 1) for i in omit_comps_as_ints]):
        error_string = control_string + ' ini: at least one Omit Comps entry is not a positive integer.'
        raise IniParseError(error_string) from None
    try:
        omit_obs_as_ints = [int(obs) for obs in control_dict['omit obs']]
    except ValueError:
        error_string = control_string + ' ini: at least one Omit Obs entry is not an integer.'
        raise ValueError(error_string) from None
    if any([(i < 1) for i in omit_obs_as_ints]):
        error_string = control_string + ' ini: at least one Omit Obs entry is not a positive integer.'
        raise IniParseError(error_string) from None
    return control_dict


def _multiline_ini_value_to_lines(value):
    lines = list(filter(None, (x.strip() for x in value.splitlines())))
    lines = [line.replace(',', ' ').strip() for line in lines]  # replace commas with spaces
    return lines


def _multiline_ini_value_to_items(value):
    lines = _multiline_ini_value_to_lines(value)
    value_list = []
    _ = [value_list.extend(line.split()) for line in lines]
    return value_list