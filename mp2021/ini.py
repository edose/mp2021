__author__ = "Eric Dose, Albuquerque"

""" This module: manages INI files for other modules. """

# Python core:
import os

# External packages:

# Author's packages:
import astropak.ini


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')
DEFAULTS_INI_FILENAME = 'defaults.ini'


class IniParseError(Exception):
    """ For parsing errors only; does not apply to float conversions etc. """


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
        'fit transform': ('Use', '+0.4', '-0.16'), or ('Use', '+0.4'), ('Fit', '1'), ('Fit', '2').
        'fit extinction': (Use', '+0.16'), or 'Yes' to fit extinction.
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
    mp_xy_list = []
    mp_xy_lines = [line.strip() for line in session_dict['mp xy']]
    for line in mp_xy_lines:
        items = line.replace(',', ' ').rsplit(maxsplit=2)  # for each line, items are: filename x y
        if len(items) != 3:
            raise IniParseError('Session ini: MP x,y line: ' + line)
        filename = items[0]
        try:
            x, y = float(items[1]), float(items[2])
        except ValueError:
            raise ValueError('Session ini: MP x or y is not a float.')
        mp_xy_list.append((filename, x, y))
    if len(mp_xy_list) != 2:
        raise IniParseError('Session ini: MP x,y lines must number exactly 2.')
    session_dict['mp xy'] = mp_xy_list

    # Selection Criteria section, Omit elements:
    session_dict['omit comps'] = _multiline_ini_value_to_items(' '.join(session_dict['omit comps']))
    session_dict['omit obs'] = _multiline_ini_value_to_items(' '.join(session_dict['omit obs']))
    session_dict['omit images'] = [s.strip() for s in session_dict['omit images']]
    try:
        omit_comps_as_ints = [int(comp) for comp in session_dict['omit comps']]
    except ValueError:
        raise ValueError('Session ini: at least one Omit Comps entry is not an integer.') from None
    if any([(i < 1) for i in omit_comps_as_ints]):
        raise IniParseError('Session ini: at least one Omit Comps entry is not a positive integer.') \
            from None
    try:
        omit_obs_as_ints = [int(obs) for obs in session_dict['omit obs']]
    except ValueError:
        raise ValueError('Session ini: at least one Omit Obs entry is not an integer.') from None
    if any([(i < 1) for i in omit_obs_as_ints]):
        raise IniParseError('Session ini: at least one Omit Obs entry is not a positive integer.') from None

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


_____INI_UTILITIES______________________________________ = 0


def _multiline_ini_value_to_lines(value):
    lines = list(filter(None, (x.strip() for x in value.splitlines())))
    lines = [line.replace(',', ' ').strip() for line in lines]  # replace commas with spaces
    return lines


def _multiline_ini_value_to_items(value):
    lines = _multiline_ini_value_to_lines(value)
    value_list = []
    _ = [value_list.extend(line.split()) for line in lines]
    return value_list