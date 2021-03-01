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
            raise ValueError(' >>>>> ERROR:', filename, 'bad extinction line:', line) from None
        extinction_dict[filter_name] = tuple([summer_extinction, winter_extinction])
    site_dict['extinctions'] = extinction_dict
    return site_dict
