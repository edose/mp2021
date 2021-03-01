__author__ = "Eric Dose, Albuquerque"

""" This module: 
      
"""

# Python core:
import os

# External packages:
import pytest

# Author's packages:
import mp2021.ini as ini


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')


def test_make_defaults_dict():
    dd = ini.make_defaults_dict(ini_dir=INI_DIRECTORY, filename='defaults.ini')
    assert dd['instrument ini'] == 'BoreaC14.ini'
    assert dd['session log filename'] == 'session.log'
    assert dd['color control filename'] == 'color_control.ini'
    assert dd.get('not an element') is None


def test_make_instrument_dict():
    mock_defaults_dict = {'instrument ini': '$test_only.ini'}
    id = ini.make_instrument_dict(mock_defaults_dict)
    assert id['min mp altitude'] == 29.5
    assert id['min moon distance'] == 45
    assert id['min hours mp observable'] == 25
    assert id['max v magnitude'] == 18
    assert id['mount model'] == 'PlaneWave L-500 hahaha'
    assert id['aperture'] == pytest.approx(0.35)
    assert id['x pixels'] == 3048
    assert id['y pixels'] == 2047
    assert id['saturation adu'] == 54000
    assert id['max vignetting pct at corner'] == 38
    assert id['pinpoint pixel scale multiplier'] == pytest.approx(0.99388)
    assert id['available filters'] == ('B', 'V', 'R', 'I', 'Clear', 'SG', 'SR', 'SI')
    assert id['mag exposures'] == {'Clear': ((13, 60), (14, 80), (15, 160), (16, 300),
                                             (17, 600), (17.5, 900)),
                                   'SR': ((13, 120), (14, 160), (15, 320), (16, 600), (17, 900)),
                                   'SI': ((13, 120), (14, 160), (15, 320), (16, 600), (17, 900))}
    assert id['transforms'] == {('Clear', 'SR', 'SR', 'SI'): (+0.5, -0.75),
                                ('SR', 'SR', 'SR', 'SI'): (-0.03125,),
                                ('R', 'SR', 'SR', 'SI'): (-0.125,),
                                ('I', 'SI', 'SR', 'SI'): (-0.0625,)}
    assert id['default color filters'] == ('R', 'I')
    assert id['default color index'] == ('SR', 'SI')
    assert id['min fwhm pixels'] == 1.5
    assert id['max fwhm pixels'] == 14
    assert id['nominal fwhm pixels'] == 5
    assert id['exposure overhead'] == 20
    assert id['max exposure no guiding'] == 119
    assert id.get('not an element') is None


def test_make_observer_dict():
    mock_defaults_dict = {'observer ini': '$test_observer.ini'}
    od = ini.make_observer_dict(mock_defaults_dict)
    assert od['name'] == 'Eric Dose'
    assert od['alcdef contact name'] == 'Eric V. Dose'
    assert od['alcdef contact info'] == 'mp@ericdose.com'
    assert od['alcdef observers'] == 'Dose, E.V.'
    assert od['alcdef filter'] == 'C'
    assert od['alcdef mag band'] == 'SR'
    assert od.get('not an element') is None


def test_make_site_dict():
    mock_defaults_dict = {'site ini': '$test_site.ini'}
    sd = ini.make_site_dict(mock_defaults_dict)
    assert sd['name'] == 'Deep Sky West, Beta building'
    assert sd['mpc code'] == 'V28'
    assert sd['longitude'] == pytest.approx(254.34647)
    assert sd['latitude'] == pytest.approx(35.3311)
    assert sd['elevation'] == 2210
    assert sd['coldest date'] == '01-25'
    assert sd['extinctions'] == {'Clear': (0.16, 0.14),
                                 'I': (0.11, 0.08)}
    assert sd.get('not an element') is None
