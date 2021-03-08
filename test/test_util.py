__author__ = "Eric Dose, Albuquerque"

""" This module: tests module "util". """

# Python core:
import os

# External packages:
import pytest
import astropy.io.fits as apyfits

# Author's packages:
import mp2021.util as util


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# INI_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'ini')
TEST_SESSIONS_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'test', '$sessions_for_test')
SOURCE_TEST_MP = 191
TEMP_TEST_MP = 1111
TEST_AN = 20200617


def test_parse_mp_id():
    assert util.parse_mp_id(1108) == '1108'
    assert util.parse_mp_id('1108') == '1108'
    assert util.parse_mp_id('1998 MX2') == '~1998 MX2'
    with pytest.raises(TypeError):
        _ = util.parse_mp_id(5.3)
    with pytest.raises(ValueError):
        _ = util.parse_mp_id(-1108)
        _ = util.parse_mp_id('Vesta')


def test_parse_an_date():
    assert util.parse_an_date(20200617) == '20200617'
    assert util.parse_an_date('20200617') == '20200617'
    with pytest.raises(TypeError):
        _ = util.parse_an_date(5.3)
    with pytest.raises(ValueError):
        util.parse_an_date('Not representing an integer')
        util.parse_an_date('21040617')
        util.parse_an_date(0)
        util.parse_an_date('2020-06-17')


def test_get_mp_filenames():
    this_directory = os.path.join(TEST_SESSIONS_DIRECTORY,
                                  'MP_' + str(SOURCE_TEST_MP), 'AN' + str(TEST_AN))
    mp_filenames = util.get_mp_filenames(this_directory)
    assert isinstance(mp_filenames, list)
    assert all([isinstance(fn, str) for fn in mp_filenames])
    assert len(mp_filenames) == 7
    assert all([fn.startswith('MP_') for fn in mp_filenames])
    assert all([fn[-4:] in util.VALID_FITS_FILE_EXTENSIONS for fn in mp_filenames])
    assert len(set(mp_filenames)) == len(mp_filenames)  # filenames are unique.


def test_fits_header_value():
    fullpath = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_191', 'AN20200617', 'MP_191-0001-Clear.fts')
    hdu = apyfits.open(fullpath)[0]
    assert util.fits_header_value(hdu, 'FILTER') == 'Clear'
    assert util.fits_header_value(hdu, 'NAXIS1') == 3072
    assert util.fits_header_value(hdu, 'INVALID') is None


def test_fits_is_plate_solved():
    fullpath = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_191', 'AN20200617', 'MP_191-0001-Clear.fts')
    hdu = apyfits.open(fullpath)[0]
    assert util.fits_is_plate_solved(hdu) is True
    hdu.header['CRVAL1'] = 'INVALID'
    assert util.fits_is_plate_solved(hdu) is False
    hdu = apyfits.open(fullpath)[0]
    del hdu.header['CD2_1']
    assert util.fits_is_plate_solved(hdu) is False


def test_fits_is_calibrated():
    fullpath = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_191', 'AN20200617', 'MP_191-0001-Clear.fts')
    hdu = apyfits.open(fullpath)[0]
    assert util.fits_is_calibrated(hdu) is True
    hdu.header['CALSTAT'] = 'INVALID'
    assert util.fits_is_calibrated(hdu) is False
    del hdu.header['CALSTAT']
    assert util.fits_is_calibrated(hdu) is False


def test_fits_focal_length():
    fullpath = os.path.join(TEST_SESSIONS_DIRECTORY, 'MP_191', 'AN20200617', 'MP_191-0001-Clear.fts')
    hdu = apyfits.open(fullpath)[0]
    focallen_value = util.fits_focal_length(hdu)
    assert focallen_value == 2713.5
    del hdu.header['FOCALLEN']
    fl_value = util.fits_focal_length(hdu)
    assert 0.97 * focallen_value <= fl_value <= 1.03 * focallen_value
