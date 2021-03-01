__author__ = "Eric Dose, Albuquerque"

# Python core:
import os
import shutil

# External packages:
import pytest

# Author's packages:
import mp2021.phot as phot
import mp2021.ini as ini
from mp2021.util import get_mp_filenames


THIS_PACKAGE_ROOT_DIRECTORY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_SESSION_TOP_DIRECTORY = os.path.join(THIS_PACKAGE_ROOT_DIRECTORY, 'test', '$sessions_for_test')
# INI_SUBDIRECTORY = 'ini'
# STARTUP_INI_FILENAME = 'defaults.ini'
SOURCE_TEST_MP = 191
TEMP_TEST_MP = 1111
TEST_AN = 20200617


def test_start(_temporary_session_directory):
    """ Create temporary test directory, fill it with images, run start(), do asserts,
            tear down temporary test directory. """
    source_path, temp_path = _temporary_session_directory
    os.chdir(source_path)
    defaults_dict = ini.make_defaults_dict()
    log_filename = defaults_dict['session log filename']
    temp_log_fullpath = os.path.join(temp_path, log_filename)
    assert not os.path.isfile(temp_log_fullpath)  # before start().
    assert os.getcwd() == source_path
    phot.start(TEST_SESSION_TOP_DIRECTORY, TEMP_TEST_MP, TEST_AN, 'Clear')
    assert os.getcwd() == temp_path
    assert os.path.isfile(temp_log_fullpath)
    assert set(get_mp_filenames(temp_path)) == set(get_mp_filenames(source_path))


def test__get_session_context():
    chdir_directory = os.path.join(TEST_SESSION_TOP_DIRECTORY,
                                   'MP_' + str(SOURCE_TEST_MP), 'AN' + str(TEST_AN))
    os.chdir(chdir_directory)  # start at source directory.
    context_directory, mp_string, an_string, filter_string = phot._get_session_context()
    assert context_directory == chdir_directory
    assert mp_string == '191'
    assert an_string == '20200617'
    assert filter_string == 'Clear'
    # We won't test exceptions at this time.


def test_resume(_temporary_session_directory):
    # Build temp directory and write session.log to it:
    source_path, temp_path = _temporary_session_directory
    phot.start(TEST_SESSION_TOP_DIRECTORY, TEMP_TEST_MP, TEST_AN, 'Clear')

    # Set current working directory elsewhere, then verify .resume() can re-establish in correct directory:
    os.chdir(source_path)
    assert os.getcwd() == source_path
    phot.resume(TEST_SESSION_TOP_DIRECTORY, TEMP_TEST_MP, TEST_AN, filter='Clear')
    assert os.getcwd() == temp_path
    assert phot._get_session_context() == (temp_path, str(TEMP_TEST_MP), str(TEST_AN), 'Clear')


def test_assess(_temporary_session_directory):
    """ Using same temporary test directory as test_start(). """
    source_path, temp_path = _temporary_session_directory
    phot.start(TEST_SESSION_TOP_DIRECTORY, TEMP_TEST_MP, TEST_AN, 'Clear')
    d = phot.assess(return_results=True)


    assert d['file not read'] == d['filter not read'] == []
    assert set(d['file count by filter']) == set([('Clear', 5), ('R', 1), ('I', 1)])
    assert d['warning count'] == 0
    assert d['not platesolved'] == d['not calibrated'] == []
    assert d['unusual fwhm'] == d['unusual focal length'] == []
    defaults_dict = ini.make_defaults_dict()
    log_filename = defaults_dict['session log filename']
    log_file_fullpath = os.path.join(new_dir_path, log_filename)
    assert os.path.isfile(log_file_fullpath)
    control_filename = defaults_dict['session control filename']
    control_fullpath = os.path.join(new_dir_path, control_filename)
    assert os.path.isfile(control_fullpath)



_____FIXTURES_and_HELPER_FUNCTIONS______________________________________ = 0


@pytest.fixture
def _temporary_session_directory():
    source_path = os.path.join(TEST_SESSION_TOP_DIRECTORY, 'MP_' + str(SOURCE_TEST_MP), 'AN' + str(TEST_AN))
    temp_path = os.path.join(TEST_SESSION_TOP_DIRECTORY, 'MP_' + str(TEMP_TEST_MP), 'AN' + str(TEST_AN))
    _delete_test_session_directory(temp_path)  # in case pre-existing.
    _make_test_session_directory(source_path, temp_path)
    yield source_path, temp_path  # this statement divides the setup (above) from the teardown (below).
    _delete_test_session_directory(temp_path)


def _make_test_session_directory(source_path, temp_path):
    """ Make a fresh test directory (probably with test MP not matching its filename MP).
    :param source_path: from which FITS files are copied, treated as read-only. [string]
    :param temp_path: new test directory to populate with FITS files from source_path. [string]
    :return: [None]
    """
    os.makedirs(temp_path, exist_ok=True)
    fits_filenames = get_mp_filenames(source_path)
    for fn in fits_filenames:
        source_fullpath = os.path.join(source_path, fn)
        shutil.copy2(source_fullpath, temp_path)


def _delete_test_session_directory(temp_path):
    shutil.rmtree(temp_path, ignore_errors=True)   # NB: this doesn't always work in test environment.