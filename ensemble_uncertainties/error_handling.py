
"""Handle errors and inform user."""

import errno

from ensemble_uncertainties.model_library import models

from os.path import isfile


def file_availability(path):
    """Checks if CSV-file exists.

    Parameters
    ----------
    path : str
        Path to the CSV-file
    """
    if not isfile(path):
        FileNotFoundError  (f'File "{path}" does not exist.')     


def file_compatibility(path):
    """Checks if CSV-file fulfills demanded requirements.

    Parameters
    ----------
    path : str
        Path to the CSV-file
    """
    with open(path) as f:
        first_line = f.readline()
        if not first_line.startswith('id;'):
            raise IOError(
                f'File "{path}" does not fulfill requirements.\n' + \
                'The file must be a semicolon-separated CSV-file ' + \
                'with a header and an index column named "id".'
            )


def y_file_compatibility(path):
    """Checks if CSV-file of targets fulfills demanded requirements.

    Parameters
    ----------
    path : str
        Path to the CSV-file
    """
    with open(path) as f:
        first_line = f.readline()
        if not first_line.startswith('id;y'):
            raise IOError(
                f'File "{path}" does not fulfill requirements.\n' + \
                'The file must be a semicolon-separated CSV-file ' + \
                'with a header and an index column named "id" and ' + \
                'the column of targets must be named "y".'
            )


def model_availability(task, model_name):
    """Checks if the model with the given
    name is available for the given task.

    Parameters
    ----------
    task : str
        'regression' for regression tasks, 'classification' otherwise
    model_name : str
        Name of the model to check for
    """
    names = models[task].keys()
    if model_name not in names:
        raise KeyError(
            f'Model "{model_name}" is not available.\n' +
            f'Available models are: {", ".join(names)}.'
        )


def writing_accessibility(path):
    """Checks if writing into the specified directory is allowed.

    Parameters
    ----------
    path : str
        Path to the directory
    """
    try:
        _ = open(path)
    except IOError as e:
        if e.errno == errno.EACCES:
            raise PermissionError(f'No permission to write in "{path}."')
