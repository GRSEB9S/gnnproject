"""
    Define error handling utility functions
"""
from gnnconst import LANGUAGE as LNG
from gnnconst import errmsg


def ok(dtype, data):
    """
    Prepare an OK message to be used in a response
    Args:
        dtype: (string) data type - can be any of the known python types
        data: a data structure that matches the type
    Returns:
        An outcome message
    """
    outcome = {
        "status": "ok",
        "type": dtype,
        "data": data
    }
    return outcome


def error(error_type):
    """
    Prepare an error message to be used in a response
    Args:
        error_type: (string) error type - Must be one of the error keys defined in the errmsg
            dictionary in gnnconst.py
    Returns:
        An error outcome message with a user friendly error message
    """
    if error_type in errmsg.keys():
        error_text = errmsg[error_type][LNG]
    else:
        error_text = errmsg["unknown"][LNG]

    outcome = {
        "status": "error",
        "message": error_text
    }

    return outcome
