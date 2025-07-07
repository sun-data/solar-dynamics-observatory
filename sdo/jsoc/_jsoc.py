import os

__all__ = [
    "set_email",
    "get_email",
    "delete_email",
]


def set_email(email: str) -> None:
    """
    Set the ``JSOC_EMAIL`` environment variable to the given string.

    Parameters
    ----------
    email
        A string to set the JSOC_EMAIL environment variable to.
    """
    os.environ["JSOC_EMAIL"] = email


def get_email() -> str:
    """
    Get the ``JSOC_EMAIL`` environment variable.
    """
    if "JSOC_EMAIL" in os.environ:
        return os.environ["JSOC_EMAIL"]
    else:
        raise ValueError(
            "JSOC email not set, "
            "register your email at http://jsoc.stanford.edu/ajax/register_email.html"
            "and then use `sdo.jsoc.set_email()` to save your email as a local"
            "environment variable."
        )


def delete_email():
    """
    Delete the ``JSOC_EMAIL`` environment variable.
    """
    if "JSOC_EMAIL" in os.environ:
        del os.environ["JSOC_EMAIL"]
