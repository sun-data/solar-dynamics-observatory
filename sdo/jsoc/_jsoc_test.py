import pytest
import sdo


@pytest.mark.parametrize(
    argnames="email",
    argvalues=[
        "sample@example.com",
    ],
)
def test_set_get_delete_email(email: str):

    sdo.jsoc.delete_email()

    with pytest.raises(ValueError):
        sdo.jsoc.get_email()

    sdo.jsoc.set_email(email)

    assert sdo.jsoc.get_email() == email

    sdo.jsoc.delete_email()
