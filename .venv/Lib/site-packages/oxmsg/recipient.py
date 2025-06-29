"""The `Recipient` object provides access to properties for a recipient of the message.

This object is accessed using the `Message.recipients` property:
```python
>>> from oxmsg import Message

>>> msg = Message.load("message.msg")
>>> recipient = msg.recipients[0]
>>> recipient.name
'Jane Doe'
```
"""

from __future__ import annotations

from oxmsg.domain import constants as c
from oxmsg.domain import model as m
from oxmsg.properties import Properties
from oxmsg.util import lazyproperty


class Recipient:
    """A recipient of an Outlook email message."""

    def __init__(self, storage: m.StorageT):
        self._storage = storage

    @lazyproperty
    def email_address(self) -> str:
        """The email address of this recipient."""
        props = self.properties
        # -- Preferred source is the SMTP address property, fall back to PidTagEmailAddress which
        # -- can theoretically be X.400 or Exchange email format.
        return (
            props.str_prop_value(c.PID_SMTP_ADDRESS)
            or props.str_prop_value(c.PID_EMAIL_ADDRESS)
            or ""
        )

    @lazyproperty
    def name(self) -> str:
        """The name of this recipient."""
        return self.properties.str_prop_value(c.PID_DISPLAY_NAME) or ""

    @lazyproperty
    def properties(self) -> Properties:
        """Provides access to the properties of this OXMSG object."""
        return Properties(self._storage, properties_header_offset=m.RECIP_HDR_OFFSET)
