"""Provides access to the properties of an attachment in an Outlook MSG file.

These properies include the attached file's original file-name, its last-modified date, size, and
content-type (MIME-type).

```python
>>> from oxmsg import Message

>>> msg = Message.load("message.msg")
>>> msg.attachment_count
1
>>> attachment = msg.attachments[0]
>>> attachment.file_name
'financial-forecast.xlsx'
>>> with open(attachment.file_name, "wb") as f:
...     f.write()
```
"""

from __future__ import annotations

import datetime as dt

from oxmsg.domain import constants as c
from oxmsg.domain import model as m
from oxmsg.properties import Properties
from oxmsg.util import lazyproperty


class Attachment:
    """A file attached to an Outlook email message."""

    def __init__(self, storage: m.StorageT):
        self._storage = storage

    @lazyproperty
    def attached_by_value(self) -> bool:
        """True when the `PidTagAttachDataBinary` property contains the attachment data.

        This is as opposed to "by-reference" where only a path or URL is stored.
        """
        attach_method = self.properties.int_prop_value(c.PID_ATTACH_METHOD)
        assert attach_method is not None
        return bool(attach_method & m.AF_BY_VALUE)

    @lazyproperty
    def file_bytes(self) -> bytes | None:
        """The attachment binary, suitable for saving to a file when detaching."""
        return self.properties.binary_prop_value(c.PID_ATTACH_DATA_BINARY)

    @lazyproperty
    def file_name(self) -> str | None:
        """The full name of this file as it was originally attached.

        Like "FY24-quarterly-projections.xlsx". Does not include a path.
        """
        return self.properties.str_prop_value(c.PID_ATTACH_LONG_FILENAME)

    @lazyproperty
    def last_modified(self) -> dt.datetime | None:
        """Timezone-aware UTC datetime when this attachment was last modified.

        `None` if this property is not present on the attachment.
        """
        return self.properties.date_prop_value(c.PID_LAST_MODIFICATION_TIME)

    @lazyproperty
    def mime_type(self) -> str | None:
        """ISO 8601 str representation of time this attachment was last modified."""
        return self.properties.str_prop_value(c.PID_ATTACH_MIME_TAG) or "application/octet-stream"

    @lazyproperty
    def properties(self) -> Properties:
        """Provides access to the properties of this OXMSG object."""
        return Properties(self._storage, properties_header_offset=m.ATTACH_HDR_OFFSET)

    @lazyproperty
    def size(self) -> int:
        """Length in bytes of this attachment."""
        return len(self.file_bytes) if self.file_bytes else 0
