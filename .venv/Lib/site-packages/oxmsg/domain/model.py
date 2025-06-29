"""Domain-model objects.

This module should have no dependencies outside the domain model. In particular, any domain model
object should be importable anywhere in the package without risk of producing an import cycle.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Iterator, Protocol

from oxmsg.util import lazyproperty

# ================================================================================================
# EXCEPTIONS
# ================================================================================================


class UnrecognizedCodePageError(Exception):
    """The specified code-page is not a known Microsoft encoding."""


class UnsupportedEncodingError(Exception):
    """The encoding used for this MSG object has no built-in Python codec."""


# ================================================================================================
# INTERFACES
# ================================================================================================


class PropStorageT(Protocol):
    """The interface required of a storage object to extract the properties of that storage."""

    @lazyproperty
    def properties_stream_bytes(self) -> bytes:
        """Bytes of stream containing properties for the top-level object this storage represents.

        Every storage must have exactly one such stream.
        """
        ...

    def property_stream_bytes(self, pid: int, ptyp: int) -> bytes:
        """The bytes of the stream for variable-length property identified by `pid`."""
        ...


class PropertyT(Protocol):
    """The interface required of a property object, regardless of its type."""

    @property
    def name(self) -> str:
        """The Microsft name for this property, like "PidTagMessageClass"."""
        ...

    @property
    def pid(self) -> int:
        """The property-id (PID) for this property, like 0x3701 for attachment bytes."""
        ...

    @property
    def ptyp(self) -> int:
        """The property-type (PTYP) for this property, like 0x0102 for PtypBinary."""
        ...

    @property
    def ptyp_name(self) -> str:
        """The Microsft name for the type of this property, like "PtypString"."""
        ...

    @lazyproperty
    def value(self) -> bool | bytes | dt.datetime | float | int | str | uuid.UUID:
        """The value of this property, its type depending on the property."""
        ...


class StorageT(Protocol):
    """Interface for a storage object."""

    def iter_attachment_storages(self) -> Iterator[StorageT]:
        """Generate each storage object specific to an attachment to this message."""
        ...

    def iter_recipient_storages(self) -> Iterator[StorageT]:
        """Generate each storage object specific to a recipient of this message."""
        ...

    @lazyproperty
    def name(self) -> str:
        """The last segment of the storage path.

        This is the empty string for the root storage. Other storages are named is clearly specified
        ways depending on the type of top-level object they contain, e.g. attachment, recipient,
        etc.
        """
        ...

    @property
    def path(self) -> str:
        """String identifier for this storage; its location within the OXMSG CFB.

        - Suitable as a unique storage identifier within the MSG file.
        - The path of the root storage is the empty string ("").
        """
        ...

    @lazyproperty
    def properties_stream_bytes(self) -> bytes:
        """Bytes of stream containing properties for the top-level object this storage represents.

        Every storage must have exactly one such stream.
        """
        ...

    def property_stream_bytes(self, pid: int, ptyp: int) -> bytes:
        """The bytes of the stream for variable-length property identified by `pid`."""
        ...


class StreamT(Protocol):
    """Interface for a stream object."""

    @lazyproperty
    def name(self) -> str:
        """The last segment of the stream path.

        Each stream contains the bytes for a property. The form of a stream name is clearly
        specified in the OXMSG standard and includes both the property-id (PID) and property
        data-type (PTYP) for the property data it contains.
        """
        ...

    @property
    def path(self) -> str:
        """String identifier for this storage; its location within the OXMSG CFB.

        Suitable as a unique storage identifier within the MSG file.
        """
        ...

    @property
    def bytes_(self) -> bytes:
        """The bytes contained in this stream."""
        ...


# ================================================================================================
# MASKS
# ================================================================================================

# -- AF = attachment flag maybe? Means attachment is embedding, i.e. referenced "by value" --
AF_BY_VALUE = 0x00000001

STORE_UNICODE_OK = 0x00040000


# ================================================================================================
# PROPERTY STREAM HEADER OFFSETS
# ================================================================================================

# -- The starting location for the 16-byte property segments within the property stream varies
# -- depending on the storage type.

ATTACH_HDR_OFFSET = 8
MSG_HDR_OFFSET = 32
RECIP_HDR_OFFSET = 8
