"""Provides access to all properties of a top-level object (message, recipient, attachment).

This object is accessed using the `.properties` property of a top-level object. The properties for
each top-level object are segregated into its own properties object.

Many property-ids (PIDs) and property-types (PTYPs) are available as constants in
`oxml.domain.contants`.

```python
>>> from oxmsg import Message
>>> from oxmsg.domain import constants as c

>>> msg = Message.load("message.msg")
>>> properties = msg.properties
>>> properties.str_prop_value(c.PID_MESSAGE_CLASS).value
'IPM.Note'
```
"""

from __future__ import annotations

import datetime as dt
import itertools
import struct
import types
import uuid
from collections import defaultdict
from typing import Final, Iterator, cast

from oxmsg.domain import constants as c
from oxmsg.domain import encodings
from oxmsg.domain import model as m
from oxmsg.domain import reference as ref
from oxmsg.util import lazyproperty


class Properties:
    """Provides access to properties from an OXMSG storage."""

    def __init__(self, storage: m.PropStorageT, properties_header_offset: int):
        self._storage = storage
        # -- Offset within the properties stream at which 16-byte property segments start. This
        # -- varies between storage types, e.g. root properties and attachment properties, etc.
        self._properties_header_offset = properties_header_offset

    def __iter__(self) -> Iterator[m.PropertyT]:
        return iter(self._property_sequence)

    def binary_prop_value(self, pid: int) -> bytes | None:
        """Retrieve bytes of PtypBinary property identified by `pid`.

        Returns `None` if property is not present in this collection.
        """
        property = self._get_property(pid, c.PTYP_BINARY)

        if property is None:
            return None

        return cast(BinaryProperty, property).value

    @lazyproperty
    def body_encoding(self) -> str:
        """The encoding used for a PidTagBody or PidTagHtml property of PtypString8/Binary.

        Must be cherry-picked because it is required before constructing the properties collection.

        Note when these are PtypString they unconditionally use UTF-16LE.
        """
        # -- Case 1: Use `PID_INTERNET_CODEPAGE` (0x3FDE) when present --
        internet_codepage = self._cherry_pick_int_prop(c.PID_INTERNET_CODEPAGE)
        if internet_codepage is not None:
            return encodings.encoding_from_codepage(internet_codepage.value)

        # -- the fallbacks are the same encoding sources as string properties --
        return self._str_prop_encoding

    def date_prop_value(self, pid: int) -> dt.datetime | None:
        """Read datetime property value from the properties stream.

        - Microseconds are truncated.
        - Returns `None` when no `pid` property is present in properties stream.
        """
        property = self._properties_mapping.get(pid)

        if property is None:
            return None

        return cast(TimeProperty, property).value

    def int_prop_value(self, pid: int) -> int | None:
        """Retrieve int value of PtypInteger32 property identified by `pid`.

        Returns `None` if no `pid` property is present in this collection.
        """
        property = self._properties_mapping.get(pid)

        if property is None:
            return None

        return cast(Int32Property, property).value

    def str_prop_value(self, pid: int) -> str | None:
        """Retrieve str value of PtypString or PtypString8 property identified by `pid`.

        Returns the empty str if property is not present in this collection.
        """
        property = self._get_property(pid, (c.PTYP_STRING, c.PTYP_STRING8))

        if property is None:
            return None

        return cast(StringProperty, property).value

    @lazyproperty
    def string_props_are_unicode(self) -> bool:  # pragma: no cover
        """True indicates PtypString properties in this message are encoded "utf-16-le"."""
        store_support_mask = self.int_prop_value(c.PID_STORE_SUPPORT_MASK)

        if store_support_mask is None:
            return False

        return bool(store_support_mask & m.STORE_UNICODE_OK)

    def _cherry_pick_int_prop(self, pid: int) -> Int32Property | None:
        """Get an Int32 property without triggering broader property load.

        Used to solve chicken-and-egg problem of determining encoding required by string
        properties before atomically loading all properties.
        """
        PID = struct.Struct("<2xH")
        for segment in self._prop_segment_sequence:
            pid_ = PID.unpack(segment[:4])[0]
            if pid_ == pid:
                return Int32Property(segment)
        return None

    def _get_property(self, pid: int, ptyps: int | tuple[int, ...]) -> m.PropertyT | None:
        """Retrieve the first property with `pid` and one of `ptyps`.

        The general expectation is that at most one property with `pid` and one of `ptyps` will be
        present in the collection. In the unusual case there could be more than one this method
        may need to be called once for each possible type to get them all or in a particular order
        of preference.
        """
        acceptable_ptyps = (ptyps,) if isinstance(ptyps, int) else ptyps
        candidate_props = self._properties_by_pid[pid]
        for p in candidate_props:
            if p.ptyp in acceptable_ptyps:
                return p
        return None

    @lazyproperty
    def _str_prop_encoding(self) -> str:
        """The encoding used for non-body properties of PtypString8.

        Must be cherry-picked because it is required before constructing the properties collection.

        Note when PtypString properties are unconditionally encoded with UTF-16LE.
        """
        # -- Case 1: `PID_TAG_MESSAGE_CODEPAGE` (0x3FFD) is present and specifies the int
        # -- code-page used to encode the non-Unicode string properties on the Message object.
        message_codepage = self._cherry_pick_int_prop(c.PID_MESSAGE_CODEPAGE)
        if message_codepage is not None:
            return encodings.encoding_from_codepage(message_codepage.value)

        # - Case 2: not specified one way or another, default to "iso-8859-15" (Latin 9) --
        return "iso-8859-15"

    @lazyproperty
    def _prop_segment_sequence(self) -> tuple[bytes, ...]:
        """16-byte segments comprising property blocks from the attachment properties stream."""
        return tuple(
            segment
            for segment in _batched_bytes(
                self._storage.properties_stream_bytes[self._properties_header_offset :], 16
            )
            # -- drop any trailing short segment, happens sometimes --
            if len(segment) == 16
        )

    @lazyproperty
    def _properties_by_pid(self) -> defaultdict[int, list[m.PropertyT]]:
        """Properties in this collection grouped by property-id (PID).

        Not sure if this solves an actual problem in practice, but it's at least
        theoretically possible that the same PID could appear twice in a property collection
        with different PTYPs.
        """
        properties_by_pid: defaultdict[int, list[m.PropertyT]] = defaultdict(list)
        for p in self._property_sequence:
            properties_by_pid[p.pid].append(p)
        return properties_by_pid

    @lazyproperty
    def _properties_mapping(self) -> types.MappingProxyType[int, m.PropertyT]:
        """The property objects in this collection keyed by pid."""
        return types.MappingProxyType({p.pid: p for p in self._property_sequence})

    @lazyproperty
    def _property_sequence(self) -> tuple[m.PropertyT, ...]:
        """Property object for each property in this collection.

        Properties are in property-id (PID) order.
        """
        PID = struct.Struct("<2xH")
        segments = sorted(self._prop_segment_sequence, key=lambda x: PID.unpack(x[:4])[0])
        return tuple(
            BaseProperty.factory(
                segment=segment,
                storage=self._storage,
                str_prop_encoding=self._str_prop_encoding,
                body_encoding=self.body_encoding,
            )
            for segment in segments
        )


class BaseProperty:
    """Base class for properties, providing common behaviors."""

    PID: Final[struct.Struct] = struct.Struct("<2xH")
    PTYP: Final[struct.Struct] = struct.Struct("<H")

    def __init__(self, segment: bytes):
        self._segment = segment

    @classmethod
    def factory(
        cls, segment: bytes, storage: m.PropStorageT, str_prop_encoding: str, body_encoding: str
    ) -> m.PropertyT:
        """Construct a property object of the appropriate sub-type for `segment`."""
        ptyp = cls.PTYP.unpack(segment[:2])[0]

        if ptyp == c.PTYP_BINARY:
            return BinaryProperty(segment, storage)

        if ptyp == c.PTYP_BOOLEAN:
            return BooleanProperty(segment)

        if ptyp == c.PTYP_FLOATING_64:
            return Float64Property(segment)

        if ptyp == c.PTYP_GUID:
            return GuidProperty(segment, storage)

        if ptyp == c.PTYP_INTEGER_16:
            return Int16Property(segment)

        if ptyp == c.PTYP_INTEGER_32:
            return Int32Property(segment)

        if ptyp == c.PTYP_STRING:
            return StringProperty(segment, storage)

        if ptyp == c.PTYP_STRING8:
            return String8Property(
                segment=segment,
                storage=storage,
                str_prop_encoding=str_prop_encoding,
                body_encoding=body_encoding,
            )

        if ptyp == c.PTYP_TIME:
            return TimeProperty(segment)

        # -- default to Int32 --
        return Int32Property(segment)

    @property
    def name(self) -> str:
        """The Microsft name for this property, like "PidTagMessageClass"."""
        prop_desc = ref.property_descriptors.get(self.pid)
        return prop_desc.ms_name if prop_desc is not None else "not recorded in model"

    @property
    def pid(self) -> int:
        """The property-id (PID) for this property, like 0x3701 for attachment bytes."""
        return self.PID.unpack(self._segment[:4])[0]

    @property
    def ptyp(self) -> int:
        """The property-type (PTYP) for this property, like 0x0102 for PtypBinary."""
        return self.PTYP.unpack(self._segment[:2])[0]

    @property
    def ptyp_name(self) -> str:
        """The Microsft name for the type of this property, like "PtypString"."""
        prop_type_desc = ref.property_type_descriptors.get(self.ptyp)
        return (
            prop_type_desc.ms_name if prop_type_desc else f"{self.ptyp:04X} not recorded in model"
        )

    @lazyproperty
    def _payload(self) -> bytes:
        """The latter 8 bytes of the property segment, where the property value is stored."""
        return self._segment[8:]


class BinaryProperty(BaseProperty):
    """Property for PtypBinary OLE properties."""

    def __init__(self, segment: bytes, storage: m.PropStorageT):
        super().__init__(segment)
        self._storage = storage

    @lazyproperty
    def value(self) -> bytes:
        """The bytes of this binary property."""
        return self._storage.property_stream_bytes(self.pid, self.ptyp)


class BooleanProperty(BaseProperty):
    """Property for PtypBoolean OLE properties."""

    SIGNED_CHAR: Final[struct.Struct] = struct.Struct("<b")

    @lazyproperty
    def value(self) -> bool:
        """The boolean value of this property."""
        return self.SIGNED_CHAR.unpack(self._payload[:1])[0] != 0


class Float64Property(BaseProperty):
    """Property for PtypFloating64 OLE properties."""

    FLOAT64: Final[struct.Struct] = struct.Struct("<d")

    @lazyproperty
    def value(self) -> float:
        """The 64-bit floating-point value of this property."""
        return self.FLOAT64.unpack(self._payload)[0]


class GuidProperty(BaseProperty):
    """Property for PtypGuid OLE properties."""

    GUID: Final[struct.Struct] = struct.Struct("<IHH8s")

    def __init__(self, segment: bytes, storage: m.PropStorageT):
        super().__init__(segment)
        self._storage = storage

    def __str__(self) -> str:
        """Hex str representation of this UUID like '9d947746-9662-40a8-a526-abd4faec9737'."""
        return str(self.value)

    @lazyproperty
    def value(self) -> uuid.UUID:
        """The value of this property as a uuid.UUID object.

        The `str` value of this object is the standard-form string for the UUID, like:
        '9d947746-9662-40a8-a526-abd4faec9737'.
        """
        # -- In the OXMSG format, a GUID (aka. UUID) is stored as four distinct fields, each in
        # -- little-endian form. Luckily Python's uuid built-in can parse this directly.
        return uuid.UUID(bytes_le=self._storage.property_stream_bytes(self.pid, self.ptyp)[:16])


class Int16Property(BaseProperty):
    """Property for PtypInteger16 OLE properties."""

    INT16: Final[struct.Struct] = struct.Struct("<H")

    @lazyproperty
    def value(self) -> int:
        """The integer value of this property."""
        return self.INT16.unpack(self._payload[:2])[0]


class Int32Property(BaseProperty):
    """Property for PtypInteger32 OLE properties."""

    INT32: Final[struct.Struct] = struct.Struct("<I")

    @lazyproperty
    def value(self) -> int:
        """The integer value of this property."""
        return self.INT32.unpack(self._payload[:4])[0]


class StringProperty(BaseProperty):
    """Property for PtypString OLE properties."""

    def __init__(self, segment: bytes, storage: m.PropStorageT):
        super().__init__(segment)
        self._storage = storage

    @lazyproperty
    def value(self) -> str:
        """The decoded str from this string property."""
        return self._storage.property_stream_bytes(self.pid, self.ptyp).decode("utf-16-le")


class String8Property(BaseProperty):
    """Property for PtypString8 (8-bit characters, not Unicode) OLE properties."""

    def __init__(
        self, segment: bytes, storage: m.PropStorageT, str_prop_encoding: str, body_encoding: str
    ):
        super().__init__(segment)
        self._storage = storage
        self._str_prop_encoding = str_prop_encoding
        self._body_encoding = body_encoding

    @lazyproperty
    def value(self) -> str:
        """The encoded bytes of this string property.

        The caller is responsible for determining the encoding and applying it to get a str value.
        """
        return self._storage.property_stream_bytes(self.pid, self.ptyp).decode(
            self._body_encoding if self.pid == c.PID_BODY else self._str_prop_encoding
        )


class TimeProperty(BaseProperty):
    """Property for PtypTime OLE properties."""

    TIME: Final[struct.Struct] = struct.Struct("<Q")

    @lazyproperty
    def value(self) -> dt.datetime:
        """The value of this property as a timezone-aware `datetime`."""
        hundred_nanosecond_intervals_since_epoch = self.TIME.unpack(self._payload)[0]
        epoch = dt.datetime(1601, 1, 1, tzinfo=dt.timezone.utc)
        seconds_since_epoch = hundred_nanosecond_intervals_since_epoch // 1e7
        return epoch + dt.timedelta(seconds=seconds_since_epoch)


def _batched_bytes(block: bytes, n: int) -> Iterator[bytes]:
    """Batch bytes from `block` into segments of `n` bytes each.

    Last batch is shorter than `n` when `block` is not evenly divisible by `n`.
    """
    if n < 1:  # pragma: no cover
        raise ValueError("n must be at least one")
    iter_bytes = iter(block)
    while batch := bytes(itertools.islice(iter_bytes, n)):
        yield batch
