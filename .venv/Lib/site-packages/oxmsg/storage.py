"""A "folder" in Microsoft Compound File Binary (CFB) format.

The CFB/OLE file format encloses a filesystem, to a first appoximation, much like a Zip
archive does. In this format, a "storage" corresponds to a directory and a "stream"
corresponds to a file. A storage can contain both streams and other storages.

Each MSG file has a "root" storage, represented in this package by `MessageStorage`.
Each attachment and recipient has their own storage in the root, with a pre-defined
name, and there are other top-level objects than can appear in a MSG file that get their
own storage.
"""

from __future__ import annotations

import dataclasses as dc
from typing import Iterator, Mapping

from olefile import OleFileIO
from olefile.olefile import STGTY_STORAGE, STGTY_STREAM, OleDirectoryEntry

from oxmsg.util import lazyproperty


@dc.dataclass
class Storage:
    """Container for streams and sub-storages."""

    path: str
    streams: tuple[Stream, ...]
    storages: tuple[Storage, ...]

    def __repr__(self) -> str:
        return (
            f"Storage(path={repr(self.path)}, {len(self.streams)} streams,"
            f" {len(self.storages)} storages)"
        )

    @classmethod
    def from_ole(
        cls, ole: OleFileIO, node: OleDirectoryEntry | None = None, prefix: str = ""
    ) -> Storage:
        """Return a Storage loaded from `node` and containing its streams and sub-storages."""
        # -- initial call is `.from_ole(ole)`; other args are only specified on recursion --
        node = node if node else ole.root

        def _iter_streams(ole: OleFileIO, node: OleDirectoryEntry, prefix: str) -> Iterator[Stream]:
            """Generate `Stream` object for each stream in `nodes`."""
            for stream_node in (k for k in node.kids if k.entry_type == STGTY_STREAM):
                path = f"{prefix}/{stream_node.name}" if prefix else stream_node.name
                with ole.openstream(path) as f:
                    bytes_ = f.read()
                yield Stream(path, bytes_)

        streams = tuple(_iter_streams(ole, node, prefix))
        sub_storages = tuple(
            cls.from_ole(ole, k, f"{prefix}/{k.name}" if prefix else k.name)
            for k in node.kids
            if k.entry_type == STGTY_STORAGE
        )
        return cls(path=prefix, streams=streams, storages=sub_storages)

    def iter_attachment_storages(self) -> Iterator[Storage]:
        """Generate storage object specific to each attachment in this message."""
        for s in self.storages:
            if s.name.startswith("__attach_version1.0_#"):
                yield s

    def iter_recipient_storages(self) -> Iterator[Storage]:
        """Generate storage object specific to each recipent in this message."""
        for s in self.storages:
            if s.name.startswith("__recip_version1.0_#"):
                yield s

    @lazyproperty
    def name(self) -> str:
        """The "directory-name" of this storage, with no path-prefix."""
        return self.path.split("/")[-1]

    @lazyproperty
    def properties_stream_bytes(self) -> bytes:
        """The bytes of the one-and-only-one properties stream in this storage."""
        # -- every storage mush have a properties stream --
        return self._streams_by_name["__properties_version1.0"].bytes_

    def property_stream_bytes(self, pid: int, ptyp: int) -> bytes:
        """Read variable-length property bytes from the stream it's stored in."""
        # -- This method should not be called unless there is an entry for this property in the
        # -- properties stream. If the property exists but its stream does not, that's an
        # -- exception, not an expected occurence.
        return self._streams_by_name[f"__substg1.0_{pid:04X}{ptyp:04X}"].bytes_

    @lazyproperty
    def _streams_by_name(self) -> Mapping[str, Stream]:
        """dict semantics on streams of this storage."""
        return {s.name: s for s in self.streams}


@dc.dataclass
class Stream:
    """Bytes of a property of a top-level object in an OXMSG file."""

    path: str
    bytes_: bytes

    def __repr__(self) -> str:
        return f"Stream(path={repr(self.path)}, {len(self.bytes_):,} bytes)"

    @lazyproperty
    def name(self) -> str:
        """The "filename" of this stream, with no path-prefix."""
        return self.path.split("/")[-1]
