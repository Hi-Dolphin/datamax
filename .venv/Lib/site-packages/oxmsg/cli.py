# pyright: reportPrivateUsage=false

"""The command-line interface (CLI) for `python-oxmsg`.

The CLI provides the command `oxmsg`.
"""

from __future__ import annotations

import json
from typing import Iterator, cast

import click
from olefile import OleFileIO

from oxmsg.attachment import Attachment
from oxmsg.domain import constants as c
from oxmsg.domain.encodings import encoding_from_codepage
from oxmsg.message import Message
from oxmsg.properties import Properties
from oxmsg.recipient import Recipient
from oxmsg.storage import Storage

# TODO: add `body` sub-command
# TODO: add `detach` sub-command


@click.group()
def oxmsg():
    """Utility CLI for `python-oxmsg`.

    Provides the subcommands listed below, useful for exploratory or diagnostic purposes.
    """
    pass


@oxmsg.command()
@click.argument("msg_file_path", type=str)
def dump(msg_file_path: str):
    """Write a summary of the MSG file's properties to stdout."""
    msg = Message.load(msg_file_path)
    print(f"{dump_message_properties(msg)}")

    for r in msg.recipients:
        print(f"{dump_recipient_properties(r)}")

    for a in msg.attachments:
        if not a.attached_by_value:
            print(f"attachment {a.file_name} is not embedded in message, file unavailable")
        print(f"{dump_attachment_properties(a)}")


@oxmsg.command()
@click.argument("msg_file_path", type=str)
def storage(msg_file_path: str):
    """Summarize low-level "directories and files" structure of MSG."""

    def iter_storage_dump_lines(storage: Storage, prefix: str = "") -> Iterator[str]:
        yield f"{prefix}{storage.name or 'root'}"
        for stream in storage.streams:
            yield f"{prefix}    {stream.name}"
        for s in storage.storages:
            yield from iter_storage_dump_lines(s, prefix + "    ")

    with OleFileIO(msg_file_path) as ole:
        root_storage = Storage.from_ole(ole)

    print("\n".join(iter_storage_dump_lines(root_storage)))


def dump_message_properties(msg: Message) -> str:
    """A summary of this MS-OXMSG object's top-level properties."""
    string_props_are_unicode = msg.properties.string_props_are_unicode
    str_prop_encoding = msg.properties._str_prop_encoding
    internet_code_page = msg.properties.int_prop_value(0x3FDE)
    internet_encoding = (
        None if internet_code_page is None else encoding_from_codepage(internet_code_page)
    )

    def iter_lines() -> Iterator[str]:
        yield ""
        yield "------------------"
        yield "Message Properties"
        yield "------------------"
        yield ""
        yield "header-properties"
        yield "-----------------"
        yield f"recipient_count:    {msg._header_prop_values[2]}"
        yield ""
        yield "distinguished-properties"
        yield "------------------------"
        yield f"attachment_count:         {msg.attachment_count}"
        yield f"internet_code_page:       {internet_encoding}"
        yield f"message_class:            {msg.message_class}"
        yield f"sender:                   {msg.sender}"
        yield f"sent_date:                {msg.sent_date}"
        yield f"string_props_are_unicode: {string_props_are_unicode}"
        if not string_props_are_unicode:
            yield f"string_props_encoding:    {str_prop_encoding}"
        yield f"subject:                  {repr(msg.subject)}"
        yield f"message_headers:\n{json.dumps(msg.message_headers, indent=4, sort_keys=True)}"
        yield ""
        yield "other properties"
        yield dump_properties(msg.properties)

    return "\n".join(iter_lines())


def dump_attachment_properties(attachment: Attachment) -> str:
    """Report of message properies suitable for writing to the console."""

    def iter_lines() -> Iterator[str]:
        yield ""
        yield "---------------------"
        yield "Attachment Properties"
        yield "---------------------"
        yield ""
        yield "distinguished-properties"
        yield "------------------------"
        yield f"attached_by_value: {attachment.attached_by_value}"
        yield f"file_name:         {attachment.file_name}"
        yield f"last_modified:     {attachment.last_modified}"
        yield f"mime_type:         {attachment.mime_type}"
        yield f"size:              {attachment.size:,}"
        yield ""
        yield "other properties"
        yield dump_properties(attachment.properties)

    return "\n".join(iter_lines())


def dump_recipient_properties(recipient: Recipient) -> str:
    """Report of message properies suitable for writing to the console."""

    def iter_lines() -> Iterator[str]:
        yield ""
        yield "---------------------"
        yield "Recipient Properties"
        yield "---------------------"
        yield ""
        yield "distinguished-properties"
        yield "------------------------"
        yield f"name:          {repr(recipient.name)}"
        yield f"email_address: {recipient.email_address}"
        yield ""
        yield "other properties"
        yield dump_properties(recipient.properties)

    return "\n".join(iter_lines())


def dump_properties(self: Properties) -> str:
    """A summary of these properties suitable for printing to the console."""

    def iter_lines() -> Iterator[str]:
        head_rule = f"{'-'*53}+{'-'*23}+{'-'*70}"
        yield head_rule
        yield "property-id" + " " * 42 + "| type" + " " * 18 + "| value"
        yield head_rule

        for p in self:
            value = p.value
            if p.ptyp in (c.PTYP_STRING, c.PTYP_STRING8):
                value = cast(str, self.str_prop_value(p.pid))
                value = repr(value)[:64] + "..." if len(value) > 64 else repr(value)
            elif p.ptyp == c.PTYP_BINARY and p.pid == c.PID_HTML:
                assert isinstance(value, bytes)
                value = value[:64]
            elif isinstance(value, bytes):
                value = f"{len(value):,} bytes"
            elif p.ptyp == c.PTYP_INTEGER_32:
                assert isinstance(value, int)
                b0 = value & 0xFF
                b1 = (value & 0xFF00) >> 8
                b2 = (value & 0xFF0000) >> 16
                b3 = (value & 0xFF000000) >> 24
                value = f"{b3:02X} {b2:02X} {b1:02X} {b0:02X}"

            yield f"0x{p.pid:04X} - {p.name:<43} | {p.ptyp_name:<21} | {value}"

    return "\n".join(iter_lines())
