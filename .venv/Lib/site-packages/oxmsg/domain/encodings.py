"""Mapping of Microsoft code-page-identifiers (ints) to Python character-encoding names.

Derived from the table at:
https://learn.microsoft.com/en-us/windows/win32/intl/code-page-identifiers

Microsoft-supported code pages that do not have a built-in Python codec are indicated by the empty
string. This allows a `.get()` on this mapping to distinguish between a codec that is unknown and
one that is recognized but not supported.
"""

from __future__ import annotations

from oxmsg.domain import model as m


def encoding_from_codepage(codepage: int) -> str:
    """Map `codepage` to Python character encoding like "iso-2022-jp".

    - Raises `UnrecognizedCodePageError` when `codepage` is not a known Microsoft character
      codepage.
    - Raises `UnsupportedEncodingError` when `codepage` is recognized but Python has no builtin
      codec for that encoding.
    """
    encoding = _CODE_PAGE_ENCODINGS.get(codepage)

    if encoding is None:
        raise m.UnrecognizedCodePageError(  # pragma: no cover
            "the code-page specified in this message is not a known Microsoft character encoding"
        )

    if encoding == "":
        raise m.UnsupportedEncodingError(  # pragma: no cover
            "the character-encoding used in this message is not supported by Python"
        )

    return encoding


_CODE_PAGE_ENCODINGS = {
    37: "IBM037",  # -- IBM EBCDIC US-Canada --
    437: "IBM437",  # -- OEM United States --
    500: "IBM500",  # -- IBM EBCDIC International --
    708: "ASMO-708",  # -- Arabic (ASMO 708) --
    709: "",  # -- Arabic (ASMO-449+, BCON V4) --
    710: "",  # -- Arabic - Transparent Arabic --
    720: "DOS-720",  # -- Arabic (Transparent ASMO) --
    737: "cp737",  # -- OEM Greek (formerly 437G) --
    775: "ibm775",  # -- OEM Baltic --
    850: "ibm850",  # -- OEM Multilingual Latin 1 --
    852: "ibm852",  # -- OEM Latin 2 --
    855: "IBM855",  # -- OEM Cyrillic (primarily Russian) --
    857: "ibm857",  # -- OEM Turkish --
    858: "cp858",  # -- OEM Multilingual Latin 1 + Euro symbol --
    860: "IBM860",  # -- OEM Portuguese --
    861: "ibm861",  # -- OEM Icelandic --
    862: "cp862",  # -- OEM Hebrew --
    863: "IBM863",  # -- OEM French Canadian --
    864: "IBM864",  # -- OEM Arabic --
    865: "IBM865",  # -- OEM Nordic --
    866: "cp866",  # -- OEM Russian --
    869: "ibm869",  # -- OEM Modern Greek --
    870: "cp870",  # -- IBM EBCDIC Multilingual/ROECE (Latin 2) --
    874: "windows-874",  # -- ANSI/OEM Thai (ISO 8859-11) --
    875: "cp875",  # -- IBM EBCDIC Greek Modern --
    932: "shift_jis",  # -- ANSI/OEM Japanese --
    936: "gb2312",  # -- ANSI/OEM Simplified Chinese (PRC, Singapore) --
    949: "ks_c_5601-1987",  # -- ANSI/OEM Korean (Unified Hangul Code) --
    950: "windows-950",  # -- ANSI/OEM Traditional Chinese (Taiwan, Hong Kong SAR, PRC) --
    1026: "IBM1026",  # -- IBM EBCDIC Turkish (Latin 5) --
    1047: "cp1047",  # -- IBM EBCDIC Latin 1/Open System --
    1140: "cp1140",  # -- IBM EBCDIC US-Canada (037 + Euro symbol) --
    1141: "cp1141",  # -- IBM EBCDIC Germany (20273 + Euro symbol) --
    1142: "cp1142",  # -- IBM EBCDIC Denmark-Norway (20277 + Euro symbol) --
    1143: "cp1143",  # -- IBM EBCDIC Finland-Sweden (20278 + Euro symbol) --
    1144: "cp1144",  # -- IBM EBCDIC Italy (20280 + Euro symbol) --
    1145: "cp1145",  # -- IBM EBCDIC Latin America-Spain (20284 + Euro symbol) --
    1146: "cp1146",  # -- IBM EBCDIC United Kingdom (20285 + Euro symbol) --
    1147: "cp1147",  # -- IBM EBCDIC France (20297 + Euro symbol) --
    1148: "cp1148ms",  # -- IBM EBCDIC International (500 + Euro symbol) --
    1149: "cp1149",  # -- IBM EBCDIC Icelandic (20871 + Euro symbol) --
    1200: "utf-16-le",  # -- Unicode UTF-16, little endian byte order (BMP of ISO 10646); --
    1201: "utf-16-be",  # -- Unicode UTF-16, big endian byte order; --
    1250: "windows-1250",  # -- ANSI Central European --
    1251: "windows-1251",  # -- ANSI Cyrillic --
    1252: "windows-1252",  # -- ANSI Latin 1 --
    1253: "windows-1253",  # -- ANSI Greek --
    1254: "windows-1254",  # -- ANSI Turkish --
    1255: "windows-1255",  # -- ANSI Hebrew --
    1256: "windows-1256",  # -- ANSI Arabic --
    1257: "windows-1257",  # -- ANSI Baltic --
    1258: "windows-1258",  # -- ANSI/OEM Vietnamese --
    1361: "Johab",  # -- Korean (Johab) --
    10000: "macintosh",  # -- MAC Roman; Western European (Mac) --
    10001: "x-mac-japanese",  # -- Japanese (Mac) --
    10002: "",  # -- (x-mac-chinesetrad) MAC Traditional Chinese (Big5) --
    10003: "x-mac-korean",  # -- Korean (Mac) --
    10004: "",  # -- (x-mac-arabic)  Arabic (Mac) --
    10005: "",  # -- (x-mac-hebrew) Hebrew (Mac) --
    10006: "x-mac-greek",  # -- Greek (Mac) --
    10007: "x-mac-cyrillic",  # -- Cyrillic (Mac) --
    10008: "",  # -- (x-mac-chinesesimp) MAC Simplified Chinese (GB 2312) --
    10010: "",  # -- (x-mac-romanian) Romanian (Mac) --
    10017: "",  # -- (x-mac-ukrainian) Ukrainian (Mac) --
    10021: "",  # -- (x-mac-thai) Thai (Mac) --
    10029: "x-mac-ce",  # -- MAC Latin 2; Central European (Mac) --
    10079: "x-mac-icelandic",  # -- Icelandic (Mac) --
    10081: "x-mac-turkish",  # -- Turkish (Mac) --
    10082: "",  # -- (x-mac-croatian) Croatian (Mac) --
    12000: "utf-32",  # -- Unicode UTF-32, little endian byte order --
    12001: "utf-32BE",  # -- Unicode UTF-32, big endian byte order --
    20000: "",  # -- (x-Chinese_CNS) CNS Taiwan --
    20001: "",  # -- (x-cp20001) TCA Taiwan --
    20002: "",  # -- (x_Chinese-Eten) Eten Taiwan --
    20003: "",  # -- (x-cp20003) IBM5550 Taiwan --
    20004: "",  # -- (x-cp20004) TeleText Taiwan --
    20005: "",  # -- (x-cp20005) Wang Taiwan --
    20105: "",  # -- (x-IA5) IA5 (IRV International Alphabet No. 5, 7-bit) --
    20106: "",  # -- (x-IA5-German) IA5 German (7-bit) --
    20107: "",  # -- (x-IA5-Swedish) IA5 Swedish (7-bit) --
    20108: "",  # -- (x-IA5-Norwegian) IA5 Norwegian (7-bit) --
    20127: "us-ascii",  # -- US-ASCII (7-bit) --
    20261: "",  # -- (x-cp20261) T.61 --
    20269: "",  # -- (x-cp20269) ISO 6937 Non-Spacing Accent --
    20273: "IBM273",  # -- IBM EBCDIC Germany --
    20277: "cp277",  # -- IBM EBCDIC Denmark-Norway --
    20278: "cp278",  # -- IBM EBCDIC Finland-Sweden --
    20280: "cp280",  # -- IBM EBCDIC Italy --
    20284: "cp284",  # -- IBM EBCDIC Latin America-Spain --
    20285: "cp285",  # -- IBM EBCDIC United Kingdom --
    20290: "cp290",  # -- IBM EBCDIC Japanese Katakana Extended --
    20297: "cp297",  # -- IBM EBCDIC France --
    20420: "cp420",  # -- IBM EBCDIC Arabic --
    20423: "",  # -- (IBM423) IBM EBCDIC Greek --
    20424: "IBM424",  # -- IBM EBCDIC Hebrew --
    20833: "cp833",  # -- IBM EBCDIC Korean Extended --
    20838: "cp838",  # -- IBM EBCDIC Thai --
    20866: "koi8-r",  # -- Russian (KOI8-R); Cyrillic (KOI8-R) --
    20871: "cp871",  # -- IBM EBCDIC Icelandic --
    20880: "",  # -- (IBM880) IBM EBCDIC Cyrillic Russian --
    20905: "IBM905",  # -- IBM EBCDIC Turkish --
    20924: "",  # -- (IBM00924) IBM EBCDIC Latin 1/Open System (1047 + Euro symbol) --
    20932: "EUC-JP",  # -- Japanese (JIS 0208-1990 and 0212-1990) --
    20936: "",  # -- (x-cp20936) Simplified Chinese (GB2312) --
    20949: "",  # -- (x-cp20949) Korean Wansung --
    21025: "cp1025",  # -- IBM EBCDIC Cyrillic Serbian-Bulgarian --
    21027: "",  # -- (deprecated) --
    21866: "koi8-u",  # -- Ukrainian (KOI8-U); Cyrillic (KOI8-U) --
    28591: "iso-8859-1",  # -- ISO 8859-1 Latin 1; Western European (ISO) --
    28592: "iso-8859-2",  # -- ISO 8859-2 Central European; Central European (ISO) --
    28593: "iso-8859-3",  # -- ISO 8859-3 Latin 3 --
    28594: "iso-8859-4",  # -- ISO 8859-4 Baltic --
    28595: "iso-8859-5",  # -- ISO 8859-5 Cyrillic --
    28596: "iso-8859-6",  # -- ISO 8859-6 Arabic --
    28597: "iso-8859-7",  # -- ISO 8859-7 Greek --
    28598: "iso-8859-8",  # -- ISO 8859-8 Hebrew --
    28599: "iso-8859-9",  # -- ISO 8859-9 Turkish --
    28603: "iso-8859-13",  # -- ISO 8859-13 Estonian --
    28605: "iso-8859-15",  # -- ISO 8859-15 Latin 9 --
    29001: "",  # -- (x-Europa) Europa 3 --
    38598: "",  # -- (iso-8859-8-i) ISO 8859-8 Hebrew --
    50220: "iso-2022-jp",  # -- ISO 2022 Japanese with no halfwidth Katakana --
    50221: "csISO2022JP",  # -- ISO 2022 Japanese with halfwidth Katakana --
    50222: "iso-2022-jp",  # -- ISO 2022 Japanese JIS X 0201-1989 --
    50225: "iso-2022-kr",  # -- ISO 2022 Korean --
    50227: "",  # -- (x-cp50227) ISO 2022 Simplified Chinese --
    50229: "",  # -- ISO 2022 Traditional Chinese --
    50930: "",  # -- EBCDIC Japanese (Katakana) Extended --
    50931: "",  # -- EBCDIC US-Canada and Japanese --
    50933: "",  # -- EBCDIC Korean Extended and Korean --
    50935: "",  # -- EBCDIC Simplified Chinese Extended and Simplified Chinese --
    50936: "",  # -- EBCDIC Simplified Chinese --
    50937: "",  # -- EBCDIC US-Canada and Traditional Chinese --
    50939: "",  # -- EBCDIC Japanese (Latin) Extended and Japanese --
    51932: "euc-jp",  # -- EUC Japanese --
    51936: "EUC-CN",  # -- EUC Simplified Chinese --
    51949: "euc-kr",  # -- EUC Korean --
    51950: "",  # -- EUC Traditional Chinese --
    52936: "hz-gb-2312",  # -- HZ-GB2312 Simplified Chinese --
    54936: "GB18030",  # -- Windows XP and later: GB18030 Simplified Chinese (4 byte) --
    57002: "",  # -- (x-iscii-de) ISCII Devanagari --
    57003: "",  # -- (x-iscii-be) ISCII Bangla --
    57004: "",  # -- (x-iscii-ta) ISCII Tamil --
    57005: "",  # -- (x-iscii-te) ISCII Telugu --
    57006: "",  # -- (x-iscii-as) ISCII Assamese --
    57007: "",  # -- (x-iscii-or) ISCII Odia --
    57008: "",  # -- (x-iscii-ka) ISCII Kannada --
    57009: "",  # -- (x-iscii-ma) ISCII Malayalam --
    57010: "",  # -- (x-iscii-gu) ISCII Gujarati --
    57011: "",  # -- (x-iscii-pa) ISCII Punjabi --
    65000: "utf-7",  # -- Unicode (UTF-7) --
    65001: "utf-8",  # -- Unicode (UTF-8) --
}
