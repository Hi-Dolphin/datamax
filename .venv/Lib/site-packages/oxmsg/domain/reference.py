"""Provides a catalog of OXMSG properties and property types.

This module provides mappings from integer PID and PTYP keys to meaningful names that can also be
used to find more detailed information on web-search.

In addition it is a source for property names the user can select from to reference the message
attibutes they are interested in.

Also used for discovery utilities to characterize a MSG file and its properties.
"""

from __future__ import annotations

import dataclasses as dc
import datetime as dt

from oxmsg.domain import constants as c


@dc.dataclass(frozen=True)
class PropertyDescriptor:
    """Describes an OXMSG property.

    All attributes of an OXMSG object are properties, even though the body of variable-length
    properties like PID_ATTACH_FILENAME are stored in a stream.
    """

    pid: int
    """The property-id. This is a 16-bit integer value specified by the MS-OXMSG standard."""

    ms_name: str
    """The name like "PidTagLastModificationTime" used by Microsoft for this property.

    This is the value to search on for the specification of the property.
    """

    ptyp: int
    """The property type, one of the OXMSG property types."""


@dc.dataclass(frozen=True)
class PropertyTypeDescriptor:
    """Describes an OXMSG property data type."""

    ptyp: int
    """The property-type id, a 16-bit integer value specified by the MS-OXMSG standard."""

    ms_name: str
    """The name like "PtypInteger32" used by Microsoft for this type."""

    python_type: type
    """The Python type a property of this OXMSG type is converted to."""


"""Mapping of property-id codes to property descriptor objects."""
property_descriptors: dict[int, PropertyDescriptor] = {
    c.PID_ACCESS: PropertyDescriptor(c.PID_ACCESS, "PidTagAccess", c.PTYP_INTEGER_32),
    c.PID_ACCESS_LEVEL: PropertyDescriptor(
        c.PID_ACCESS_LEVEL, "PidTagAccessLevel", c.PTYP_INTEGER_32
    ),
    c.PID_ACKNOWLEDGEMENT_MODE: PropertyDescriptor(
        c.PID_ACKNOWLEDGEMENT_MODE, "PidTagAcknowledgementMode", c.PTYP_INTEGER_32
    ),
    c.PID_ADDRESS_BOOK_FOLDER_PATHNAME: PropertyDescriptor(
        c.PID_ADDRESS_BOOK_FOLDER_PATHNAME, "PidTagAddressBookFolderPathname", c.PTYP_STRING
    ),
    c.PID_ADDRESS_BOOK_HOME_MESSAGE_DATABASE: PropertyDescriptor(
        c.PID_ADDRESS_BOOK_HOME_MESSAGE_DATABASE,
        "PidTagAddressBookHomeMessageDatabase",
        c.PTYP_OBJECT,
    ),
    c.PID_ADDRESS_BOOK_IS_MEMBER_OF_DISTRIBUTION_LIST: PropertyDescriptor(
        c.PID_ADDRESS_BOOK_IS_MEMBER_OF_DISTRIBUTION_LIST,
        "PidTagAddressBookIsMemberOfDistributionList",
        c.PTYP_OBJECT,
    ),
    c.PID_ADDRESS_BOOK_MANAGER_DISTINGUISHED_NAME: PropertyDescriptor(
        c.PID_ADDRESS_BOOK_MANAGER_DISTINGUISHED_NAME,
        "PidTagAddressBookManagerDistinguishedName",
        c.PTYP_OBJECT,
    ),
    c.PID_ADDRESS_BOOK_MEMBER: PropertyDescriptor(
        c.PID_ADDRESS_BOOK_MEMBER, "PidTagAddressBookMember", c.PTYP_OBJECT
    ),
    c.PID_ADDRESS_TYPE: PropertyDescriptor(c.PID_ADDRESS_TYPE, "PidTagAddressType", c.PTYP_STRING),
    c.PID_ALTERNATE_RECIPIENT_ALLOWED: PropertyDescriptor(
        c.PID_ALTERNATE_RECIPIENT_ALLOWED, "PidTagAlternateRecipientAllowed", c.PTYP_BOOLEAN
    ),
    c.PID_ATTACHMENT_FLAGS: PropertyDescriptor(
        c.PID_ATTACHMENT_FLAGS, "PidTagAttachmentFlags", c.PTYP_INTEGER_32
    ),
    c.PID_ATTACHMENT_HIDDEN: PropertyDescriptor(
        c.PID_ATTACHMENT_HIDDEN, "PidTagAttachmentHidden", c.PTYP_BOOLEAN
    ),
    c.PID_ATTACHMENT_LINK_ID: PropertyDescriptor(
        c.PID_ATTACHMENT_LINK_ID, "PidTagAttachmentLinkId", c.PTYP_INTEGER_32
    ),
    c.PID_ATTACH_CONTENT_ID: PropertyDescriptor(
        c.PID_ATTACH_CONTENT_ID, "PidTagAttachContentId", c.PTYP_STRING
    ),
    c.PID_ATTACH_DATA_BINARY: PropertyDescriptor(
        c.PID_ATTACH_DATA_BINARY, "PidTagAttachDataBinary", c.PTYP_BINARY
    ),
    c.PID_ATTACH_ENCODING: PropertyDescriptor(
        c.PID_ATTACH_ENCODING, "PidTagAttachEncoding", c.PTYP_BINARY
    ),
    c.PID_ATTACH_EXTENSION: PropertyDescriptor(
        c.PID_ATTACH_EXTENSION, "PidTagAttachExtension", c.PTYP_STRING
    ),
    c.PID_ATTACH_FILENAME: PropertyDescriptor(
        c.PID_ATTACH_FILENAME, "PidTagAttachFilename", c.PTYP_STRING
    ),
    c.PID_ATTACH_FLAGS: PropertyDescriptor(
        c.PID_ATTACH_FLAGS, "PidTagAttachFlags", c.PTYP_INTEGER_32
    ),
    c.PID_ATTACH_LONG_FILENAME: PropertyDescriptor(
        c.PID_ATTACH_LONG_FILENAME, "PidTagAttachLongFilename", c.PTYP_STRING
    ),
    c.PID_ATTACH_METHOD: PropertyDescriptor(
        c.PID_ATTACH_METHOD, "PidTagAttachMethod", c.PTYP_INTEGER_32
    ),
    c.PID_ATTACH_MIME_SEQUENCE: PropertyDescriptor(
        c.PID_ATTACH_MIME_SEQUENCE, "PidTagAttachMimeSequence", c.PTYP_INTEGER_32
    ),
    c.PID_ATTACH_MIME_TAG: PropertyDescriptor(
        c.PID_ATTACH_MIME_TAG, "PidTagAttachMimeTag", c.PTYP_STRING
    ),
    c.PID_ATTACH_NUMBER: PropertyDescriptor(
        c.PID_ATTACH_NUMBER, "PidTagAttachNumber", c.PTYP_INTEGER_32
    ),
    c.PID_ATTACH_RENDERING: PropertyDescriptor(
        c.PID_ATTACH_RENDERING, "PidTagAttachRendering", c.PTYP_BINARY
    ),
    c.PID_ATTACH_RENDERING_POSITION: PropertyDescriptor(
        c.PID_ATTACH_RENDERING_POSITION, "PidTagAttachRenderingPosition", c.PTYP_INTEGER_32
    ),
    c.PID_ATTACH_SIZE: PropertyDescriptor(c.PID_ATTACH_SIZE, "PidTagAttachSize", c.PTYP_INTEGER_32),
    c.PID_ATTACH_TAG: PropertyDescriptor(c.PID_ATTACH_TAG, "PidTagAttachTag", c.PTYP_BINARY),
    c.PID_AUTHORIZING_USERS: PropertyDescriptor(
        c.PID_AUTHORIZING_USERS, "PidTagAuthorizingUsers", c.PTYP_BINARY
    ),
    c.PID_AUTO_FORWARDED: PropertyDescriptor(
        c.PID_AUTO_FORWARDED, "PidTagAutoForwarded", c.PTYP_BOOLEAN
    ),
    c.PID_AUTO_FORWARD_COMMENT: PropertyDescriptor(
        c.PID_AUTO_FORWARD_COMMENT, "PidTagAutoForwardComment", c.PTYP_STRING
    ),
    c.PID_BODY: PropertyDescriptor(c.PID_BODY, "PidTagBody", c.PTYP_STRING),
    c.PID_CHANGE_KEY: PropertyDescriptor(c.PID_CHANGE_KEY, "PidTagChangeKey", c.PTYP_BINARY),
    c.PID_CLIENT_SUBMIT_TIME: PropertyDescriptor(
        c.PID_CLIENT_SUBMIT_TIME, "PidTagClientSubmitTime", c.PTYP_TIME
    ),
    c.PID_CONTENT_CONFIDENTIALITY_ALGORITHM_ID: PropertyDescriptor(
        c.PID_CONTENT_CONFIDENTIALITY_ALGORITHM_ID,
        "PidTagContentConfidentialityAlgorithmId",
        c.PTYP_BINARY,
    ),
    c.PID_CONTENT_CORRELATOR: PropertyDescriptor(
        c.PID_CONTENT_CORRELATOR, "PidTagContentCorrelator", c.PTYP_BINARY
    ),
    c.PID_CONTENT_IDENTIFIER: PropertyDescriptor(
        c.PID_CONTENT_IDENTIFIER, "PidTagContentIdentifier", c.PTYP_STRING
    ),
    c.PID_CONTENT_LENGTH: PropertyDescriptor(
        c.PID_CONTENT_LENGTH, "PidTagContentLength", c.PTYP_INTEGER_32
    ),
    c.PID_CONTENT_RETURN_REQUESTED: PropertyDescriptor(
        c.PID_CONTENT_RETURN_REQUESTED, "PidTagContentReturnRequested", c.PTYP_BOOLEAN
    ),
    c.PID_CONVERSATION_INDEX: PropertyDescriptor(
        c.PID_CONVERSATION_INDEX, "PidTagConversationIndex", c.PTYP_BINARY
    ),
    c.PID_CONVERSATION_INDEX_TRACKING: PropertyDescriptor(
        c.PID_CONVERSATION_INDEX_TRACKING, "PidTagConversationIndexTracking", c.PTYP_BOOLEAN
    ),
    c.PID_CONVERSATION_KEY: PropertyDescriptor(
        c.PID_CONVERSATION_KEY, "PidTagConversationKey", c.PTYP_BINARY
    ),
    c.PID_CONVERSATION_TOPIC: PropertyDescriptor(
        c.PID_CONVERSATION_TOPIC, "PidTagConversationTopic", c.PTYP_STRING
    ),
    c.PID_CONVERSION_EITS: PropertyDescriptor(
        c.PID_CONVERSION_EITS, "PidTagConversionEits", c.PTYP_BINARY
    ),
    c.PID_CONVERSION_WITH_LOSS_PROHIBITED: PropertyDescriptor(
        c.PID_CONVERSION_WITH_LOSS_PROHIBITED, "PidTagConversionWithLossProhibited", c.PTYP_BOOLEAN
    ),
    c.PID_CONVERTED_EITS: PropertyDescriptor(
        c.PID_CONVERTED_EITS, "PidTagConvertedEits", c.PTYP_BINARY
    ),
    c.PID_CREATION_TIME: PropertyDescriptor(c.PID_CREATION_TIME, "PidTagCreationTime", c.PTYP_TIME),
    c.PID_CREATOR_ADDRESS_TYPE: PropertyDescriptor(
        c.PID_CREATOR_ADDRESS_TYPE, "PidTagCreatorAddressType", c.PTYP_STRING
    ),
    c.PID_CREATOR_EMAIL_ADDRESS: PropertyDescriptor(
        c.PID_CREATOR_EMAIL_ADDRESS, "PidTagCreatorEmailAddress", c.PTYP_STRING
    ),
    c.PID_CREATOR_SIMPLE_DISPLAY_NAME: PropertyDescriptor(
        c.PID_CREATOR_SIMPLE_DISPLAY_NAME, "PidTagCreatorSimpleDisplayName", c.PTYP_STRING
    ),
    c.PID_DEFERRED_DELIVERY_TIME: PropertyDescriptor(
        c.PID_DEFERRED_DELIVERY_TIME, "PidTagDeferredDeliveryTime", c.PTYP_TIME
    ),
    c.PID_DELEGATION: PropertyDescriptor(c.PID_DELEGATION, "PidTagDelegation", c.PTYP_BINARY),
    c.PID_DELETE_AFTER_SUBMIT: PropertyDescriptor(
        c.PID_DELETE_AFTER_SUBMIT, "PidTagDeleteAfterSubmit", c.PTYP_BOOLEAN
    ),
    c.PID_DELIVER_TIME: PropertyDescriptor(c.PID_DELIVER_TIME, "PidTagDeliverTime", c.PTYP_TIME),
    c.PID_DISCARD_REASON: PropertyDescriptor(
        c.PID_DISCARD_REASON, "PidTagDiscardReason", c.PTYP_INTEGER_32
    ),
    c.PID_DISCLOSURE_OF_RECIPIENTS: PropertyDescriptor(
        c.PID_DISCLOSURE_OF_RECIPIENTS, "PidTagDisclosureOfRecipients", c.PTYP_BOOLEAN
    ),
    c.PID_DISPLAY_BCC: PropertyDescriptor(c.PID_DISPLAY_BCC, "PidTagDisplayBcc", c.PTYP_STRING),
    c.PID_DISPLAY_CC: PropertyDescriptor(c.PID_DISPLAY_CC, "PidTagDisplayCc", c.PTYP_STRING),
    c.PID_DISPLAY_NAME: PropertyDescriptor(c.PID_DISPLAY_NAME, "PidTagDisplayName", c.PTYP_STRING),
    c.PID_DISPLAY_TO: PropertyDescriptor(c.PID_DISPLAY_TO, "PidTagDisplayTo", c.PTYP_STRING),
    c.PID_DISTRIBUTION_LIST_EXPANSION_HISTORY: PropertyDescriptor(
        c.PID_DISTRIBUTION_LIST_EXPANSION_HISTORY,
        "PidTagDistributionListExpansionHistory",
        c.PTYP_BINARY,
    ),
    c.PID_DISTRIBUTION_LIST_EXPANSION_PROHIBITED: PropertyDescriptor(
        c.PID_DISTRIBUTION_LIST_EXPANSION_PROHIBITED,
        "PidTagDistributionListExpansionProhibited",
        c.PTYP_BOOLEAN,
    ),
    c.PID_EMAIL_ADDRESS: PropertyDescriptor(
        c.PID_EMAIL_ADDRESS, "PidTagEmailAddress", c.PTYP_STRING
    ),
    c.PID_END_DATE: PropertyDescriptor(c.PID_END_DATE, "PidTagEndDate", c.PTYP_TIME),
    c.PID_ENTRY_ID: PropertyDescriptor(c.PID_ENTRY_ID, "PidTagEntryId", c.PTYP_BINARY),
    c.PID_EXCEPTION_END_TIME: PropertyDescriptor(
        c.PID_EXCEPTION_END_TIME, "PidTagExceptionEndTime", c.PTYP_TIME
    ),
    c.PID_EXCEPTION_START_TIME: PropertyDescriptor(
        c.PID_EXCEPTION_START_TIME, "PidTagExceptionStartTime", c.PTYP_TIME
    ),
    c.PID_EXPIRY_TIME: PropertyDescriptor(c.PID_EXPIRY_TIME, "PidTagExpiryTime", c.PTYP_TIME),
    c.PID_FLAG_STATUS: PropertyDescriptor(c.PID_FLAG_STATUS, "PidTagFlagStatus", c.PTYP_INTEGER_32),
    c.PID_HAS_ATTACHMENTS: PropertyDescriptor(
        c.PID_HAS_ATTACHMENTS, "PidTagHasAttachments", c.PTYP_BOOLEAN
    ),
    c.PID_HTML: PropertyDescriptor(c.PID_HTML, "PidTagHtml", c.PTYP_BINARY),
    c.PID_ICON_INDEX: PropertyDescriptor(c.PID_ICON_INDEX, "PidTagIconIndex", c.PTYP_INTEGER_32),
    c.PID_IMPLICIT_CONVERSION_PROHIBITED: PropertyDescriptor(
        c.PID_IMPLICIT_CONVERSION_PROHIBITED, "PidTagImplicitConversionProhibited", c.PTYP_BOOLEAN
    ),
    c.PID_IMPORTANCE: PropertyDescriptor(c.PID_IMPORTANCE, "PidTagImportance", c.PTYP_INTEGER_32),
    c.PID_INITIAL_DETAILS_PANE: PropertyDescriptor(
        c.PID_INITIAL_DETAILS_PANE, "PidTagInitialDetailsPane", c.PTYP_INTEGER_32
    ),
    c.PID_INSTANCE_KEY: PropertyDescriptor(c.PID_INSTANCE_KEY, "PidTagInstanceKey", c.PTYP_BINARY),
    c.PID_INTERNET_CODEPAGE: PropertyDescriptor(
        c.PID_INTERNET_CODEPAGE, "PidTagInternetCodepage", c.PTYP_INTEGER_32
    ),
    c.PID_INTERNET_MESSAGE_ID: PropertyDescriptor(
        c.PID_INTERNET_MESSAGE_ID, "PidTagInternetMessageId", c.PTYP_STRING
    ),
    c.PID_INTERNET_REFERENCES: PropertyDescriptor(
        c.PID_INTERNET_REFERENCES, "PidTagInternetReferences", c.PTYP_STRING
    ),
    c.PID_IN_REPLY_TO_ID: PropertyDescriptor(
        c.PID_IN_REPLY_TO_ID, "PidTagInReplyToId", c.PTYP_STRING
    ),
    c.PID_LAST_MODIFICATION_TIME: PropertyDescriptor(
        c.PID_LAST_MODIFICATION_TIME, "PidTagLastModificationTime", c.PTYP_TIME
    ),
    c.PID_LAST_MODIFIER_NAME: PropertyDescriptor(
        c.PID_LAST_MODIFIER_NAME, "PidTagLastModifierName", c.PTYP_STRING
    ),
    c.PID_LATEST_DELIVERY_TIME: PropertyDescriptor(
        c.PID_LATEST_DELIVERY_TIME, "PidTagLatestDeliveryTime", c.PTYP_TIME
    ),
    c.PID_LID_CONTACT_ITEM_DATA: PropertyDescriptor(
        c.PID_LID_CONTACT_ITEM_DATA, "PidLidContactItemData", c.PTYP_STRING
    ),
    c.PID_MESSAGE_CC_ME: PropertyDescriptor(
        c.PID_MESSAGE_CC_ME, "PidTagMessageCcMe", c.PTYP_BOOLEAN
    ),
    c.PID_MESSAGE_CLASS: PropertyDescriptor(
        c.PID_MESSAGE_CLASS, "PidTagMessageClass", c.PTYP_STRING
    ),
    c.PID_MESSAGE_DELIVERY_ID: PropertyDescriptor(
        c.PID_MESSAGE_DELIVERY_ID, "PidTagMessageDeliveryId", c.PTYP_BINARY
    ),
    c.PID_MESSAGE_DELIVERY_TIME: PropertyDescriptor(
        c.PID_MESSAGE_DELIVERY_TIME, "PidTagMessageDeliveryTime", c.PTYP_TIME
    ),
    c.PID_MESSAGE_FLAGS: PropertyDescriptor(
        c.PID_MESSAGE_FLAGS, "PidTagMessageFlags", c.PTYP_INTEGER_32
    ),
    c.PID_MESSAGE_LOCALE_ID: PropertyDescriptor(
        c.PID_MESSAGE_LOCALE_ID, "PidTagMessageLocaleId", c.PTYP_INTEGER_32
    ),
    c.PID_MESSAGE_RECIPIENT_ME: PropertyDescriptor(
        c.PID_MESSAGE_RECIPIENT_ME, "PidTagMessageRecipientMe", c.PTYP_STRING
    ),
    c.PID_MESSAGE_SECURITY_LABEL: PropertyDescriptor(
        c.PID_MESSAGE_SECURITY_LABEL, "PidTagMessageSecurityLabel", c.PTYP_BINARY
    ),
    c.PID_MESSAGE_SIZE_EXTENDED: PropertyDescriptor(
        c.PID_MESSAGE_SIZE_EXTENDED, "PidTagMessageSizeExtended", c.PTYP_INTEGER_32
    ),
    c.PID_MESSAGE_SUBMISSION_ID: PropertyDescriptor(
        c.PID_MESSAGE_SUBMISSION_ID, "PidTagMessageSubmissionId", c.PTYP_BINARY
    ),
    c.PID_MESSAGE_TO_ME: PropertyDescriptor(
        c.PID_MESSAGE_TO_ME, "PidTagMessageToMe", c.PTYP_BOOLEAN
    ),
    c.PID_NATIVE_BODY: PropertyDescriptor(c.PID_NATIVE_BODY, "PidTagNativeBody", c.PTYP_INTEGER_32),
    c.PID_NORMALIZED_SUBJECT: PropertyDescriptor(
        c.PID_NORMALIZED_SUBJECT, "PidTagNormalizedSubject", c.PTYP_STRING
    ),
    c.PID_NON_RECEIPT_NOTIFICATION_REQUESTED: PropertyDescriptor(
        c.PID_NON_RECEIPT_NOTIFICATION_REQUESTED,
        "PidTagNonReceiptNotificationRequested",
        c.PTYP_BOOLEAN,
    ),
    c.PID_OBJECT_TYPE: PropertyDescriptor(c.PID_OBJECT_TYPE, "PidTagObjectType", c.PTYP_INTEGER_32),
    c.PID_OBSOLETED_MESSAGE_IDS: PropertyDescriptor(
        c.PID_OBSOLETED_MESSAGE_IDS, "PidTagObsoletedMessageIds", c.PTYP_BINARY
    ),
    c.PID_ORIGINALLY_INTENDED_RECIPIENT_NAME: PropertyDescriptor(
        c.PID_ORIGINALLY_INTENDED_RECIPIENT_NAME,
        "PidTagOriginallyIntendedRecipientName",
        c.PTYP_BINARY,
    ),
    c.PID_ORIGINALLY_INTENDED_RECIP_ADDRTYPE: PropertyDescriptor(
        c.PID_ORIGINALLY_INTENDED_RECIP_ADDRTYPE,
        "PidTagOriginallyIntendedRecipAddrtype",
        c.PTYP_STRING,
    ),
    c.PID_ORIGINALLY_INTENDED_RECIP_EMAIL_ADDRESS: PropertyDescriptor(
        c.PID_ORIGINALLY_INTENDED_RECIP_EMAIL_ADDRESS,
        "PidTagOriginallyIntendedRecipEmailAddress",
        c.PTYP_STRING,
    ),
    c.PID_ORIGINAL_AUTHOR_ADDRESS_TYPE: PropertyDescriptor(
        c.PID_ORIGINAL_AUTHOR_ADDRESS_TYPE, "PidTagOriginalAuthorAddressType", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_AUTHOR_EMAIL_ADDRESS: PropertyDescriptor(
        c.PID_ORIGINAL_AUTHOR_EMAIL_ADDRESS, "PidTagOriginalAuthorEmailAddress", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_AUTHOR_ENTRY_ID: PropertyDescriptor(
        c.PID_ORIGINAL_AUTHOR_ENTRY_ID, "PidTagOriginalAuthorEntryId", c.PTYP_BINARY
    ),
    c.PID_ORIGINAL_AUTHOR_NAME: PropertyDescriptor(
        c.PID_ORIGINAL_AUTHOR_NAME, "PidTagOriginalAuthorName", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_DELIVERY_TIME: PropertyDescriptor(
        c.PID_ORIGINAL_DELIVERY_TIME, "PidTagOriginalDeliveryTime", c.PTYP_TIME
    ),
    c.PID_ORIGINAL_DISPLAY_BCC: PropertyDescriptor(
        c.PID_ORIGINAL_DISPLAY_BCC, "PidTagOriginalDisplayBcc", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_DISPLAY_CC: PropertyDescriptor(
        c.PID_ORIGINAL_DISPLAY_CC, "PidTagOriginalDisplayCc", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_DISPLAY_TO: PropertyDescriptor(
        c.PID_ORIGINAL_DISPLAY_TO, "PidTagOriginalDisplayTo", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_EITS: PropertyDescriptor(
        c.PID_ORIGINAL_EITS, "PidTagOriginalEits", c.PTYP_BINARY
    ),
    c.PID_ORIGINAL_MESSAGE_CLASS: PropertyDescriptor(
        c.PID_ORIGINAL_MESSAGE_CLASS, "PidTagOriginalMessageClass", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_MESSAGE_ID: PropertyDescriptor(
        c.PID_ORIGINAL_MESSAGE_ID, "PidTagOriginalMessageId", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_SENDER_ADDRESS_TYPE: PropertyDescriptor(
        c.PID_ORIGINAL_SENDER_ADDRESS_TYPE, "PidTagOriginalSenderAddressType", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_SENDER_EMAIL_ADDRESS: PropertyDescriptor(
        c.PID_ORIGINAL_SENDER_EMAIL_ADDRESS, "PidTagOriginalSenderEmailAddress", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_SENDER_ENTRY_ID: PropertyDescriptor(
        c.PID_ORIGINAL_SENDER_ENTRY_ID, "PidTagOriginalSenderEntryId", c.PTYP_BINARY
    ),
    c.PID_ORIGINAL_SENDER_NAME: PropertyDescriptor(
        c.PID_ORIGINAL_SENDER_NAME, "PidTagOriginalSenderName", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_SENDER_SEARCH_KEY: PropertyDescriptor(
        c.PID_ORIGINAL_SENDER_SEARCH_KEY, "PidTagOriginalSenderSearchKey", c.PTYP_BINARY
    ),
    c.PID_ORIGINAL_SENSITIVITY: PropertyDescriptor(
        c.PID_ORIGINAL_SENSITIVITY, "PidTagOriginalSensitivity", c.PTYP_INTEGER_32
    ),
    c.PID_ORIGINAL_SENT_REPRESENTING_ADDRESS_TYPE: PropertyDescriptor(
        c.PID_ORIGINAL_SENT_REPRESENTING_ADDRESS_TYPE,
        "PidTagOriginalSentRepresentingAddressType",
        c.PTYP_STRING,
    ),
    c.PID_ORIGINAL_SENT_REPRESENTING_EMAIL_ADDRESS: PropertyDescriptor(
        c.PID_ORIGINAL_SENT_REPRESENTING_EMAIL_ADDRESS,
        "PidTagOriginalSentRepresentingEmailAddress",
        c.PTYP_STRING,
    ),
    c.PID_ORIGINAL_SENT_REPRESENTING_ENTRY_ID: PropertyDescriptor(
        c.PID_ORIGINAL_SENT_REPRESENTING_ENTRY_ID,
        "PidTagOriginalSentRepresentingEntryId",
        c.PTYP_BINARY,
    ),
    c.PID_ORIGINAL_SENT_REPRESENTING_NAME: PropertyDescriptor(
        c.PID_ORIGINAL_SENT_REPRESENTING_NAME, "PidTagOriginalSentRepresentingName", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_SENT_REPRESENTING_SEARCH_KEY: PropertyDescriptor(
        c.PID_ORIGINAL_SENT_REPRESENTING_SEARCH_KEY,
        "PidTagOriginalSentRepresentingSearchKey",
        c.PTYP_BINARY,
    ),
    c.PID_ORIGINAL_SUBJECT: PropertyDescriptor(
        c.PID_ORIGINAL_SUBJECT, "PidTagOriginalSubject", c.PTYP_STRING
    ),
    c.PID_ORIGINAL_SUBMIT_TIME: PropertyDescriptor(
        c.PID_ORIGINAL_SUBMIT_TIME, "PidTagOriginalSubmitTime", c.PTYP_TIME
    ),
    c.PID_ORIGINATOR_CERTIFICATE: PropertyDescriptor(
        c.PID_ORIGINATOR_CERTIFICATE, "PidTagOriginatorCertificate", c.PTYP_BINARY
    ),
    c.PID_ORIGINATOR_DELIVERY_REPORT_REQUESTED: PropertyDescriptor(
        c.PID_ORIGINATOR_DELIVERY_REPORT_REQUESTED,
        "PidTagOriginatorDeliveryReportRequested",
        c.PTYP_BOOLEAN,
    ),
    c.PID_ORIGINATOR_RETURN_ADDRESS: PropertyDescriptor(
        c.PID_ORIGINATOR_RETURN_ADDRESS, "PidTagOriginatorReturnAddress", c.PTYP_BINARY
    ),
    c.PID_ORIGIN_CHECK: PropertyDescriptor(c.PID_ORIGIN_CHECK, "PidTagOriginCheck", c.PTYP_BINARY),
    c.PID_OWNER_APPOINTMENT_ID: PropertyDescriptor(
        c.PID_OWNER_APPOINTMENT_ID, "PidTagOwnerAppointmentId", c.PTYP_INTEGER_32
    ),
    c.PID_PARENT_DISPLAY: PropertyDescriptor(
        c.PID_PARENT_DISPLAY, "PidTagParentDisplay", c.PTYP_STRING
    ),
    c.PID_PARENT_KEY: PropertyDescriptor(c.PID_PARENT_KEY, "PidTagParentKey", c.PTYP_BINARY),
    c.PID_PREDECESSOR_CHANGE_LIST: PropertyDescriptor(
        c.PID_PREDECESSOR_CHANGE_LIST, "PidTagPredecessorChangeList", c.PTYP_BINARY
    ),
    c.PID_PRIORITY: PropertyDescriptor(c.PID_PRIORITY, "PidTagPriority", c.PTYP_INTEGER_32),
    c.PID_PROOF_OF_SUBMISSION_REQUESTED: PropertyDescriptor(
        c.PID_PROOF_OF_SUBMISSION_REQUESTED, "PidTagProofOfSubmissionRequested", c.PTYP_BOOLEAN
    ),
    c.PID_READ_RECEIPT_ENTRY_ID: PropertyDescriptor(
        c.PID_READ_RECEIPT_ENTRY_ID, "PidTagReadReceiptEntryId", c.PTYP_BINARY
    ),
    c.PID_READ_RECEIPT_REQUESTED: PropertyDescriptor(
        c.PID_READ_RECEIPT_REQUESTED, "PidTagReadReceiptRequested", c.PTYP_BOOLEAN
    ),
    c.PID_READ_RECEIPT_SEARCH_KEY: PropertyDescriptor(
        c.PID_READ_RECEIPT_SEARCH_KEY, "PidTagReadReceiptSearchKey", c.PTYP_BINARY
    ),
    c.PID_RECEIPT_TIME: PropertyDescriptor(c.PID_RECEIPT_TIME, "PidTagReceiptTime", c.PTYP_TIME),
    c.PID_RECEIVED_BY_ADDRESS_TYPE: PropertyDescriptor(
        c.PID_RECEIVED_BY_ADDRESS_TYPE, "PidTagReceivedByAddressType", c.PTYP_STRING
    ),
    c.PID_RECEIVED_BY_EMAIL_ADDRESS: PropertyDescriptor(
        c.PID_RECEIVED_BY_EMAIL_ADDRESS, "PidTagReceivedByEmailAddress", c.PTYP_STRING
    ),
    c.PID_RECEIVED_BY_ENTRY_ID: PropertyDescriptor(
        c.PID_RECEIVED_BY_ENTRY_ID, "PidTagReceivedByEntryId", c.PTYP_BINARY
    ),
    c.PID_RECEIVED_BY_NAME: PropertyDescriptor(
        c.PID_RECEIVED_BY_NAME, "PidTagReceivedByName", c.PTYP_STRING
    ),
    c.PID_RECEIVED_BY_SEARCH_KEY: PropertyDescriptor(
        c.PID_RECEIVED_BY_SEARCH_KEY, "PidTagReceivedBySearchKey", c.PTYP_BINARY
    ),
    c.PID_RECEIVED_REPRESENTING_ADDRESS_TYPE: PropertyDescriptor(
        c.PID_RECEIVED_REPRESENTING_ADDRESS_TYPE,
        "PidTagReceivedRepresentingAddressType",
        c.PTYP_STRING,
    ),
    c.PID_RECEIVED_REPRESENTING_EMAIL_ADDRESS: PropertyDescriptor(
        c.PID_RECEIVED_REPRESENTING_EMAIL_ADDRESS,
        "PidTagReceivedRepresentingEmailAddress",
        c.PTYP_STRING,
    ),
    c.PID_RECEIVED_REPRESENTING_ENTRY_ID: PropertyDescriptor(
        c.PID_RECEIVED_REPRESENTING_ENTRY_ID, "PidTagReceivedRepresentingEntryId", c.PTYP_BINARY
    ),
    c.PID_RECEIVED_REPRESENTING_NAME: PropertyDescriptor(
        c.PID_RECEIVED_REPRESENTING_NAME, "PidTagReceivedRepresentingName", c.PTYP_STRING
    ),
    c.PID_RECEIVED_REPRESENTING_SEARCH_KEY: PropertyDescriptor(
        c.PID_RECEIVED_REPRESENTING_SEARCH_KEY, "PidTagReceivedRepresentingSearchKey", c.PTYP_BINARY
    ),
    c.PID_RECIPIENT_REASSIGNMENT_PROHIBITED: PropertyDescriptor(
        c.PID_RECIPIENT_REASSIGNMENT_PROHIBITED,
        "PidTagRecipientReassignmentProhibited",
        c.PTYP_BOOLEAN,
    ),
    c.PID_RECIPIENT_TYPE: PropertyDescriptor(
        c.PID_RECIPIENT_TYPE,
        "PidTagRecipientType",
        c.PTYP_INTEGER_32,
    ),
    c.PID_RECORD_KEY: PropertyDescriptor(c.PID_RECORD_KEY, "PidTagRecordKey", c.PTYP_BINARY),
    c.PID_REDIRECTION_HISTORY: PropertyDescriptor(
        c.PID_REDIRECTION_HISTORY, "PidTagRedirectionHistory", c.PTYP_BINARY
    ),
    c.PID_REPLY_RECIPIENT_ENTRIES: PropertyDescriptor(
        c.PID_REPLY_RECIPIENT_ENTRIES, "PidTagReplyRecipientEntries", c.PTYP_BINARY
    ),
    c.PID_REPLY_RECIPIENT_NAMES: PropertyDescriptor(
        c.PID_REPLY_RECIPIENT_NAMES, "PidTagReplyRecipientNames", c.PTYP_STRING
    ),
    c.PID_REPLY_TIME: PropertyDescriptor(c.PID_REPLY_TIME, "PidTagReplyTime", c.PTYP_TIME),
    c.PID_REPORT: PropertyDescriptor(c.PID_REPORT, "PidTagReportTag", c.PTYP_BINARY),
    c.PID_REPORT_DISPOSITION: PropertyDescriptor(
        c.PID_REPORT_DISPOSITION, "PidTagReportDisposition", c.PTYP_STRING
    ),
    c.PID_REPORT_DISPOSITION_MODE: PropertyDescriptor(
        c.PID_REPORT_DISPOSITION_MODE, "PidTagReportDispositionMode", c.PTYP_STRING
    ),
    c.PID_REPORT_ENTRY_ID: PropertyDescriptor(
        c.PID_REPORT_ENTRY_ID, "PidTagReportEntryId", c.PTYP_BINARY
    ),
    c.PID_REPORT_NAME: PropertyDescriptor(c.PID_REPORT_NAME, "PidTagReportName", c.PTYP_STRING),
    c.PID_REPORT_SEARCH_KEY: PropertyDescriptor(
        c.PID_REPORT_SEARCH_KEY, "PidTagReportSearchKey", c.PTYP_BINARY
    ),
    c.PID_REPORT_TIME: PropertyDescriptor(c.PID_REPORT_TIME, "PidTagReportTime", c.PTYP_TIME),
    c.PID_RESPONSE_REQUESTED: PropertyDescriptor(
        c.PID_RESPONSE_REQUESTED, "PidTagResponseRequested", c.PTYP_BOOLEAN
    ),
    c.PID_ROW_ID: PropertyDescriptor(c.PID_ROW_ID, "PidTagRowId", c.PTYP_INTEGER_32),
    c.PID_RTF_COMPRESSED: PropertyDescriptor(
        c.PID_RTF_COMPRESSED, "PidTagRtfCompressed", c.PTYP_BINARY
    ),
    c.PID_RTF_IN_SYNC: PropertyDescriptor(c.PID_RTF_IN_SYNC, "PidTagRtfInSync", c.PTYP_BOOLEAN),
    c.PID_SEARCH_KEY: PropertyDescriptor(c.PID_SEARCH_KEY, "PidTagSearchKey", c.PTYP_BINARY),
    c.PID_SECURITY: PropertyDescriptor(c.PID_SECURITY, "PidTagSecurity", c.PTYP_INTEGER_32),
    c.PID_SENDER_ADDRESS_TYPE: PropertyDescriptor(
        c.PID_SENDER_ADDRESS_TYPE, "PidTagSenderAddressType", c.PTYP_STRING
    ),
    c.PID_SENDER_EMAIL_ADDRESS: PropertyDescriptor(
        c.PID_SENDER_EMAIL_ADDRESS, "PidTagSenderEmailAddress", c.PTYP_STRING
    ),
    c.PID_SENDER_ENTRY_ID: PropertyDescriptor(
        c.PID_SENDER_ENTRY_ID, "PidTagSenderEntryId", c.PTYP_BINARY
    ),
    c.PID_SENDER_NAME: PropertyDescriptor(c.PID_SENDER_NAME, "PidTagSenderName", c.PTYP_STRING),
    c.PID_SENDER_SEARCH_KEY: PropertyDescriptor(
        c.PID_SENDER_SEARCH_KEY, "PidTagSenderSearchKey", c.PTYP_BINARY
    ),
    c.PID_SENDER_SMTP_ADDRESS: PropertyDescriptor(
        c.PID_SENDER_SMTP_ADDRESS, "PidTagSenderSmtpAddress", c.PTYP_STRING
    ),
    c.PID_SENSITIVITY: PropertyDescriptor(
        c.PID_SENSITIVITY, "PidTagSensitivity", c.PTYP_INTEGER_32
    ),
    c.PID_SENT_REPRESENTING_ADDRESS_TYPE: PropertyDescriptor(
        c.PID_SENT_REPRESENTING_ADDRESS_TYPE, "PidTagSentRepresentingAddressType", c.PTYP_STRING
    ),
    c.PID_SENT_REPRESENTING_EMAIL_ADDRESS: PropertyDescriptor(
        c.PID_SENT_REPRESENTING_EMAIL_ADDRESS, "PidTagSentRepresentingEmailAddress", c.PTYP_STRING
    ),
    c.PID_SENT_REPRESENTING_ENTRY_ID: PropertyDescriptor(
        c.PID_SENT_REPRESENTING_ENTRY_ID, "PidTagSentRepresentingEntryId", c.PTYP_BINARY
    ),
    c.PID_SENT_REPRESENTING_NAME: PropertyDescriptor(
        c.PID_SENT_REPRESENTING_NAME, "PidTagSentRepresentingName", c.PTYP_STRING
    ),
    c.PID_SENT_REPRESENTING_SEARCH_KEY: PropertyDescriptor(
        c.PID_SENT_REPRESENTING_SEARCH_KEY, "PidTagSentRepresentingSearchKey", c.PTYP_BINARY
    ),
    c.PID_SMTP_ADDRESS: PropertyDescriptor(c.PID_SMTP_ADDRESS, "PidTagSmtpAddress", c.PTYP_STRING),
    c.PID_START_DATE: PropertyDescriptor(c.PID_START_DATE, "PidTagStartDate", c.PTYP_TIME),
    c.PID_STORE_SUPPORT_MASK: PropertyDescriptor(
        c.PID_STORE_SUPPORT_MASK, "PidTagStoreSupportMask", c.PTYP_INTEGER_32
    ),
    c.PID_STORE_UNICODE_MASK: PropertyDescriptor(
        c.PID_STORE_UNICODE_MASK, "PidTagStoreUnicodeMask", c.PTYP_INTEGER_32
    ),
    c.PID_SUBJECT: PropertyDescriptor(c.PID_SUBJECT, "PidTagSubject", c.PTYP_STRING),
    c.PID_SUBJECT_MESSAGE_ID: PropertyDescriptor(
        c.PID_SUBJECT_MESSAGE_ID, "PidTagSubjectMessageId", c.PTYP_BINARY
    ),
    c.PID_SUBJECT_PREFIX: PropertyDescriptor(
        c.PID_SUBJECT_PREFIX, "PidTagSubjectPrefix", c.PTYP_STRING
    ),
    c.PID_TNEF_CORRELATION_KEY: PropertyDescriptor(
        c.PID_TNEF_CORRELATION_KEY, "PidTagTnefCorrelationKey", c.PTYP_BINARY
    ),
    c.PID_TRANSPORT_MESSAGE_HEADERS: PropertyDescriptor(
        c.PID_TRANSPORT_MESSAGE_HEADERS, "PidTagTransportMessageHeaders", c.PTYP_STRING
    ),
}


"""Mapping of property-type codes to descriptor objects."""
property_type_descriptors: dict[int, PropertyTypeDescriptor] = {
    c.PTYP_BINARY: PropertyTypeDescriptor(c.PTYP_BINARY, "PtypBinary", bytes),
    c.PTYP_BOOLEAN: PropertyTypeDescriptor(c.PTYP_BOOLEAN, "PtypBoolean", bool),
    c.PTYP_FLOATING_64: PropertyTypeDescriptor(c.PTYP_FLOATING_64, "PtypFloating64", float),
    c.PTYP_GUID: PropertyTypeDescriptor(c.PTYP_GUID, "PtypGuid", bytes),
    c.PTYP_INTEGER_16: PropertyTypeDescriptor(c.PTYP_INTEGER_16, "PtypInteger16", int),
    c.PTYP_INTEGER_32: PropertyTypeDescriptor(c.PTYP_INTEGER_32, "PtypInteger32", int),
    c.PTYP_MULTIPLE_INTEGER_32: PropertyTypeDescriptor(
        c.PTYP_MULTIPLE_INTEGER_32, "PtypMultipleInteger32", list[int]
    ),
    c.PTYP_MULTIPLE_STRING: PropertyTypeDescriptor(
        c.PTYP_MULTIPLE_STRING, "PtypMultipleString", list[str]
    ),
    c.PTYP_OBJECT: PropertyTypeDescriptor(c.PTYP_OBJECT, "PtypObject", bytes),
    c.PTYP_STRING: PropertyTypeDescriptor(c.PTYP_STRING, "PtypString", str),
    c.PTYP_STRING8: PropertyTypeDescriptor(c.PTYP_STRING8, "PtypString8", str),
    c.PTYP_TIME: PropertyTypeDescriptor(c.PTYP_TIME, "PtypTime", dt.datetime),
}
