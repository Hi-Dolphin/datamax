from datamax.utils.data_cleaner import AbnormalCleaner, TextFilter, PrivacyDesensitization
from datamax.utils.env_setup import setup_environment


def clean_original_text(text):
    """
    Clean the original text.

    :param text: The original text to be cleaned.
    :return: The cleaned text.
    """
    abnormal_cleaner = AbnormalCleaner(text)
    text = abnormal_cleaner.to_clean()
    text_filter = TextFilter(text)
    text = text_filter.to_filter()
    return text


def clean_original_privacy_text(text):
    """
    Clean the original text with privacy desensitization.

    :param text: The original text to be cleaned.
    :return: The cleaned text with privacy desensitization applied.
    """
    abnormal_cleaner = AbnormalCleaner(parsed_data={"text": text})
    text = abnormal_cleaner.to_clean()
    text_filter = TextFilter(parsed_data={"text": text})
    text = text_filter.to_filter()
    privacy_desensitization = PrivacyDesensitization(parsed_data={"text": text})
    text = privacy_desensitization.to_private()
    return text