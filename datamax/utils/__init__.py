# Import from new cleaner module for backward compatibility
from datamax.cleaner import (
    AbnormalCleaner,
    PrivacyDesensitization,
    TextFilter,
)

# Import from new generator module for backward compatibility
from datamax.generator import (
    DomainTree,
    full_qa_labeling_process,
    get_instruction_prompt,
    get_system_prompt_for_answer,
    get_system_prompt_for_domain_tree,
    get_system_prompt_for_match_label,
    get_system_prompt_for_question,
    process_domain_tree,
    process_match_tags,
    process_questions,
    generatr_qa_pairs,
    multimodal_qa_generator,
)

from datamax.utils.env_setup import setup_environment


# Conditionally import the UNO processor
try:
    from datamax.utils.uno_handler import (
        HAS_UNO,
        UnoManager,
        cleanup_uno_manager,
        convert_with_uno,
        get_uno_manager,
        uno_manager_context,
    )
except ImportError:
    HAS_UNO = False
    UnoManager = None
    get_uno_manager = None
    convert_with_uno = None
    cleanup_uno_manager = None
    uno_manager_context = None


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
