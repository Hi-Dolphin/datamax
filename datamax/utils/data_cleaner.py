import os
import re
import sys
from collections import Counter
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    # Save the original standard output stream
    original_stdout = sys.stdout
    # Redirect standard output to an empty device ('nul' on Windows, '/dev/null' on Unix/Linux/MacOS)
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        try:
            yield
        finally:
            # Restore the original standard output stream
            sys.stdout = original_stdout


with suppress_stdout():
    import jionlp as jio


class AbnormalCleaner:
    def __init__(self, parsed_data):
        self.parsed_data = parsed_data

    def extract_references(self) -> str:
        """
        Extract reference entries and assign to self.parsed_data
        (Original text will be replaced with extracted references, each item on a separate line)

        Returns:
            str: Extracted reference text (same as self.parsed_data)
        """
        patterns = [
            r"([A-Z][a-z]+(?:, [A-Z](?:\.[a-z]*)?)+(?: et al\.)? $\d{4}$[^\n]+)",  # APA format
            r"($$\d+$$[^\n]+)",  # Numbered references like [1]
            r"(DOI:\s?\S+|https?://\S+)",  # DOI/URL
            r"([A-Z][a-z]+, [A-Z]\.?,? & [A-Z][a-z]+, [A-Z]\. \d{4}[^\n]+)",  # Multi-author APA
        ]
        references = []
        for pattern in patterns:
            try:
                references.extend(re.findall(pattern, self.parsed_data))
            except re.error as e:
                print(f"Regex error {pattern}: {e}")

        # Assign extraction results to parsed_data (each item on a separate line)
        self.parsed_data = "\n".join(
            list(set(references))
        )  # Deduplicate and merge into string
        return self.parsed_data

    # Exception cleaning class
    def remove_abnormal_chars(self):
        """Remove abnormal characters from text"""
        self.parsed_data = jio.remove_exception_char(self.parsed_data)
        return self.parsed_data

    def remove_html_tags(self):
        """Remove HTML tags"""
        self.parsed_data = jio.remove_html_tag(self.parsed_data)
        return self.parsed_data

    def convert_newlines(self):
        """Convert \r to \n and multiple \n to a single \n"""
        self.parsed_data = re.sub(r"\r", "", self.parsed_data)
        self.parsed_data = re.sub(r"\n+", "\n", self.parsed_data)
        return self.parsed_data

    def single_space(self):
        """Convert strings with more than 2 spaces to a single space"""
        self.parsed_data = re.sub(r" {2,}", " ", self.parsed_data)
        return self.parsed_data

    def tabs_to_spaces(self):
        """Convert tab characters to 4 spaces"""
        self.parsed_data = self.parsed_data.replace("\t", "    ")
        return self.parsed_data

    def remove_invisible_chars(self):
        """Remove invisible ASCII characters"""
        self.parsed_data = re.sub(
            r"[\x00-\x09\x0b-\x1f\x7f-\xa0]", "", self.parsed_data
        )
        return self.parsed_data

    def simplify_chinese(self):
        """Convert traditional Chinese characters to simplified Chinese"""
        self.parsed_data = jio.tra2sim(self.parsed_data, mode="word")
        return self.parsed_data

    def nlp_clean(self):
        # jio nlp rough text cleaning
        return jio.clean_text(self.parsed_data)

    def point_conversion(self):
        """Bullet point conversion"""
        self.parsed_data = self.parsed_data.replace("\n• ", "\n- ")
        return self.parsed_data

    def clean_space(self):
        self.parsed_data = self.parsed_data.replace(" ", "")
        return self.parsed_data

    def clean_tips(self):
        self.parsed_data = self.parsed_data.replace(
            "EvaluationWarning:ThedocumentwascreatedwithSpire.DocforPython.", ""
        )
        return self.parsed_data

    def markdown_format(self):
        pass

    def no_html_clean(self):
        """Perform cleaning without executing HTML cleaning"""
        try:
            self.convert_newlines()
            self.single_space()
            self.tabs_to_spaces()
            self.simplify_chinese()

            self.remove_invisible_chars()
            # After cleaning invisible characters, perform another multi-line merge, remove space operation
            self.convert_newlines()

            result = {"text": self.parsed_data}
            return result

        except Exception as e:
            print(f"Error: {e}, line: {e.__traceback__.tb_lineno}")
            return {}

    def to_clean(self):
        """Perform all cleaning operations"""
        try:
            self.point_conversion()
            self.remove_html_tags()
            self.convert_newlines()
            self.single_space()
            self.tabs_to_spaces()
            self.simplify_chinese()

            self.remove_invisible_chars()
            # After cleaning invisible characters, perform another multi-line merge, remove space operation
            self.convert_newlines()
            # self.clean_space()
            self.clean_tips()

            result = {"text": self.parsed_data}
            return result

        except Exception as e:
            print(f"Error: {e}, line: {e.__traceback__.tb_lineno}")
            return {}


class TextFilter:
    def __init__(self, parsed_data):
        self.parsed_data = parsed_data

    def filter_by_word_repetition(self, threshold=0.6):
        """Filter by word repetition rate"""
        if not isinstance(self.parsed_data, str):
            return False

        text = str(self.parsed_data)
        bi_grams = [text[i : i + 2] for i in range(0, len(text) - 1, 2)]
        word_count = len(bi_grams)
        if word_count == 0:
            print("No words found.")
            return False

        word_freq = Counter(bi_grams)
        most_common_word, most_common_count = word_freq.most_common(1)[0]
        repetition_rate = most_common_count / word_count
        print(f"Word repetition rate: {repetition_rate}")

        return repetition_rate <= threshold

    def filter_by_char_count(self, min_chars=30, max_chars=500000):
        """Filter by character count"""
        char_count = len(self.parsed_data)
        if char_count < min_chars or char_count > max_chars:
            return False
        return True

    def filter_by_numeric_content(self, threshold=0.6):
        """Filter by numeric content"""
        text = self.parsed_data
        total_chars = len(text)
        numeric_chars = len(re.findall(r"\d", text))
        if numeric_chars / total_chars > threshold:
            return False
        return True

    def to_filter(self):
        """Perform all filtering operations and filter out texts that do not meet the conditions"""
        if not self.filter_by_word_repetition():
            return {}
        elif not self.filter_by_char_count():
            return {}
        elif not self.filter_by_numeric_content():
            return {}
        else:
            result = {"text": self.parsed_data}
            return result


class PrivacyDesensitization:
    def __init__(self, parsed_data):
        self.parsed_data = parsed_data

    # Privacy data replacement class
    def replace_ip(self, token="COSCO_IP"):
        # Replace IP addresses
        self.parsed_data = jio.replace_ip_address(self.parsed_data, token)
        return self.parsed_data

    def replace_email(self, token="COSCO_EMAIL"):
        # Replace email addresses
        self.parsed_data = jio.replace_email(self.parsed_data, token)
        return self.parsed_data

    def replace_customer_number(self, token="COSCO_NUMBER"):
        # Customer service hotlines are not easy to match and are not considered private data
        self.parsed_data = re.sub(r"\d+-\d+-\d+", token, self.parsed_data)
        return self.parsed_data

    def replace_bank_id(self, token="COSCO_NUMBER"):
        # Match bank card numbers and replace
        BANK_ID_PATTERN = r"\b(?:(?:\d{4}[ -]?){4}\d{3}|(?:\d{4}[ -]?){3}\d{4}|(?:4\d{3}|5[1-5]\d{2}|6[045]\d{2})(?:[ -]?\d{4}){3}|3[47]\d{2}[ -]?\d{6}[ -]?\d{5})\b"

        def luhn_check(card_number):
            digits = [int(d) for d in card_number if d.isdigit()]
            if len(digits) not in (13, 15, 16, 19):
                return False
            checksum = sum(digits[-1::-2])
            checksum += sum(sum(divmod(d * 2, 10)) for d in digits[-2::-2])
            return checksum % 10 == 0

        bank_card_numbers = re.findall(BANK_ID_PATTERN, self.parsed_data)

        for card_number in bank_card_numbers:
            if luhn_check(card_number):
                self.parsed_data = re.sub(card_number, token, self.parsed_data)
        return self.parsed_data

    def replace_phone_number(self, token="COSCO_NUMBER"):
        # Match phone numbers and replace
        self.parsed_data = jio.replace_phone_number(self.parsed_data, token)
        return self.parsed_data

    def replace_qq(self, token="COSCO_NUMBER"):
        # Match QQ numbers and replace
        self.parsed_data = jio.replace_qq(self.parsed_data, token)
        return self.parsed_data

    def replace_id_card(self, token="COSCO_NUMBER"):
        # Match ID card numbers and replace
        self.parsed_data = jio.replace_id_card(self.parsed_data, token)
        return self.parsed_data

    def replace_number(self):
        # Replace all types of numeric private data
        # Bank card
        self.parsed_data = self.replace_bank_id(
            token="BANK_ID"
        )  # nosec B106 - 这是数据脱敏标记，不是密码

        # Landline + mobile phone
        self.parsed_data = jio.replace_phone_number(self.parsed_data, "COSCO_NUMBER")
        # QQ
        self.parsed_data = jio.replace_qq(self.parsed_data, "COSCO_NUMBER")
        # ID card
        self.parsed_data = jio.replace_id_card(self.parsed_data, "COSCO_NUMBER")

        return self.parsed_data

    def to_private(self):
        """Perform all privacy data replacement operations"""
        self.replace_ip()
        self.replace_email()
        self.replace_number()
        result = {"text": self.parsed_data}
        return result
