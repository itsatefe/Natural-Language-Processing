import re
from cleantext import clean
import hazm

_HTML = re.compile('<.*?>')
_WEIRD = re.compile('[' +
    '\U0001F600-\U0001F64F' +
    '\U0001F300-\U0001F5FF' +
    '\U0001F680-\U0001F6FF' +
    '\U0001F1E0-\U0001F1FF' +
    '\u2702-\u27B0' +
    '\u24C2-\u1F251' +
    '\U0001f926-\U0001f937' +
    '\U00010000-\U0010ffff' +
    '\u200d\u2640-\u2642\u2600-\u2B55\u23cf\u23e9\u231a\u3030\ufe0f\u2069\u2066\u2068\u2067' +
    ']', flags=re.UNICODE)
_NORMALIZER = hazm.Normalizer()

def clean_html(text: str) -> str:
    return re.sub(_HTML, '', text)

def normalize_text(text: str) -> str:
    text = text.strip()
    text = clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=True,
        no_punct=False,
        replace_with_url='',
        replace_with_email='',
        replace_with_phone_number='',
        replace_with_number='',
        replace_with_digit='0',
        replace_with_currency_symbol='',
    )
    text = clean_html(text)
    text = _NORMALIZER.normalize(text)
    text = _WEIRD.sub('', text)
    text = re.sub('#', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
