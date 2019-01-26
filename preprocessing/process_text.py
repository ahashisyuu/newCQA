import unicodedata
import re
import string
import json
import nltk

wn_lemmatizer = nltk.stem.WordNetLemmatizer()


img_pattern = re.compile(u'<img .*?src=".+?".*?>')
html_pattern = re.compile(r"<.*?>", re.S)
http_pattern = re.compile(u"(?isu)(https?\://[a-zA-Z0-9\.\?/&\=\:\-_~]+)")
email_pattern = re.compile(u"[a-zA-Z0-9\._]+@[a-zA-Z0-9\.]+")
time_pattern = re.compile(u"(\d+([\:\-\/]\d+)+)")
number_pattern = re.compile(u"(\d+(\.\d+)*)")
atperson_re = re.compile(r'^@\w+$')
punc_re = r"([%s])+" % string.punctuation


def duplicate(sent, mark):
    mark_re = r"%s( +%s)+" % (mark, mark)
    dup = re.sub(mark_re, r'\1', sent)
    return dup


def replace_img(sent):
    sent = img_pattern.sub(' image ', sent)
    return duplicate(sent, 'image')


def replace_http(sent):
    return duplicate(re.sub(http_pattern, ' website ', sent), 'website')


def replace_email(sent):
    return duplicate(re.sub(email_pattern, ' email ', sent), "email")


def replace_atperson(sent):
    return duplicate(atperson_re.sub(' A_T_P_E_R ', sent), 'A_T_P_E_R')


def replace_time(sent):
    return duplicate(time_pattern.sub(' time ', sent), 'time')


def replace_num(sent):
    return duplicate(number_pattern.sub(' number ', sent), 'number')


def replace_punc(sent):
    return re.sub(punc_re, r"\1", sent)


def run_strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":  # [Mn] Mark, Nonspacing  e.g. accents
            continue
        output.append(char)
    return "".join(output)


def run_split_on_punc(text):
    """Splits punctuation on a piece of text."""
    if text in ['I_M_G', 'H_T_T_P', 'E_M_A_I_L',
                'A_T_P_E_R', 'T_I_M_E', 'N_U_M',
                "'s", "n't", "'m"]:
        return [text]

    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1
    return ["".join(x) for x in output]


def clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def process_xml(text):
    return html_pattern.sub(' ', text)


def process_text(text):
    text = clean_text(text.lower())
    text = process_xml(text)
    text = replace_img(text)
    # text = replace_atperson(text)
    text = replace_http(text)
    text = replace_email(text)
    text = replace_time(text)
    text = replace_num(text)
    text = replace_punc(text)
    tokens = [wn_lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)]

    split_tokens = []
    for token in tokens:
        token = run_strip_accents(token)
        token = run_split_on_punc(token)
        split_tokens.extend(token)

    return " ".join(split_tokens).split()


if __name__ == "__main__":
    print(process_text("'document"))
    test_file = "view.json"
    with open(test_file) as fr:
        test_examples = []
        for line in fr:
            test_examples.append(json.loads(line))

    for i, example in enumerate(test_examples):

        question = example['RelQuestion']
        rel_qs = process_text(question['RelQSubject'])
        rel_qb = process_text(question['RelQBody'])

        rel_c = example['RelComments']
        rel_c_text = []
        for comment in rel_c:
            rel_c_text.append(process_text(comment['RelCText']))

        print('E%d ' % i, rel_qs, rel_qb)

