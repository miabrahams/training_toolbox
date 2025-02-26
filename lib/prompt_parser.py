import re
from collections import namedtuple
from typing import List

re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)

def parse_prompt_attention(text):
    r"""
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text

    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith('\\'):
            res.append([text[1:], 1.0])
        elif text == '(':
            round_brackets.append(len(res))
        elif text == '[':
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ')' and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == ']' and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def strip_prompt_markup(prompt: str) -> str:
    """
    Remove LoRA markup and token weight syntax from a prompt.

    For example:
      "<lora:styles/smooth anime 2 style:0.7>" is removed.
      "(masterpiece, best quality:1.1)" becomes "masterpiece, best quality"
    """
    # Remove LoRA markup (e.g., <lora:...>)
    prompt = re.sub(r'<lora:[^>]+>', '', prompt)
    # Remove weight values inside parentheses: remove colon and weights
    prompt = re.sub(r':\s*[0-9.]+', '', prompt)
    # Optionally remove the remaining parentheses
    prompt = prompt.replace('(', '').replace(')', '')
    return prompt.strip()

special_tags =  ['source_furry', 'source_anime', 'source_cartoon']
special_tags += ['score_9', 'score_8_up', 'score_7_up', 'score_6_up', 'score_5_up', 'score_4_up']
special_tags += ['rating_explicit', 'rating_questionable', 'rating_safe']
special_tags += ['explicit', 'questionable', 'safe']
special_tags += ['illustration', 'digital illustration art', 'official art', 'edit']
special_tags += ['masterpiece', 'best quality', 'absurdres', 'hires', 'hi res', 'very awa', 'very aesthetic', '()',]
special_tags += ['newest', 'year 2022', 'year 2023', 'year 2024', '2022', '2023', '2024']
ignore_tags = ['trmk2', 'csr style', ]

def remove_extra_commas(prompt: str) -> str:
    """
    Removes residual punctuation such as ',,,' and extra spaces.
    """
    # Replace multiple commas with a single comma
    prompt = re.sub(r',\s*,+', ',', prompt)
    # Standardize spaces around commas
    prompt = re.sub(r'\s*,\s*', ', ', prompt)
    # Remove any trailing commas
    prompt = re.sub(r'(,\s*)+$', '', prompt)
    return prompt.strip()

def sort_prompt_tokens(prompt: str) -> str:
    """
    Split the prompt by comma, remove empty tokens, sort them lexicographically,
    and join them back together.
    """
    tokens = [token.strip() for token in prompt.split(',')]
    tokens = [token for token in tokens if token]
    tokens.sort()
    return ', '.join(tokens)

def clean_prompt(prompt: str) -> str:
    # First, strip out markup
    prompt = strip_prompt_markup(prompt)
    prompt = prompt.lower()
    for tag in special_tags:
        prompt = prompt.replace(tag + ', ', '').replace(tag, '')
    prompt = prompt.strip().replace('\n', ' ').replace('\r', ' ')
    # Remove redundant commas/punctuation
    prompt = remove_extra_commas(prompt)
    # Split by commas, sort the tokens, then recombine into a uniform string.
    prompt = sort_prompt_tokens(prompt)
    return prompt