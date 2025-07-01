import string

import math
import json
import jsonlines
import re

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def read_jsonl(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in jsonlines.Reader(f):
            lines.append(line)
    return lines


def save_jsonl(data, filename, print_log=True):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e, ensure_ascii=False) for e in data]))

    if print_log:
        print('save %d samples to %s' % (len(data), filename))


def extract_pred_option_regex(s, choices=['A', 'B', 'C', 'D', 'E', 'F']):
    s = s.strip().strip(string.punctuation)
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]

    # Find the text after any of the answer prefixes
    for answer_prefix in answer_prefixes:
        prefix_pattern = re.escape(answer_prefix)
        match = re.search(prefix_pattern, s, re.IGNORECASE)
        if match:
            s = s[match.end():].strip()
            break  # Exit the loop once the relevant prefix is found

    # After removing the prefix, continue with the existing logic
    if len(s.split()) > 10 and not re.search("[ABCDEF]", s):
        return ""
    matches = re.search(r'[ABCDEF]', s)
    if matches is None:
        for choice in choices:
            if s.lower() in choice.lower():
                return choice[1]
        return ""
    return matches[0]