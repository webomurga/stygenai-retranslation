import os
import re
import boto3
import pandas as pd

translator = boto3.client("translate", region_name="eu-central-1")

START = "[[[TGT_START]]]"
END = "[[[TGT_END]]]"


def build_context(sentences, idx, max_chars=1000):
    center = sentences[idx]

    left = idx - 1
    right = idx + 1

    context = [center]
    total_len = len(center)

    while True:
        added = False

        if left >= 0 and total_len + len(sentences[left]) + 1 <= max_chars:
            context.insert(0, sentences[left])
            total_len += len(sentences[left]) + 1
            left -= 1
            added = True

        if right < len(sentences) and total_len + len(sentences[right]) + 1 <= max_chars:
            context.append(sentences[right])
            total_len += len(sentences[right]) + 1
            right += 1
            added = True

        if not added:
            break

    return context


def extract_between_tags(text):
    text = text.strip()

    pattern = re.compile(
        r"\[{2,3}\s*TGT_START\s*\]{2,3}(.*?)\[{2,3}\s*TGT_END\s*\]{2,3}",
        re.DOTALL | re.IGNORECASE
    )

    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    return text


translations = []
full_translations = []

for i in range(len(sentences)):
    context = build_context(sentences, i)

    block_lines = []
    target_inserted = False

    for s in context:
        if not target_inserted and s == sentences[i]:
            block_lines.append("")
            block_lines.append(START)
            block_lines.append(s)
            block_lines.append(END)
            block_lines.append("")
            target_inserted = True
        else:
            block_lines.append(s)

    context_text = "\n".join(block_lines)

    result = translator.translate_text(
        Text=context_text,
        SourceLanguageCode="en",
        TargetLanguageCode="tr"
    )

    translated_text = result["TranslatedText"]

    full_translations.append(translated_text)

    extracted = extract_between_tags(translated_text)
    translations.append(extracted)

    print(sentences[i], "->", extracted)


result_df = pd.DataFrame({
    "original": sentences,
    "translated_target": translations,
    "translated_full": full_translations
})

result_df.to_excel("aws_translations.xlsx", index=False)
