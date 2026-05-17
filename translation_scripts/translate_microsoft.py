import os
import re
import requests
import pandas as pd
import time

AZURE_KEY = ""
AZURE_ENDPOINT = "https://api.cognitive.microsofttranslator.com/"
AZURE_REGION = "francecentral"

if not AZURE_KEY:
    raise ValueError("AZURE_TRANSLATOR_KEY is not set")

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
        r"\[+\s*TGT_START\s*\]+(.*?)\[+\s*TGT_END\s*\]+",
        re.DOTALL | re.IGNORECASE
    )

    match = pattern.search(text)
    if match:
        return match.group(1).strip()

    return text


translations = []
full_translations = []

url = f"{AZURE_ENDPOINT}/translate?api-version=3.0&from=en&to=tr"

headers = {
    "Ocp-Apim-Subscription-Key": AZURE_KEY,
    "Ocp-Apim-Subscription-Region": AZURE_REGION,
    "Content-type": "application/json"
}

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

    body = [{"text": context_text}]

    response = requests.post(url, headers=headers, json=body)

    if response.status_code == 429:
        print("Rate limited, sleeping 2 seconds and retrying...")
        time.sleep(2)
        response = requests.post(url, headers=headers, json=body)

    result = response.json()

    if response.status_code != 200:
        print("FAILED SENTENCE:", sentences[i])
        print("STATUS:", response.status_code)
        print("RESULT:", result)
        translations.append(None)
        full_translations.append(None)
        continue

    translated_text = result[0]["translations"][0]["text"]

    full_translations.append(translated_text)

    extracted = extract_between_tags(translated_text)
    translations.append(extracted)

    print(sentences[i], "->", extracted)


result_df = pd.DataFrame({
    "original": sentences,
    "translated_target": translations,
    "translated_full": full_translations
})

result_df.to_excel("azure_translations.xlsx", index=False)
