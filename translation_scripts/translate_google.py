import requests
import os

url = "https://translation.googleapis.com/language/translate/v2"

START = "<<<START>>>"
END = "<<<END>>>"

TR_START = "<<<BAŞLANGIÇ>>>"
TR_END = "<<<SON>>>"

def build_context(sentences, idx, max_chars=1000):
    center = sentences[idx]

    left = idx - 1
    right = idx + 1

    context = [center]
    total_len = len(center)

    while True:
        added = False

        if left >= 0 and total_len + len(sentences[left]) <= max_chars:
            context.insert(0, sentences[left])
            total_len += len(sentences[left])
            left -= 1
            added = True

        if right < len(sentences) and total_len + len(sentences[right]) <= max_chars:
            context.append(sentences[right])
            total_len += len(sentences[right])
            right += 1
            added = True

        if not added:
            break

    return context


def extract_between_tags(text):
    text = text.strip()

    replacements = {
        "<<< START >>>": START,
        "<<< END >>>": END,
        "<<< BAŞLANGIÇ >>>": TR_START,
        "<<< SON >>>": TR_END,
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    tag_pairs = [
        (START, END),
        (TR_START, TR_END),
        (START, TR_END),
        (TR_START, END),
    ]

    for start_tag, end_tag in tag_pairs:
        if start_tag in text and end_tag in text:
            return text.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()

    return text


translations = []            # extracted target sentences
full_translations = []       # full translated blocks

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

    params = {
        "q": context_text,
        "source": "en",
        "target": "tr",
        "format": "text",
        "key": API_KEY
    }

    response = requests.post(url, params=params)
    data = response.json()

    translated_text = data["data"]["translations"][0]["translatedText"]

    # store full block
    full_translations.append(translated_text)

    # extract target
    extracted = extract_between_tags(translated_text)
    translations.append(extracted)

    print(sentences[i], "->", extracted)

result_df = pd.DataFrame({
    "original": sentences,
    "translated_target": translations,
    "translated_full": full_translations
})

result_df.to_excel("google_translations.xlsx", index=False)
