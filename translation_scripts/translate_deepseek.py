import os
import pandas as pd
from openai import OpenAI

# Load Excel file
file_path = "Alice_Alignment.xls"

df = pd.read_excel(file_path)

sentences = df.iloc[:, 0].dropna().astype(str).tolist()

client = OpenAI(
    base_url="https://api.deepseek.com"
)

MODEL = "deepseek-chat"


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


translations = []

for i in range(len(sentences)):

    context = build_context(sentences, i, max_chars=1000)

    target_sentence = sentences[i]
    full_block = "\n".join(context)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a professional English-to-Turkish literary translator."
            },
            {
                "role": "user",
                "content": f"""
Translate the TARGET sentence from English to Turkish.

Use surrounding context for meaning only.

Rules:
- Output ONLY the translated TARGET sentence
- Do NOT translate the full context
- Do NOT add explanations
- Keep punctuation consistent

TARGET:
{target_sentence}

CONTEXT:
{full_block}
""".strip()
            }
        ],
        temperature=0
    )

    translated = response.choices[0].message.content.strip()

    translations.append(translated)

    print(target_sentence, "->", translated)


df = pd.DataFrame({
    "original": sentences,
    "translated_target": translations
})

df.to_excel("deepseek_translations.xlsx", index=False)

print("Saved to deepseek_translations.xlsx")
