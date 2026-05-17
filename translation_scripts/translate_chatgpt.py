import os
import pandas as pd
from openai import OpenAI

client = OpenAI()


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

    prompt = f"""
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

    response = client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )

    translated_sentence = response.output_text.strip()
    translations.append(translated_sentence)

    print(target_sentence, "->", translated_sentence)

result_df = pd.DataFrame({
    "original": sentences,
    "translated_target": translations
})

result_df.to_excel("chatgpt_target_only_translations.xlsx", index=False)
print("Saved to chatgpt_target_only_translations.xlsx")
