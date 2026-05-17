import os
import pandas as pd
import anthropic

client = anthropic.Anthropic()

MODEL = "claude-sonnet-4-6"


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

i = 25
context = build_context(sentences, i, max_chars=1000)

target_sentence = sentences[i]
full_block = "\n".join(context)

prompt = f"""
Translate the TARGET SENTENCE from English to Turkish.

Use the surrounding text ONLY as context.

Rules:
- Output ONLY the translated TARGET SENTENCE.
- Do NOT translate the full context.
- Do NOT add explanations.
- Keep punctuation consistent.

TARGET SENTENCE:
{target_sentence}

CONTEXT:
{full_block}
""".strip()

response = client.messages.create(
    model=MODEL,
    max_tokens=1024,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

translated = response.content[0].text.strip()
translations.append(translated)

print(target_sentence, "->", translated)
