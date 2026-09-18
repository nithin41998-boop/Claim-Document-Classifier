#!/usr/bin/env python3
"""
classify_claim_document.py
Classifies an insurance-claim attachment (image) into one of Truuth's
established claim-document categories, using a vision-capable LLM
(GPT-5.6 Luna via AWS Bedrock). This is a CLOSED-LIST classifier -- it
picks the single best-matching category from categories.txt (in the
same folder as this script), or "Unclassified" if genuinely nothing
fits.

This is a separate, standalone tool from the identity-document
dimension-check pipeline.

Usage:
    python3 classify_claim_document.py <image_path>

Output: a single line to stdout: the matched category name, or "Unclassified".
"""

import sys
import os
import base64
import json
import cv2
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CATEGORIES_DESC_FILE = os.path.join(SCRIPT_DIR, "categories_with_descriptions.txt")
CATEGORIES_FILE = os.path.join(SCRIPT_DIR, "categories.txt")


def load_categories_with_descriptions():
    if os.path.exists(CATEGORIES_DESC_FILE):
        pairs = []
        with open(CATEGORIES_DESC_FILE, "r") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t", 1)
                name = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                pairs.append((name, desc))
        return pairs
    with open(CATEGORIES_FILE, "r") as f:
        return [(line.strip(), "") for line in f if line.strip()]


def load_categories():
    return [name for name, _ in load_categories_with_descriptions()]



def classify(image_path):
    category_pairs = load_categories_with_descriptions()
    categories = [name for name, _ in category_pairs]
    try:
        from openai import OpenAI
        client = OpenAI()

        with open(image_path, "rb") as f:
            image_bytes = f.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "Unclassified"
        h, w = img.shape[:2]
        scale = min(1.0, 1024 / max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        ok, buf = cv2.imencode(".jpg", img)
        image_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        category_lines = []
        for name, desc in category_pairs:
            if desc:
                category_lines.append(f"- {name}: {desc}")
            else:
                category_lines.append(f"- {name}")
        category_list_text = "\n".join(category_lines)
        prompt_lines = [
            "This image is an attachment from an insurance claim file. Classify it into EXACTLY ONE of the following categories. Each category is listed with a short description -- use the description to help decide, but respond with ONLY the category name (the text before the colon), not the description:",
            "",
            category_list_text,
            "",
            "If the document could reasonably fit more than one category, prefer the category that most specifically describes what the document actually is (e.g. its stated subject, sender type, or purpose) over a category based only on tone, urgency, or writing style.",
            "Read any visible text carefully. If the document has a clear title, header, or status field (e.g. \"CLAIM APPROVAL\", \"Claim Status: APPROVED\"), that should take priority over other details mentioned in the body when choosing the category. Respond with ONLY the exact category text from the list above, no other words, no punctuation, no explanation.",
            "",
            "If none of the categories genuinely fit, respond with exactly: Unclassified"
        ]
        prompt = "\n".join(prompt_lines)
        def call_luna(max_tok):
            resp = client.chat.completions.create(
                model="global.openai.gpt-5.6-luna",
                max_completion_tokens=max_tok,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}
                    ]
                }]
            )
            return resp

        response = call_luna(40)
        content = response.choices[0].message.content

        if content is None:
            # Likely spent the whole budget on invisible reasoning tokens
            # with none left for the visible answer -- retry once with a
            # much larger budget rather than defaulting every call to it.
            response = call_luna(400)
            content = response.choices[0].message.content

        in_tok = response.usage.prompt_tokens
        out_tok = response.usage.completion_tokens
        log_path = os.environ.get("TOKEN_LOG_FILE")
        if log_path:
            with open(log_path, "a") as lf:
                lf.write(f"{in_tok},{out_tok}\n")

        if content is None:
            return "Unclassified"

        label = content.strip()



        if label in categories:
            return label
        for c in categories:
            if c.lower() == label.lower():
                return c
        return "Unclassified"
    except Exception:
        return "Unclassified"


if __name__ == "__main__":
    print(classify(sys.argv[1]))
