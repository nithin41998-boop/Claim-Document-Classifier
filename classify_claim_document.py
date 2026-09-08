#!/usr/bin/env python3
"""
classify_claim_document.py
Classifies an insurance-claim attachment (image) into one of Truuth's
established claim-document categories, using a vision-capable LLM
(Claude Haiku via AWS Bedrock). This is a CLOSED-LIST classifier -- it
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
        import boto3
        session = boto3.Session()
        client = session.client("bedrock-runtime", region_name="ap-southeast-2")

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

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 40,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        }
        response = client.invoke_model(
            modelId="au.anthropic.claude-haiku-4-5-20251001-v1:0",
            body=json.dumps(body)
        )
        result = json.loads(response["body"].read())

        usage = result.get("usage", {})
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        log_path = os.environ.get("TOKEN_LOG_FILE")
        if log_path:
            with open(log_path, "a") as lf:
                lf.write(f"{in_tok},{out_tok}\n")

        label = result["content"][0]["text"].strip()

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
