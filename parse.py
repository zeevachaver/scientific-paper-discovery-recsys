from pathlib import Path
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent / ".env")
client = OpenAI()


def llm_parse_user_message(user_message: str) -> dict:
    system_prompt = (
        "Extract the target paper and the desired relationship from the user's message. "
        "Return JSON with these keys: target_paper, relationship, and search_terms. "
        "The JSON format output should look like this: "
        "{\n"
        '  "target_paper": "string or null",\n'
        '  "relationship": "critique | extension | application | unrelated | null",\n'
        '  "search_terms": ["string", "string"]\n'
        "}. "
        "relationship must be one of: critique, extension, application, unrelated."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0,
        )
        data = json.loads(response.choices[0].message.content)

        data.setdefault("target_paper", None)
        data.setdefault("relationship", None)
        data.setdefault("search_terms", [])
        return data
    except Exception:
        return {
            "target_paper": None,
            "relationship": None,
            "search_terms": [],
        }

print(llm_parse_user_message("Find papers that critique Attention Is All You Need"))