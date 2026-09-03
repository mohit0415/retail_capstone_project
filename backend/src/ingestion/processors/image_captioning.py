import base64
from pathlib import Path

from configs.settings import settings

CAPTION_PROMPT = (
    "This image comes from a retail policy or compliance document. Describe what it shows in "
    "two or three sentences. If it is a flowchart, decision tree or process diagram, state the "
    "steps and the decision points in order. If it carries labels, thresholds, dates or role "
    "names, reproduce them exactly. Do not speculate about anything the image does not show."
)

MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

MAX_CAPTION_TOKENS = 400


def generate_caption(image_path: str) -> str:
    from openai import AzureOpenAI

    path = Path(image_path)
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    mime_type = MIME_TYPES.get(path.suffix.lower(), "image/png")

    client = AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )

    response = client.chat.completions.create(
        model=settings.azure_openai_vision_deployment,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                    },
                ],
            }
        ],
        max_tokens=MAX_CAPTION_TOKENS,
        temperature=0.0,
    )

    return (response.choices[0].message.content or "").strip()
