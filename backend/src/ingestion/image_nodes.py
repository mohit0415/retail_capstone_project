import base64
import logging
import shutil
from pathlib import Path

from llama_index.core.schema import TextNode

from configs.settings import settings

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "This image comes from a retail policy or compliance document. Describe what it shows in "
    "two or three sentences. If it is a flowchart, decision tree or process diagram, state the "
    "steps and the decision points in order. If it carries labels, thresholds, dates or role "
    "names, reproduce them exactly. Do not speculate about anything the image does not show."
)

MIN_IMAGE_BYTES = 4096


def _storage_dir() -> Path:
    directory = Path(settings.images_storage_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def extract_images_from_pdf(file_path: str, output_dir: str) -> list[dict]:
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF is not installed; image extraction skipped")
        return []

    extracted: list[dict] = []

    try:
        document = fitz.open(file_path)
    except Exception as exc:
        logger.warning("could not open %s for image extraction: %s", file_path, exc)
        return []

    for page_number in range(len(document)):
        page = document[page_number]

        for image_index, image in enumerate(page.get_images(full=True)):
            xref = image[0]

            try:
                pixmap = fitz.Pixmap(document, xref)

                if pixmap.n - pixmap.alpha >= 4:
                    pixmap = fitz.Pixmap(fitz.csRGB, pixmap)

                data = pixmap.tobytes("png")
            except Exception:
                continue

            if len(data) < MIN_IMAGE_BYTES:
                continue

            target = Path(output_dir) / f"page{page_number + 1}_img{image_index}.png"
            target.write_bytes(data)

            extracted.append(
                {
                    "path": str(target),
                    "page": page_number + 1,
                    "image_index": image_index,
                }
            )

    document.close()

    return extracted


def generate_caption(image_path: str) -> str:
    from openai import AzureOpenAI as AzureOpenAIClient

    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")

    client = AzureOpenAIClient(
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
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            }
        ],
        max_tokens=400,
        temperature=0.0,
    )

    return (response.choices[0].message.content or "").strip()


def build_image_nodes(images: list[dict], base_metadata: dict) -> list[TextNode]:
    if not images:
        return []

    storage = _storage_dir()
    source_stem = Path(base_metadata.get("original_file_name", "document")).stem.replace(" ", "_")

    nodes: list[TextNode] = []

    for image in images:
        try:
            caption = generate_caption(image["path"])
        except Exception as exc:
            logger.warning("caption generation failed for %s: %s", image["path"], exc)
            continue

        if not caption:
            continue

        permanent = storage / f"{source_stem}_page{image['page']}_img{image['image_index']}.png"
        shutil.copy2(image["path"], permanent)

        metadata = {
            **base_metadata,
            "content_type": "image_caption",
            "modality": "diagram",
            "page": image["page"],
            "page_label": str(image["page"]),
            "image_index": image["image_index"],
            "element_label": f"page{image['page']}_img{image['image_index']}",
            "image_path": str(permanent),
        }

        nodes.append(TextNode(text=caption, metadata=metadata))

    return nodes
