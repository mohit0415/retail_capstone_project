import logging
import shutil
from pathlib import Path

from llama_index.core.schema import TextNode

from configs.settings import settings
from src.ingestion.elements import RawImage
from src.ingestion.metadata.stamping import apply_exclusions, image_node_metadata
from src.ingestion.processors.image_captioning import generate_caption

logger = logging.getLogger(__name__)


def storage_dir() -> Path:
    directory = Path(settings.images_storage_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    return directory


class ImageProcessor:
    def __init__(self):
        self.storage = None

    def _store(self, image: RawImage, source_stem: str) -> str:
        if self.storage is None:
            self.storage = storage_dir()

        slot = "fig" if image.kind == "figure" else "img"

        permanent = self.storage / f"{source_stem}_page{image.page}_{slot}{image.image_index}.png"
        shutil.copy2(image.path, permanent)

        return str(permanent)

    def process(self, images: list[RawImage], document_metadata: dict) -> list[TextNode]:
        source_stem = Path(document_metadata.get("original_file_name", "document")).stem.replace(" ", "_")

        nodes: list[TextNode] = []

        for image in images:
            try:
                caption = generate_caption(image.path)
            except Exception as exc:
                logger.warning("caption generation failed for %s: %s", image.path, exc)
                continue

            if not caption:
                logger.warning("the vision model returned an empty caption for %s", image.path)
                continue

            metadata = image_node_metadata(document_metadata, image, self._store(image, source_stem))

            nodes.append(apply_exclusions(TextNode(text=caption, metadata=metadata)))

        logger.info("built %s image node(s) from %s extracted image(s)", len(nodes), len(images))

        return nodes
