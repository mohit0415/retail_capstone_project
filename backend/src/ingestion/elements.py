from dataclasses import dataclass


@dataclass(frozen=True)
class ParseRoute:
    parser: str
    reason: str
    has_tables: bool = False
    has_images: bool = False

    @property
    def is_multimodal(self) -> bool:
        return self.parser == "llamaparse"


@dataclass(frozen=True)
class ParsedDocument:
    text: str
    route: ParseRoute

    @property
    def parser(self) -> str:
        return self.route.parser

    @property
    def parse_reason(self) -> str:
        return self.route.reason


@dataclass(frozen=True)
class ClauseSection:
    heading: str
    clause_number: str
    body: str
    order: int


@dataclass(frozen=True)
class RawTable:
    markdown: str
    table_index: int
    heading: str
    clause_number: str


@dataclass(frozen=True)
class RawImage:
    path: str
    page: int
    image_index: int
    kind: str = "raster"
