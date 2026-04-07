from .backend_base import OCRBackend
from .backends import ChandraOCRBackend, DoclingBackend, LightOnOCRBackend, GLMOCRBackend, MarkerBackend
from .schema import ExtractionManifest, ExtractionResult

BACKEND_REGISTRY: dict[str, type[OCRBackend]] = {
    "lighton": LightOnOCRBackend,
}
if DoclingBackend:
    BACKEND_REGISTRY["docling"] = DoclingBackend
if ChandraOCRBackend:
    BACKEND_REGISTRY["chandra"] = ChandraOCRBackend
if GLMOCRBackend:
    BACKEND_REGISTRY["glm"] = GLMOCRBackend
if MarkerBackend:
    BACKEND_REGISTRY["marker"] = MarkerBackend



def get_ocr_backend(name: str = "marker") -> OCRBackend:
    """Instantiate an OCR backend by name.

    Args:
        name: Key in BACKEND_REGISTRY (default: "marker").

    Returns:
        An instance of the requested OCRBackend.

    Raises:
        ValueError: If the name is not registered.
    """
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown OCR backend '{name}'. "
            f"Available: {list(BACKEND_REGISTRY.keys())}"
        )
    return cls()


class DocProcessor:
    def __init__(self, backend: str | OCRBackend = "marker"):
        """Create a document processor with the given OCR backend.

        Args:
            backend: Either a backend name string (looked up in
                     BACKEND_REGISTRY) or an OCRBackend instance.
        """
        if isinstance(backend, str):
            self.backend = get_ocr_backend(backend)
        else:
            self.backend = backend

    def process_document(self, source_pdf_path: str) -> ExtractionResult:
        """
        Process a PDF and return an ExtractionResult.
        If processing fails, returns an empty result instead of raising an exception. Corrupt document shouldn't crash the entire pipeline.
        """
        try:
            return self.backend.extract(source_pdf_path)
        except Exception as e:
            print(f"Failed to process document {source_pdf_path}: {e}")
            return ExtractionResult(
                source_chunks=[],
                manifest_json=ExtractionManifest(
                    doc_id="failed_doc",
                    source_pdf_path=source_pdf_path,
                    markdown_path="",
                    images=[],
                    tables=[],
                    equations=[],
                    references=[]
                ),
                image_count=0,
                table_count=0,
                equation_count=0,
                chunk_count=0
            )
