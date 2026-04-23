# ===================  docling  ===================
from docling.document_converter import DocumentConverter

# source = "home/ccw/Documents/code/rnd/flutter-app/claim-agent/playground/test-claims/phone.pdf"
source = "/home/ccw/Documents/code/rnd/flutter-app/claim-agent/playground/chroma/test-claims-handwritten/test_handwritten_medical.jpg"

converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export_to_markdown())


# ===================  markitdown ===================
# from markitdown import MarkItDown
# source = "/home/ccw/Documents/code/rnd/flutter-app/claim-agent/playground/test-claims/phone.pdf"
# source = "/home/ccw/Documents/code/rnd/flutter-app/claim-agent/playground/chroma/test-claims-handwritten/test_handwritten_medical.jpg"

# md = MarkItDown(
#     enable_plugins=True,
# )
# result = md.convert(source)
# print(result.text_content)
