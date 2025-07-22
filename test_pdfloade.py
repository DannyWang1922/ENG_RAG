# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.document_loaders.parsers import RapidOCRBlobParser

# loader = PyPDFLoader(
#     "1012.pdf",
#     mode="page",
# 	extract_images=True,
# 	images_parser=RapidOCRBlobParser(),
#     images_inner_format="text",
# )
# docs = loader.load()
# print("LangChain PyPDFLoader 结果:")
# print(docs)
# print(f"文档数量: {len(docs)}")
# for i, doc in enumerate(docs):
#     print(f"文档 {i} 内容长度: {len(doc.page_content)}")
#     print(f"文档 {i} 元数据: {doc.metadata}")


# from PyPDF2 import PdfReader
# from PIL import Image
# import io
# from rapidocr_onnxruntime import RapidOCR

# reader = PdfReader("1012.pdf")
# page = reader.pages[0]
# images = page.images

# ocr = RapidOCR()
# for i, img in enumerate(images):
#     image_bytes = img.data
#     pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     result, _ = ocr(pil_image)
#     print(f"OCR 图像{i} 结果：")
#     for line in result:
#         print(line[1])

from main import OCRPDFLoader
from config import Config

file = "1012.pdf"
loader = OCRPDFLoader(file, ocr_config=Config.OCR_CONFIG)
docs = loader.load()
print(docs)

# 打印每一页的内容
for i, doc in enumerate(docs):
    print(f"\n=== Page {i+1} Content ===")
    print(doc.page_content)
    print(f"=== Page {i+1} Mate Data ===")
    print(doc.metadata)
    print("\n\n\n")