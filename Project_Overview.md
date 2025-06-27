This file overviews the steps tp build an AI agent that reads research papers, extracts knowledge, and generates literature reviews or hypotheses using LLMs.

# Parse research papers (PDFs)

Extracting the raw text from the PDFs of research papers using a PDF parsing library, I will be using `PyMuPDF` (also known as `fitz`).<br>
This library allows us to read PDF files and extract text, images, and other content from them. It is efficient and easy to use for parsing PDFs.<br>
Our goal is to convert a research paper into raw text (title, abstract, paragraphs, sections).<br>
More on PyMuPDF: [PyMuPDF Documentation](parseText.md)<br>

Resources:
- [PyMuPDF Installation and Website](https://pymupdf.readthedocs.io/en/latest/installation.html)
- [Youtube Tutorial on PyMuPDF by pyGuru](https://www.youtube.com/playlist?list=PLHlrXTRZkTLBQ7k06CFoP3-gayAmfp-GG)



