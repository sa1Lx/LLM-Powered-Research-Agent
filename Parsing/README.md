# Opening a pdf and getting some details

1. `pip install PyMuPDF` # Install the PyMuPDF library

2. `import fitz` # Import the PyMuPDF library

3. `print(fitz.__doc__)` # Print the documentation of PyMuPDF

4. `pdf = fitz.open(‘a.pdf’)` # Open a PDF file named 'a.pdf'

5. `print(pdf.page_count)` # Print the number of pages in the PDF

6. `print(pdf.metadata)` # Print the metadata of the PDF, namely 
    * `print(pdf.metadata[‘author’])` # Print the author of the PDF

7. `print(pdf.get_toc())` # Print the table of contents of the PDF
    * `print(pdf.get_toc()[0:10])` # Print the first 10 entries of the table of contents


# Loading a page from a PDF

        page = pdf.load_page(pno)  # loads page number 'pno' of the document (0-based)
        page = pdf[pno]  # the short form

Any integer -∞ < pno < page_count is possible here. Negative numbers count backwards from the end, so pdf[-1] is the last page, like with Python sequences.

Some more advanced way would be using the document as an iterator over its pages:

        for page in pdf:
            # do something with 'page'

        # ... or read backwards
        for page in reversed(pdf):
            # do something with 'page'

        # ... or even use 'slicing'
        for page in pdf.pages(start, stop, step):
            # do something with 'page'

## Inspecting the Links

        # get all links on a page
        links = page.get_links()

You can also use an iterator which emits one link at a time:
        for link in page.links():
            # do something with 'link'

## Rendering a Page and Saving it as a file

        pix = page.get_pixmap()  # render page to an image
        pix.save("page-%i.png" % page.number)  # save the image as a PNG file

        # or save it as a JPEG file
        pix.save("page.jpg")

        # or save it as a PDF file
        pix.save("page.pdf")

## Extracting Text and Images

        text = page.get_text(opt)

Use one of the following strings for opt to obtain different formats:

1. “text”: (default) plain text with line breaks. No formatting, no text position details, no images.

2. “blocks”: generate a list of text blocks (= paragraphs).

3. “words”: generate a list of words (strings not containing spaces).

4. “html”: creates a full visual version of the page including any images. This can be displayed with your internet browser.

5. “dict” / “json”: same information level as HTML, but provided as a Python dictionary or resp. JSON string. See TextPage.extractDICT() for details of its structure.

6. “rawdict” / “rawjson”: a super-set of “dict” / “json”. It additionally provides character detail information like XML. See TextPage.extractRAWDICT() for details of its structure.

7. “xhtml”: text information level as the TEXT version but includes images. Can also be displayed by internet browsers.

8. “xml”: contains no images, but full position and font information down to each single text character. Use an XML module to interpret.

## Seaching for Text

        areas = page.search_for("mupdf")

This delivers a list of rectangles (see [Rect](https://pymupdf.readthedocs.io/en/latest/rect.html#rect)), each of which surrounds one occurrence of the string “mupdf” (case insensitive).

