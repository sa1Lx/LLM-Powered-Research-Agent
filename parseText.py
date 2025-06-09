import fitz

#print(fitz.__doc__)
pdf = fitz.open('Lec Notes.pdf')
with open('text.txt', 'w', encoding="utf-8") as file:
    for i in range(147):
        page = pdf.load_page(i)
        text = page.get_text('text')
        file.write(text)


pdf.close()

