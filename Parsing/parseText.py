import fitz

#print(fitz.__doc__)
pdf = fitz.open('bipedal_robot.pdf')
with open('text.txt', 'w', encoding="utf-8") as file: #creates a text output, w means write mode, encoding is set to utf-8 to handle special characters
    for i in range(pdf.page_count):
        page = pdf.load_page(i)
        text = page.get_text('text')
        file.write(text)


pdf.close()

