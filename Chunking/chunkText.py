from langchain.text_splitter import RecursiveCharacterTextSplitter


with open(r"E:\IITB\SoC LLM Research Agent\LLM-Powered-Research-Agent\Parsing\text.txt", "r", encoding="utf-8") as f:
        text = f.read()

text_splitter = RecursiveCharacterTextSplitter(
            # separator=["\n\n", "\n", " ", ""] default list
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
)
# texts = text_splitter.create_documents([text]) # Create a list of documents with the text split into chunks
# print(texts[0]) #prints the first chunk of text along with metadata
# print(texts[0].page_content) #prints the content of the first chunk of text

chunks = text_splitter.split_text(text)

# Save those chunks into another .txt file
with open("chunks_output.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"--- CHUNK {i+1} START ---\n")
        f.write(chunk)
        f.write("\n--- CHUNK END ---\n\n")


