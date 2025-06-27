Once we have the text:
- Clean it (remove headers, footers, strange symbols, blank spaces)
- **Chunk** it: break the long document into smaller parts (like paragraphs or 200-word sections)

This is important because LLMs and vector databases work better with smaller, digestible pieces.

I will be using Langchain's `RecursiveCharacterTextSplitter` for this task. It allows us to split the text into smaller chunks while keeping the context intact. 
Langchain a popular framework for developing applications with large language models (LLMs), offers a variety of text splitting techniques. Each method is designed to cater to different types of documents and specific use cases.

References:
    1. [Langchain Text Splitters Documentation](https://python.langchain.com/docs/concepts/text_splitters/)
    2. [Mastering Text Splitting in Langchain](https://medium.com/@harsh.vardhan7695/mastering-text-splitting-in-langchain-735313216e01)

There are several reasons to split documents:

1. Handling non-uniform document lengths: Real-world document collections often contain texts of varying sizes. Splitting ensures consistent processing across all documents.
2. Overcoming model limitations: Many embedding models and language models have maximum input size constraints. Splitting allows us to process documents that would otherwise exceed these limits.
3. Improving representation quality: For longer documents, the quality of embeddings or other representations may degrade as they try to capture too much information. Splitting can lead to more focused and accurate representations of each section.
4. Enhancing retrieval precision: In information retrieval systems, splitting can improve the granularity of search results, allowing for more precise matching of queries to relevant document sections.
5. Optimizing computational resources: Working with smaller chunks of text can be more memory-efficient and allow for better parallelization of processing tasks.

For installation, run: `pip install -qU langchain-text-splitters`, `pip install langchain`


## Approches for Chunking Text

### Length Based Chunking

The most intuitive strategy is to split documents based on their length. This simple yet effective approach ensures that each chunk doesn't exceed a specified size limit. Key benefits of length-based splitting:
- Straightforward implementation
- Consistent chunk sizes
- Easily adaptable to different model requirements

        from langchain_text_splitters import CharacterTextSplitter

        # Load an example document
        with open("state_of_the_union.txt") as f:
            state_of_the_union = f.read()

        text_splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([state_of_the_union])
        print(texts[0])

👉 use .split_text() if you only care about text content
👉 use .create_documents() if you want text + metadata in a Document wrapper

### Text Structure Based Chunking


