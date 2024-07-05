import getpass
import os

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from grader import answer

os.environ["OPENAI_API_KEY"] = getpass.getpass()
question: str = "台灣有名的飲料是什麼？"
model: str = "gpt-4o"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    llm = ChatOpenAI(model=model)
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    generation = rag_chain.invoke(question)
    print("user question: " + question, "retrieve result: " + generation)

    score = answer.run(question, generation)
    print("score: " + score.binary_score)

    if score.binary_score == "yes":
        print("answer resolves question, generation result:\n" + generation)
    else:
        generation = llm.invoke(question)
        print(
            "answer did not resolve question, regenerate answer by llm:\n"
            + generation.content
        )
