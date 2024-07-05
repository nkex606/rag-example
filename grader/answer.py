from langchain import hub
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class AnswerGrader(BaseModel):
    binary_score: str = Field(
        description="Is the answer resolves the question or not('yes' or 'no')"
    )


def run(question, generation: str):
    prompt = hub.pull("efriis/self-rag-answer-grader")
    model = "gpt-4o"
    llm = ChatOpenAI(model=model, temperature=0)
    structured_llm_grader = llm.with_structured_output(AnswerGrader)
    answer_grader = prompt | structured_llm_grader
    return answer_grader.invoke({"question": question, "generation": generation})
