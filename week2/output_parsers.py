"""
output_parsers.py
----------------------
LEARNING GOALS
  - Output parsers are Runnables that convert a model's raw AIMessage/string into
    a structured Python object -- but unlike with_structured_output(), they work
    by (a) injecting format instructions INTO the prompt and (b) parsing the raw
    text response themselves, so they work even on models without native
    structured-output support.
  - StrOutputParser: trivial parser, just extracts .content as a plain string --
    used constantly at the end of chains to avoid dealing with AIMessage objects.
  - JsonOutputParser: parses raw JSON text into a dict, optionally validated
    against a Pydantic schema. Supports STREAMING partial JSON (advanced).
  - PydanticOutputParser: like JsonOutputParser but returns a validated Pydantic
    instance, and exposes get_format_instructions() to embed in your prompt.

TASKS
  1. Compare with_structured_output() (previous file) vs PydanticOutputParser
     here on the SAME schema/input -- which is simpler code? Which is more
     robust if the model slightly malforms JSON?
  2. Deliberately use a weaker/smaller model and see if PydanticOutputParser
     raises a parsing error -- then inspect get_format_instructions() to see
     exactly what's being injected into the prompt to guide the model.
  3. Try JsonOutputParser WITHOUT a Pydantic schema (just parses to dict) and
     stream it with .stream() -- watch partial dicts arrive as valid JSON forms.
"""

from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from model_factory import get_model

PROVIDER = "groq"


def str_output_parser_demo():
    model = get_model(PROVIDER)
    parser = StrOutputParser()
    chain = model | parser  # AIMessage -> plain str

    result = chain.invoke("Say hello in 3 languages.")
    print("Type:", type(result).__name__)  # str, not AIMessage
    print("Result:", result)


class KeyIssue(BaseModel):
    summary: str = Field(description="One-line summary of the customer review")
    sentiment: str = Field(description="positive, negative, or neutral")
    issues: list[str] = Field(description="Specific problems mentioned, empty list if none")


def pydantic_output_parser_demo():
    parser = PydanticOutputParser(pydantic_object=KeyIssue)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract structured info from reviews.\n{format_instructions}"),
        ("human", "{review}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    model = get_model(PROVIDER)
    chain = prompt | model | parser  # full LCEL chain: prompt -> model -> parser

    review = "Delivery took 2 weeks and the box arrived crushed. Product itself works fine though."
    result = chain.invoke({"review": review})
    print("\nType:", type(result).__name__)
    print("Parsed:", result)
    print("Format instructions preview:\n", parser.get_format_instructions()[:300], "...")


def json_output_parser_demo():
    parser = JsonOutputParser(pydantic_object=KeyIssue)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Return JSON matching this schema:\n{format_instructions}"),
        ("human", "{review}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    chain = prompt | get_model(PROVIDER) | parser
    review = "Loved the packaging and speed, but customer support never replied to my email."

    print("\nStreaming partial JSON as it forms:")
    for partial in chain.stream({"review": review}):
        print(partial)


if __name__ == "__main__":
    str_output_parser_demo()
    pydantic_output_parser_demo()
    json_output_parser_demo()