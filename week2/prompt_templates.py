"""
prompt_templates.py
-----------------------
LEARNING GOALS
  - ChatPromptTemplate turns a prompt into a REUSABLE, parameterized Runnable.
    Instead of f-string concatenation, you define placeholders once and `.invoke()`
    with different variables each time -- this is what makes prompts composable
    into chains later.
  - Understand from_messages() (list of role/template tuples), from_template()
    (single string), and MessagesPlaceholder (inject a whole message list, e.g.
    chat history, at a fixed position).
  - Understand .partial() for pre-filling some variables ahead of time.

TASKS
  1. Build a ChatPromptTemplate with 2+ variables and call .invoke() with different
     dicts -- confirm it's stateless and reusable (no leftover values between calls).
  2. Add a MessagesPlaceholder("history") and pass in a growing list of past
     messages to simulate a chatbot with memory.
  3. Use .partial() to lock in a "persona" variable so downstream callers only need
     to supply the topic, not the persona every time.
  4. Print prompt.invoke({...}) directly (before piping to a model) and inspect the
     resulting ChatPromptValue / list of messages it produces.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from model_factory import get_model

PROVIDER = "groq"


def basic_template():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a {persona}. Answer in {style} style."),
        ("human", "{question}"),
    ])

    # Reusable: same template, different inputs
    filled_1 = prompt.invoke({"persona": "history professor", "style": "formal", "question": "Why did Rome fall?"})
    filled_2 = prompt.invoke({"persona": "stand-up comedian", "style": "sarcastic", "question": "Why did Rome fall?"})

    print("Filled prompt #1 messages:", filled_1.to_messages())
    print("Filled prompt #2 messages:", filled_2.to_messages())
    return prompt


def template_with_history():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise assistant."),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    history = [
        HumanMessage(content="My name is Rahul."),
        AIMessage(content="Nice to meet you, Rahul."),
    ]

    chain = prompt | get_model(PROVIDER)
    response = chain.invoke({"history": history, "question": "What's my name?"})
    print("\nResponse using injected history:", response.content)


def partial_prompt():
    prompt = ChatPromptTemplate.from_template(
        "You are a {persona}. Explain {topic} in under 3 sentences."
    )
    locked_persona_prompt = prompt.partial(persona="patient teacher for beginners")

    chain = locked_persona_prompt | get_model(PROVIDER)
    response = chain.invoke({"topic": "recursion"})
    print("\nPartial-filled prompt response:", response.content)


if __name__ == "__main__":
    basic_template()
    template_with_history()
    partial_prompt()