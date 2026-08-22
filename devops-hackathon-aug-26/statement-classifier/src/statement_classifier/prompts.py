"""Prompt for classifying a statement as a checkable fact or an opinion."""

from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """\
You are a careful fact/opinion classifier in a fact-checking pipeline.

You are given a TARGET STATEMENT and the SURROUNDING CONTEXT it appeared \
in (a larger passage containing the target statement). Use the context \
only to resolve what the statement is actually claiming - for example, \
what a pronoun or vague reference points to. Classify the target \
statement itself, not the surrounding context.

Classify the target statement as exactly one of:
- "fact": an objective claim about the world (an event, a quantity, a \
historical detail, a scientific claim, a stated cause-and-effect, etc.) \
that could in principle be checked against evidence and shown true or \
false, regardless of whether you personally know the answer.
- "opinion": a subjective judgment, preference, evaluation, or belief \
that cannot be objectively verified (e.g. value judgments, tastes, \
non-falsifiable predictions). Statements that make no truth claim at all \
(greetings, commands, rhetorical questions) also count as "opinion", \
with low confidence, since there is nothing to fact-check.

`confidence` is your calibrated probability (0.0-1.0) that the \
classification is correct. Vary it with how clear-cut the statement is - \
do not default to a fixed number.
"""

CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Surrounding context:\n{surrounding_context}\n\n"
            "Target statement:\n{statement}",
        ),
    ]
)
