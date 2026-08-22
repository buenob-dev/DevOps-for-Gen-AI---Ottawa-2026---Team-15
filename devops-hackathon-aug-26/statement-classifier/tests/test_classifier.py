from statement_classifier import (
    StatementInput,
    StatementsInput,
    classify,
    classify_batch,
    classify_statement,
)


def test_classify_statement_returns_llm_verdict(fake_fact_llm):
    result = classify_statement(
        "The Eiffel Tower is 330 metres tall.",
        "We visited Paris last summer. The Eiffel Tower is 330 metres tall. It was hot.",
        llm=fake_fact_llm,
    )

    assert result.class_ == "fact"
    assert result.confidence == 0.9


def test_classify_statement_formats_prompt_with_both_inputs(fake_fact_llm):
    classify_statement("target", "context", llm=fake_fact_llm)

    [messages] = fake_fact_llm.structured.invocations
    rendered = "\n".join(message.content for message in messages)
    assert "target" in rendered
    assert "context" in rendered


def test_classify_preserves_statement_and_context(fake_opinion_llm):
    item = StatementInput(surroundingContext="Some context.", statement="Some statement.")

    result = classify(item, llm=fake_opinion_llm)

    assert result.surrounding_context == "Some context."
    assert result.statement == "Some statement."
    assert result.classification.class_ == "opinion"
    assert result.classification.confidence == 0.8


def test_classify_round_trips_camel_case_json():
    item = StatementInput.model_validate(
        {"surroundingContext": "Some context.", "statement": "Some statement."}
    )

    assert item.surrounding_context == "Some context."


def test_classified_statement_serializes_with_camel_case_and_class_alias(fake_fact_llm):
    item = StatementInput(surroundingContext="Some context.", statement="Some statement.")

    result = classify(item, llm=fake_fact_llm)

    dumped = result.model_dump(by_alias=True)
    assert dumped["surroundingContext"] == "Some context."
    assert dumped["classification"]["class"] == "fact"


def test_classify_batch_classifies_every_statement(fake_fact_llm):
    batch = StatementsInput.model_validate(
        {
            "statements": [
                {"surroundingContext": "ctx one", "statement": "statement one"},
                {"surroundingContext": "ctx two", "statement": "statement two"},
            ]
        }
    )

    result = classify_batch(batch, llm=fake_fact_llm)

    assert [s.statement for s in result.statements] == ["statement one", "statement two"]
    assert all(s.classification.class_ == "fact" for s in result.statements)
