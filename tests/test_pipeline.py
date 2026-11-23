import os
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from schemas import TechnicalEntities
from chain import ExtractionError, build_chain, build_model, process_text
import main


def envelope(parsed=None, *, error=None, finish_reason="stop"):
    return {
        "raw": AIMessage(content="", response_metadata={"finish_reason": finish_reason}),
        "parsed": parsed,
        "parsing_error": error,
    }


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.schema = None
        self.include_raw = None
        self.prompts = []

    def with_structured_output(self, schema, *, include_raw):
        self.schema = schema
        self.include_raw = include_raw

        async def answer(prompt):
            self.calls += 1
            self.prompts.append(prompt)
            value = next(self.responses)
            if isinstance(value, Exception):
                raise value
            return value

        return RunnableLambda(answer)


VALID = TechnicalEntities(
    tecnologias=["FastAPI", "Redis", "PostgreSQL"],
    nivel_de_criticidad="alta",
    resumen_tecnico="API with cache and database contention.",
)


class SchemaTests(unittest.TestCase):
    def test_validated_contract(self):
        self.assertEqual(VALID.nivel_de_criticidad.value, "alta")
        self.assertEqual(TechnicalEntities.model_validate({
            "tecnologias": [],
            "nivel_de_criticidad": "baja",
            "resumen_tecnico": "No technology is named.",
        }).tecnologias, [])

        for invalid in (
            {"tecnologias": [""], "nivel_de_criticidad": "alta", "resumen_tecnico": "Valid"},
            {"tecnologias": ["Redis"], "nivel_de_criticidad": "urgente", "resumen_tecnico": "Valid"},
            {"tecnologias": ["Redis"], "nivel_de_criticidad": "alta", "resumen_tecnico": "  "},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                TechnicalEntities.model_validate(invalid)


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_chain_uses_schema_and_returns_validated_model(self):
        model = FakeModel([envelope(VALID)])
        result = await build_chain(model).ainvoke({
            "text": "FastAPI uses Redis and PostgreSQL.",
            "format_instructions": "Return the requested fields.",
        })
        self.assertEqual(result, VALID)
        self.assertIs(model.schema, TechnicalEntities)
        self.assertTrue(model.include_raw)
        self.assertEqual(model.calls, 1)
        self.assertIn("Return the requested fields.", model.prompts[0].messages[0].content)
        self.assertIn("FastAPI uses Redis", model.prompts[0].messages[1].content)

    async def test_malformed_output_is_retried_once(self):
        model = FakeModel([envelope(error=ValueError("bad JSON")), envelope(VALID)])
        with self.assertLogs("chain", level="WARNING") as logs:
            result = await build_chain(model).ainvoke({"text": "Redis failed", "format_instructions": "JSON"})
        self.assertEqual(result, VALID)
        self.assertEqual(model.calls, 2)
        self.assertIn("invalid structured output", " ".join(logs.output))

    async def test_truncated_output_is_retried_before_accepting_parsed_value(self):
        for reason in ("length", "max_tokens"):
            with self.subTest(reason=reason):
                model = FakeModel([envelope(VALID, finish_reason=reason), envelope(VALID)])
                with self.assertLogs("chain", level="WARNING"):
                    result = await build_chain(model).ainvoke({"text": "Redis failed", "format_instructions": "JSON"})
                self.assertEqual(result, VALID)
                self.assertEqual(model.calls, 2)

    async def test_incomplete_fields_are_retried_once(self):
        incomplete = {"tecnologias": ["Redis"], "nivel_de_criticidad": "alta"}
        model = FakeModel([envelope(incomplete), envelope(VALID)])
        with self.assertLogs("chain", level="WARNING"):
            result = await build_chain(model).ainvoke({"text": "Redis failed", "format_instructions": "JSON"})
        self.assertEqual(result, VALID)
        self.assertEqual(model.calls, 2)

    async def test_failed_validation_has_controlled_error_after_two_attempts(self):
        model = FakeModel([envelope(error=ValueError("bad JSON")), envelope(error=ValueError("bad JSON"))])
        with patch("chain.build_chain", return_value=build_chain(model)):
            with self.assertLogs("chain", level="WARNING"):
                with self.assertRaises(ExtractionError):
                    await process_text("Redis failed")
        self.assertEqual(model.calls, 2)

    async def test_provider_error_is_not_retried(self):
        model = FakeModel([RuntimeError("network failed")])
        with self.assertRaises(RuntimeError):
            await build_chain(model).ainvoke({"text": "Redis failed", "format_instructions": "JSON"})
        self.assertEqual(model.calls, 1)

    async def test_empty_input_rejected_without_model_call(self):
        with self.assertRaises(ValueError):
            await process_text("  ")


class ProviderTests(unittest.TestCase):
    def test_provider_selection_uses_environment_configuration(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test-key"}, clear=True), \
             patch("chain.load_dotenv"), patch("chain.ChatAnthropic") as chat:
            self.assertIs(build_model(), chat.return_value)
            self.assertEqual(chat.call_args.kwargs["model"], "claude-haiku-4-5-20251001")

    def test_missing_key_is_configuration_error(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": ""}, clear=True), \
             patch("chain.load_dotenv"):
            with self.assertRaises(ExtractionError):
                build_model()

    def test_openai_is_default_provider(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True), \
             patch("chain.load_dotenv"), patch("chain.ChatOpenAI") as chat:
            self.assertIs(build_model(), chat.return_value)
            self.assertEqual(chat.call_args.kwargs["model"], "gpt-4o-mini")


class ScriptTests(unittest.TestCase):
    def test_demo_prints_validated_json(self):
        output = StringIO()
        with patch("sys.argv", ["main.py", "--text", "Redis failed"]), \
             patch("main.process_text", new_callable=AsyncMock, return_value=VALID), \
             redirect_stdout(output):
            self.assertEqual(main.main(), 0)
        self.assertIn('"nivel_de_criticidad": "alta"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
