import asyncio
import datetime as dt
from datetime import timedelta
from unittest.mock import AsyncMock, patch

import mistralai.client.models
import pytest
from mistralai import workflows
from mistralai.client.models import ChatCompletionResponse
from mistralai.workflows.testing import create_test_worker
from worker.workflows.veille_media import (
    ArticleDigest,
    ToolCallClient,
    VeilleDigest,
    VeilleInput,
    VeilleMediaWorkflow,
    _fit_message,
    _format_digest,
    collect_articles,
    send_slack_dm,
)

WORKFLOW_EXECUTION_TIMEOUT = timedelta(seconds=10)

DIGEST = VeilleDigest(
    articles=[
        ArticleDigest(
            media="Le Monde",
            titre="Nouveau modèle d'IA",
            resume="Le journal décrit la sortie du modèle.",
            lien="https://www.lemonde.fr/article-1",
            date_publication="2026-02-10",
        ),
        ArticleDigest(
            media="Les Échos",
            titre="Marché de l'IA",
            resume="Les Échos analysent la croissance du marché.",
            lien="https://www.lesechos.fr/article-2",
        ),
    ]
)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [type("Text", (), {"text": text})()]


def _fake_agent_chunks() -> list:
    chunk = type("Chunk", (), {})()
    chunk.text = '[{"media": "Le Monde", "titre": "Nouveau modèle d\'IA"}]'
    return [chunk]


def _patch_externals(slack_calls: list[tuple[str, dict]]):
    async def fake_call_tool(self, tool_name, arguments=None):
        slack_calls.append((tool_name, arguments))
        if tool_name == "slack_read_user_profile":
            return _FakeResponse('{"ok": true, "user": {"id": "U0CGGJEE9KR"}}')
        return _FakeResponse('{"ok": true}')

    async def fake_chat_parse(params, response_format) -> ChatCompletionResponse:
        return ChatCompletionResponse.model_validate(
            {
                "id": "fake-parse",
                "model": "mistral-small-latest",
                "object": "chat.completion",
                "created": 0,
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": DIGEST.model_dump_json(),
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    @workflows.activity(name="mistralai_chat_parse")
    async def fake_chat_parse_activity(
        params: mistralai.client.models.ChatCompletionRequest,
        response_format: mistralai.client.models.ResponseFormat,
    ) -> ChatCompletionResponse:
        return await fake_chat_parse(params, response_format)

    return (
        patch(
            "mistralai.workflows.plugins.mistralai.Runner.run",
            new=AsyncMock(return_value=_fake_agent_chunks()),
        ),
        patch.object(ToolCallClient, "call_tool", fake_call_tool),
        fake_chat_parse_activity,
    )


def test_format_digest_groups_by_media() -> None:
    message = _format_digest(
        DIGEST, ["Le Monde", "Les Échos"], ["IA"], dt.date(2026, 2, 9)
    )
    assert "Veille d'actualité" in message
    assert "*Le Monde*" in message
    assert "*Les Échos*" in message
    assert "<https://www.lemonde.fr/article-1|Nouveau modèle d'IA>" in message
    assert "(2026-02-10)" in message


def test_format_digest_without_articles() -> None:
    message = _format_digest(VeilleDigest(), ["Le Monde"], ["IA"], dt.date(2026, 2, 9))
    assert "0 article(s) trouvé(s)" in message


def test_fit_message_truncates() -> None:
    assert _fit_message("court") == "court"
    fitted = _fit_message("x" * 5000)
    assert len(fitted) <= 3900
    assert "tronqué" in fitted


def test_input_rejects_empty_medias() -> None:
    with pytest.raises(ValueError):
        VeilleInput(medias=[], themes=["IA"], depuis="2026-02-09")


async def test_workflow_sends_digest_to_self_dm(temporal_env) -> None:
    slack_calls: list[tuple[str, dict]] = []
    runner_patch, call_patch, fake_chat_parse = _patch_externals(slack_calls)
    with runner_patch, call_patch:
        async with create_test_worker(
            temporal_env,
            workflows=[VeilleMediaWorkflow],
            activities=[collect_articles, send_slack_dm, fake_chat_parse],
        ):
            handle = await temporal_env.client.start_workflow(
                "veille-media-slack",
                VeilleInput(
                    medias=["Le Monde"], themes=["IA"], depuis="2026-02-09"
                ).model_dump(mode="json"),
                id="test-veille-media",
                task_queue="test-task-queue",
                execution_timeout=WORKFLOW_EXECUTION_TIMEOUT,
            )
            result = await asyncio.wait_for(handle.result(), timeout=15)

    assert result["nb_articles"] == 2
    assert result["canal"] == "dm-self"
    tool_names = [name for name, _ in slack_calls]
    assert "slack_send_message" in tool_names
    sent = next(args for name, args in slack_calls if name == "slack_send_message")
    assert sent["channel_id"] == "U0CGGJEE9KR"
    assert "Veille d'actualité" in sent["message"]
    assert "lemonde.fr" in sent["message"]
