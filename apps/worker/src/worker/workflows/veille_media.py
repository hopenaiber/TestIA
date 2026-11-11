"""Veille d'actualité : recherche web par médias et thèmes, récapitulatif envoyé en DM Slack.

Entrées : liste de médias, liste de thèmes, date de début (inclusive) et optionnellement
l'ID Slack du destinataire (par défaut : l'utilisateur qui lance le workflow, donc son DM
avec lui-même). Sortie : le récapitulatif formaté et le statut de l'envoi Slack.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import mistralai.workflows.plugins.mistralai as workflows_mistralai
import pydantic
from mistralai import workflows
from mistralai.workflows import Depends
from mistralai.workflows.plugins.mistralai import (
    ChatCompletionRequest,
    UserMessage,
    chat_parse_to_model,
)
from mistralai.workflows.plugins.mistralai.connectors import (
    ToolCallClient,
    connector,
    uses_connectors,
)

slack = connector("slack")

SEARCH_MODEL = "mistral-medium-latest"
PARSE_MODEL = "mistral-small-latest"
MAX_AGENT_TURNS = 20
SLACK_MESSAGE_LIMIT = 3900

SEARCH_INSTRUCTIONS = (
    "Tu es un assistant de veille d'actualité. On te donne une liste de médias, une liste de "
    "thèmes et une date de début. Tu cherches sur le web les articles parus depuis cette date "
    "(inclusive) dans ces médias sur ces thèmes, en privilégiant les sites officiels des médias "
    "demandés. Pour chaque article trouvé, tu notes : le média, le titre exact, l'URL de l'article, "
    "sa date de publication (AAAA-MM-JJ) et une phrase de résumé. Tu ignores les articles plus "
    "anciens que la date de début, les pages d'accueil, les flux RSS et les rubriques vides. "
    "Tu réponds uniquement par une liste d'articles au format JSON, sans commentaire."
)

PARSE_INSTRUCTIONS = (
    "Extrais la liste des articles du texte ci-dessous. Pour chaque article : media (nom du média), "
    "titre (titre de l'article), resume (une seule phrase en français), lien (URL de l'article), "
    "date_publication (AAAA-MM-JJ ou chaîne vide). Ignore ce qui n'est pas un article daté et lié."
)


class VeilleInput(pydantic.BaseModel):
    medias: list[str] = pydantic.Field(min_length=1)
    themes: list[str] = pydantic.Field(min_length=1)
    depuis: dt.date
    slack_user_id: str = ""


class ArticleDigest(pydantic.BaseModel):
    media: str
    titre: str
    resume: str
    lien: str
    date_publication: str = ""


class VeilleDigest(pydantic.BaseModel):
    articles: list[ArticleDigest] = pydantic.Field(default_factory=list)


class VeilleOutput(pydantic.BaseModel):
    nb_articles: int
    canal: str
    message: str
    resume: str


def _tool_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content if isinstance(content, list) else [content]:
        text = getattr(item, "text", None)
        if text is None and isinstance(item, dict):
            text = item.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _extract_user_id(result: Any) -> str:
    match = re.search(r"\bU[A-Z0-9]{8,}\b", _tool_text(result))
    return match.group(0) if match else ""


def _format_digest(
    digest: VeilleDigest, medias: list[str], themes: list[str], depuis: dt.date
) -> str:
    header = (
        f":newspaper: *Veille d'actualité* — depuis le {depuis.isoformat()}\n"
        f"Médias : {', '.join(medias)}\nThèmes : {', '.join(themes)}\n"
        f"{len(digest.articles)} article(s) trouvé(s)"
    )
    if not digest.articles:
        return header
    lines = [header]
    by_media: dict[str, list[ArticleDigest]] = {}
    for article in digest.articles:
        by_media.setdefault(article.media, []).append(article)
    for media, articles in by_media.items():
        lines.append(f"\n*{media}*")
        for article in articles:
            published = (
                f" ({article.date_publication})" if article.date_publication else ""
            )
            lines.append(
                f"• <{article.lien}|{article.titre}>{published} — {article.resume}"
            )
    return "\n".join(lines)


def _fit_message(message: str) -> str:
    if len(message) <= SLACK_MESSAGE_LIMIT:
        return message
    suffix = "\n… _récapitulatif tronqué_"
    return message[: SLACK_MESSAGE_LIMIT - len(suffix)].rstrip() + suffix


@workflows.activity(name="collecter-articles")
async def collect_articles(medias: list[str], themes: list[str], depuis: str) -> str:
    from mistralai.client.models import TextChunk, WebSearchTool

    agent = workflows_mistralai.Agent(
        model=SEARCH_MODEL,
        name="veille-media-agent",
        description="Recherche les articles parus sur les médias et thèmes demandés",
        instructions=SEARCH_INSTRUCTIONS,
        tools=[WebSearchTool()],
    )
    prompt = (
        f"Médias : {', '.join(medias)}\n"
        f"Thèmes : {', '.join(themes)}\n"
        f"Articles parus depuis le {depuis} (inclus).\n"
        "Cherche sur le web et renvoie la liste JSON des articles trouvés."
    )
    outputs = await workflows_mistralai.Runner.run(
        agent=agent, inputs=prompt, max_turns=MAX_AGENT_TURNS
    )
    return "\n".join(output.text for output in outputs if isinstance(output, TextChunk))


@workflows.activity(name="envoyer-dm-slack")
async def send_slack_dm(
    user_id: str,
    message: str,
    slack_client: ToolCallClient = Depends(slack),  # noqa: B008 — pattern d'injection du SDK
) -> str:
    target = user_id
    if not target:
        profile = await slack_client.call_tool(
            tool_name="slack_read_user_profile",
            arguments={},
        )
        target = _extract_user_id(profile)
    if not target:
        raise workflows.WorkflowError(
            "Impossible de déterminer le destinataire Slack du récapitulatif."
        )
    result = await slack_client.call_tool(
        tool_name="slack_send_message",
        arguments={"channel_id": target, "message": message},
    )
    return _tool_text(result) or "message envoyé"


@workflows.workflow.define(name="veille-media-slack", on_behalf_of=True)
@uses_connectors(slack)
class VeilleMediaWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, prompt: VeilleInput) -> VeilleOutput:
        raw_articles = await collect_articles(
            prompt.medias,
            prompt.themes,
            prompt.depuis.isoformat(),
        )

        parse_request = ChatCompletionRequest(
            model=PARSE_MODEL,
            messages=[
                UserMessage(content=f"{PARSE_INSTRUCTIONS}\n\n{raw_articles}"),
            ],
        )
        digest = await chat_parse_to_model(VeilleDigest, parse_request)
        digest.articles = [a for a in digest.articles if a.lien.startswith("http")]

        message = _fit_message(
            _format_digest(digest, prompt.medias, prompt.themes, prompt.depuis)
        )
        slack_response = await send_slack_dm(prompt.slack_user_id, message)

        return VeilleOutput(
            nb_articles=len(digest.articles),
            canal=prompt.slack_user_id or "dm-self",
            message=message,
            resume=slack_response,
        )
