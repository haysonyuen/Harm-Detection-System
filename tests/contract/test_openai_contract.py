"""OpenAI Moderation + analyst adapter contracts using the SDK's own response types (no network)."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from openai.types import ModerationCreateResponse

from harm_detection.improvement.schemas import AnalysisOutput
from harm_detection.providers.base import ProviderError, RetryPolicy
from harm_detection.providers.openai_analyst import OpenAIAnalyst
from harm_detection.providers.openai_moderation import (
    OpenAIModerationProvider,
    normalize_moderation,
)
from harm_detection.schemas.content import ContentInput

ITEM = ContentInput(case_id="c1", source="manual", text="text", parent_text="ignored for parity")
CATS = [
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/instructions",
    "self-harm/intent",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
]


def moderation_payload(h=0.8, ht=0.1, hf=True, htf=False):
    scores = {c: 0.0 for c in CATS} | {"harassment": h, "harassment/threatening": ht}
    flags = {c: False for c in CATS} | {"harassment": hf, "harassment/threatening": htf}
    return {
        "id": "modr-1",
        "model": "omni-moderation-2024-09-26",
        "results": [
            {
                "flagged": hf or htf,
                "categories": flags,
                "category_scores": scores,
                "category_applied_input_types": {c: ["text"] for c in CATS},
            }
        ],
    }


class FakeModerations:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return ModerationCreateResponse.model_validate(r)


def make_mod(responses):
    fake = SimpleNamespace(moderations=FakeModerations(responses))
    return OpenAIModerationProvider(
        "sk-test", client=fake, retry=RetryPolicy(max_attempts=2), sleep=lambda s: None
    ), fake


def test_sdk_type_normalization():
    sdk_obj = ModerationCreateResponse.model_validate(moderation_payload())
    signals, model, rid = normalize_moderation(sdk_obj.model_dump(by_alias=True))
    assert signals == {
        "harassment": 0.8,
        "harassment_flagged": 1.0,
        "harassment_threatening": 0.1,
        "harassment_threatening_flagged": 0.0,
    }
    assert model.startswith("omni-moderation") and rid == "modr-1"


def test_provider_sends_text_only():
    prov, fake = make_mod([moderation_payload()])
    res = prov.detect(ITEM)
    assert res.status.value == "success"
    assert fake.moderations.calls[0] == {"model": "omni-moderation-latest", "input": "text"}
    assert set(res.raw_response["results"][0]["categories"]) == {
        "harassment",
        "harassment/threatening",
    }


def _status_error(cls, code):
    req = httpx.Request("POST", "https://api.openai.com/v1/moderations")
    return cls("err", response=httpx.Response(code, request=req), body=None)


@pytest.mark.parametrize(
    "exc,code",
    [
        (lambda: _status_error(openai.AuthenticationError, 401), "authentication_failed"),
        (lambda: _status_error(openai.PermissionDeniedError, 403), "permission_denied"),
        (lambda: _status_error(openai.NotFoundError, 404), "unsupported_model"),
        (lambda: _status_error(openai.BadRequestError, 400), "bad_request"),
    ],
)
def test_permanent_errors(exc, code):
    prov, fake = make_mod([exc()])
    res = prov.detect(ITEM)
    assert res.error["code"] == code and res.signals is None and len(fake.moderations.calls) == 1


def test_rate_limit_retried():
    prov, fake = make_mod([_status_error(openai.RateLimitError, 429), moderation_payload()])
    assert prov.detect(ITEM).status.value == "success" and len(fake.moderations.calls) == 2


def test_missing_category_is_failure():
    p = moderation_payload()
    del p["results"][0]["category_scores"]["harassment/threatening"]
    with pytest.raises(ProviderError):
        normalize_moderation(p)


def test_missing_key():
    assert OpenAIModerationProvider(None).detect(ITEM).error["code"] == "missing_credentials"


# ---------------------------------------------------------------- analyst

VALID = {
    "analysis_id": "a",
    "scope": "single_case",
    "error_type": "false_positive",
    "observations": [],
    "hypotheses": [],
    "recommendations": [],
    "limitations": [],
    "unsupported_or_unknown": [],
}


class FakeResponses:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def parse(self, **kw):
        self.kwargs = kw
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_analyst_structured_parse():
    resp = SimpleNamespace(
        output_parsed=AnalysisOutput.model_validate(VALID),
        id="resp_1",
        model="gpt-5.4-mini",
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
    )
    fake = SimpleNamespace(responses=FakeResponses(resp))
    out = OpenAIAnalyst("sk-test", client=fake).analyze("data")
    assert out.input_tokens == 1000 and out.output.analysis_id == "a"
    assert fake.responses.kwargs["text_format"] is AnalysisOutput
    assert fake.responses.kwargs["model"] == "gpt-5.4-mini"
    assert "UNTRUSTED" in fake.responses.kwargs["instructions"]


def test_analyst_dict_output_validated():
    resp = SimpleNamespace(output_parsed={**VALID, "scope": "bogus"}, usage=None)
    with pytest.raises(ProviderError):
        OpenAIAnalyst("sk-test", client=SimpleNamespace(responses=FakeResponses(resp))).analyze("x")


def test_analyst_no_parsed_output():
    resp = SimpleNamespace(output_parsed=None, usage=None)
    with pytest.raises(ProviderError, match="no parsable"):
        OpenAIAnalyst("sk-test", client=SimpleNamespace(responses=FakeResponses(resp))).analyze("x")


def test_analyst_without_key():
    with pytest.raises(ProviderError):
        OpenAIAnalyst(None).analyze("x")
