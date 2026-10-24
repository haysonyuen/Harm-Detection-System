"""Jev adapter contract tests against the documented System One OpenAPI contract (v0.2.0)."""

import json

import httpx
import pytest
import respx

from harm_detection.providers.base import ProviderError, RetryPolicy
from harm_detection.providers.jev import JevProvider, build_request, normalize_response
from harm_detection.schemas.content import ContentInput
from tests.conftest import jev_response

URL = "https://api.typesafe.ai/v1/systemone"
SIG = {
    "targeted_individual": 0.8,
    "abusive_behavior": 0.7,
    "physical_threat": 0.1,
    "context_sufficient": 0.9,
}
ITEM = ContentInput(
    case_id="c1", source="manual", text="you are a clown", parent_text="parent here"
)


def provider(questions, **kw):
    return JevProvider(
        "tk-test-key-123456",
        questions,
        retry=RetryPolicy(max_attempts=3),
        sleep=lambda s: None,
        **kw,
    )


def test_request_matches_documented_schema(questions):
    req = build_request(ITEM, questions, "jev-latest")
    assert set(req) == {"model", "state", "questions"}
    assert req["model"] == "jev-latest"
    assert req["state"]["comment"] == "you are a clown"
    assert req["state"]["parent_comment"] == "parent here"
    assert "target_context" not in req["state"]  # missing context is omitted, not invented
    assert set(req["questions"]) == set(SIG)
    for q in req["questions"].values():
        assert q["type"] == "noul" and isinstance(q["instructions"], str)
        assert set(q["criteria"]) == {"true", "false"}
    json.dumps(req)  # serializable


def test_text_only_mode_drops_context(questions):
    req = build_request(ITEM, questions, "jev-latest", include_context=False)
    assert "parent_comment" not in req["state"]


@respx.mock
def test_success_normalization_and_headers(questions):
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=jev_response(SIG)))
    res = provider(questions).detect(ITEM)
    assert res.status.value == "success"
    assert res.signals == SIG
    assert res.resolved_model == "jev-1.14.0" and res.requested_model == "jev-latest"
    assert res.provider_usage["input_tokens"] == 512
    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer tk-test-key-123456"
    assert "tk-test-key-123456" not in json.dumps(res.raw_response)


def test_documented_example_response_shape(questions):
    body = {
        "model": "jev-latest",
        "answers": {k: {"type": "noul", "noul": 0.98} for k in SIG},
        "usage": {"input_tokens": 120, "output_tokens": 12},
    }
    signals, model, usage = normalize_response(body, questions)
    assert (
        all(v == 0.98 for v in signals.values())
        and model == "jev-latest"
        and usage["output_tokens"] == 12
    )


@pytest.mark.parametrize(
    "body,code",
    [
        (
            {
                "model": "m",
                "answers": {
                    k: {"type": "noul", "noul": v} for k, v in SIG.items() if k != "physical_threat"
                },
                "usage": {},
            },
            "missing_answer",
        ),
        (
            {
                "model": "m",
                "answers": {
                    **{k: {"type": "noul", "noul": v} for k, v in SIG.items()},
                    "physical_threat": {"type": "noul", "noul": 1.2},
                },
            },
            "invalid_probability",
        ),
        (
            {
                "model": "m",
                "answers": {
                    **{k: {"type": "noul", "noul": v} for k, v in SIG.items()},
                    "physical_threat": {"type": "noul", "noul": "high"},
                },
            },
            "invalid_probability",
        ),
        (
            {
                "model": "m",
                "answers": {
                    **{k: {"type": "noul", "noul": v} for k, v in SIG.items()},
                    "physical_threat": {"type": "noul"},
                },
            },
            "invalid_probability",
        ),
        (
            {
                "model": "m",
                "answers": {
                    **{k: {"type": "noul", "noul": v} for k, v in SIG.items()},
                    "physical_threat": {"type": "choice", "choice": "a"},
                },
            },
            "invalid_response",
        ),
        ({"answers": {}}, "invalid_response"),
        ({"model": "m"}, "invalid_response"),
        ([1, 2], "invalid_response"),
    ],
)
@respx.mock
def test_invalid_responses_never_fabricate(questions, body, code):
    respx.post(URL).mock(return_value=httpx.Response(200, json=body))
    res = provider(questions).detect(ITEM)
    assert res.status.value == "failure" and res.signals is None and res.error["code"] == code


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "authentication_failed"),
        (403, "permission_denied"),
        (404, "unsupported_model"),
        (413, "payload_too_large"),
        (400, "bad_request"),
        (415, "bad_request"),
    ],
)
@respx.mock
def test_http_errors_not_retried(questions, status, code):
    route = respx.post(URL).mock(return_value=httpx.Response(status, json={"detail": "nope"}))
    res = provider(questions).detect(ITEM)
    assert res.error["code"] == code and route.call_count == 1


@respx.mock
def test_422_on_model_is_unsupported_model(questions):
    respx.post(URL).mock(
        return_value=httpx.Response(
            422,
            json={
                "detail": [
                    {"loc": ["body", "model"], "msg": "unknown model", "type": "value_error"}
                ]
            },
        )
    )
    assert provider(questions, model="jev-nope").detect(ITEM).error["code"] == "unsupported_model"


@respx.mock
def test_422_other_is_validation_error(questions):
    respx.post(URL).mock(
        return_value=httpx.Response(
            422,
            json={
                "detail": [{"loc": ["body", "questions", "x"], "msg": "bad", "type": "value_error"}]
            },
        )
    )
    assert provider(questions).detect(ITEM).error["code"] == "validation_error"


@pytest.mark.parametrize("status", [429, 529, 500, 503])
@respx.mock
def test_retryable_then_success(questions, status):
    route = respx.post(URL).mock(
        side_effect=[
            httpx.Response(status, headers={"retry-after": "0"}),
            httpx.Response(200, json=jev_response(SIG)),
        ]
    )
    res = provider(questions).detect(ITEM)
    assert (
        res.status.value == "success"
        and route.call_count == 2
        and res.provider_usage["attempts"] == 2
    )


@respx.mock
def test_retry_exhaustion(questions):
    route = respx.post(URL).mock(return_value=httpx.Response(429))
    res = provider(questions).detect(ITEM)
    assert res.error["code"] == "rate_limited" and route.call_count == 3


@respx.mock
def test_timeout_and_network(questions):
    respx.post(URL).mock(side_effect=httpx.ReadTimeout("slow"))
    assert provider(questions).detect(ITEM).error["code"] == "timeout"
    respx.post(URL).mock(side_effect=httpx.ConnectError("down"))
    assert provider(questions).detect(ITEM).error["code"] == "network_error"


@respx.mock
def test_non_json_body(questions):
    respx.post(URL).mock(return_value=httpx.Response(200, content=b"<html>oops"))
    assert provider(questions).detect(ITEM).error["code"] == "invalid_response"


def test_missing_credentials_no_request(questions):
    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(URL)
        res = JevProvider(None, questions).detect(ITEM)
        assert res.error["code"] == "missing_credentials" and route.call_count == 0


@respx.mock
def test_list_models(questions):
    respx.get("https://api.typesafe.ai/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [{"name": "jev-latest", "description": "d", "release_date": "2026-10-14"}]
            },
        )
    )
    assert provider(questions).list_models()[0]["name"] == "jev-latest"
    respx.get("https://api.typesafe.ai/v1/models").mock(return_value=httpx.Response(401))
    with pytest.raises(ProviderError):
        provider(questions).list_models()
