import pytest

from harm_detection.security.redaction import REDACTED, redact, redact_text
from harm_detection.settings import AppMode, load_settings


def test_default_mode_offline():
    s = load_settings({}, use_dotenv=False)
    assert s.app_mode == AppMode.LOCAL_OFFLINE
    assert not s.live_calls_allowed and s.writes_allowed


def test_invalid_mode():
    with pytest.raises(ValueError):
        load_settings({"APP_MODE": "production"}, use_dotenv=False)


def test_public_demo_ignores_credentials():
    s = load_settings(
        {
            "APP_MODE": "public_demo",
            "TYPESAFE_API_KEY": "tk-secret-123456",
            "OPENAI_API_KEY": "sk-secret",
        },
        use_dotenv=False,
    )
    assert s.typesafe_api_key is None and s.openai_api_key is None
    assert not s.writes_allowed and not s.live_calls_allowed and not s.jev_available


def test_offline_mode_disables_live_even_with_keys():
    s = load_settings({"TYPESAFE_API_KEY": "tk-123456789"}, use_dotenv=False)
    assert s.credential_status()["TYPESAFE_API_KEY"] and not s.jev_available


def test_repr_hides_secrets():
    s = load_settings(
        {"APP_MODE": "local_live", "TYPESAFE_API_KEY": "tk-very-secret-key"}, use_dotenv=False
    )
    assert "tk-very-secret-key" not in repr(s)


def test_redaction():
    obj = {
        "Authorization": "Bearer abcdefghijkl",
        "nested": [{"api_key": "x"}, "token sk-abcdefghijklmnopqrstu"],
        "msg": "failed with my-secret-value-1",
    }
    out = redact(obj, ["my-secret-value-1"])
    assert out["Authorization"] == REDACTED and out["nested"][0]["api_key"] == REDACTED
    assert "sk-abc" not in out["nested"][1] and "my-secret-value-1" not in out["msg"]
    assert redact_text("Bearer abcdefghijklmnop") == REDACTED
