"""Offline service composition, bounded token refresh and named-service composition."""

import io
import json
from datetime import UTC, datetime
from urllib.parse import parse_qs

import pytest
from orsia_data.http import JsonSession
from orsia_data.services.cdse import CdseAuth, CdseClient
from orsia_data.services.cdse import client as cdse_client
from orsia_data.services.earthdata import EarthdataAuth, EarthdataClient
from orsia_data.services.earthdata import client as earthdata_client


def transport(replies, calls):
    responses = iter(replies)

    def open_url(request, timeout):
        calls.append(request)
        reply = next(responses)
        if isinstance(reply, Exception):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())

    return JsonSession(open_url=open_url)


def test_cdse_login_reuse_refresh_rotation_and_secret_repr():
    now, calls = [100.0], []
    session = transport(
        [
            {"access_token": "one", "refresh_token": "refresh-one", "expires_in": 60},
            {"access_token": "two", "refresh_token": "refresh-two", "expires_in": 90},
            {"access_token": "three", "expires_in": 100},
            {},
        ],
        calls,
    )
    with CdseAuth.login(
        "user",
        "password",
        totp="123456",
        session=session,
        clock=lambda: now[0],
        refresh_margin_s=10,
    ) as auth:
        assert auth.get_token() == auth.get_token() == "one"
        assert len(calls) == 1
        grant = parse_qs(calls[0].data.decode())
        assert grant == {
            "client_id": ["cdse-public"],
            "grant_type": ["password"],
            "username": ["user"],
            "password": ["password"],
            "totp": ["123456"],
        }
        assert calls[0].get_header("Authorization") is None
        now[0] = 151
        assert auth.get_token() == "two"
        assert parse_qs(calls[1].data.decode())["refresh_token"] == ["refresh-one"]
        now[0] = 241
        assert auth.get_token() == "three"
        assert parse_qs(calls[2].data.decode())["refresh_token"] == ["refresh-two"]
        assert all(secret not in repr(auth) for secret in ("password", "refresh-two", "three"))
    with pytest.raises(RuntimeError, match="closed"):
        auth.get_token()
    with pytest.raises(RuntimeError):
        auth.__enter__()
    assert session.request("https://example.test/") == {}  # Borrowed, not closed.


def test_cdse_expiry_without_refresh_and_single_failure():
    now = [0.0]
    auth = CdseAuth("short", expires_in=10, clock=lambda: now[0])
    assert auth.get_token() == "short"
    now[0] = 9
    assert auth.get_token() == "short"  # No refresh token: use the whole valid lifetime.
    now[0] = 11
    with pytest.raises(RuntimeError, match="login required"):
        auth.get_token()
    auth.close()
    calls = []
    session = transport([ValueError("response exposing secret")], calls)
    auth = CdseAuth(
        "old", expires_in=1, refresh_token="refresh", session=session, clock=lambda: now[0]
    )
    now[0] += 2
    with pytest.raises(RuntimeError, match="exchange failed") as error:
        auth.get_token()
    assert "secret" not in str(error.value) and len(calls) == 1


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"access_token": "bad token", "expires_in": 60},
        {"access_token": "one", "expires_in": 0},
        {"access_token": "one", "expires_in": "nan"},
        {"access_token": "one", "expires_in": 30, "refresh_token": ""},
    ],
)
def test_malformed_token_response_is_sanitized(reply):
    with pytest.raises(RuntimeError, match="exchange failed"):
        CdseAuth.login("user", "password", session=transport([reply], []))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expires_in": 0},
        {"expires_in": float("inf")},
        {"expires_in": 30, "refresh_margin_s": -1},
        {"expires_in": 30, "refresh_token": "bad token"},
    ],
)
def test_invalid_token_lifetimes(kwargs):
    with pytest.raises(ValueError):
        CdseAuth("value", **kwargs)


def test_invalid_login_and_owned_auth_session(monkeypatch):
    for username, password in (("", "x"), ("x", "")):
        with pytest.raises(ValueError):
            CdseAuth.login(username, password)
    with pytest.raises(ValueError):
        CdseAuth.login("a", "b", refresh_margin_s=-1)
    from orsia_data.services.cdse import auth as module

    owned = transport([{"access_token": "one", "expires_in": 60}], [])
    monkeypatch.setattr(module, "JsonSession", lambda **kw: owned)
    with CdseAuth.login("user", "password") as auth:
        assert auth.get_token() == "one"
    with pytest.raises(RuntimeError, match="closed"):
        owned.request("https://example.test")
    failed = transport([ValueError("failure")], [])
    monkeypatch.setattr(module, "JsonSession", lambda **kw: failed)
    with pytest.raises(RuntimeError):
        CdseAuth.login("user", "password")
    with pytest.raises(RuntimeError, match="closed"):
        failed.request("https://example.test")


def test_cdse_client_catalogue_stac_and_download(monkeypatch, tmp_path):
    requests, downloads = [], []
    session = transport([{"value": []}, {"conformsTo": ["native"]}, {}], requests)
    auth = CdseAuth("my-token", expires_in=60)
    monkeypatch.setattr(
        cdse_client, "download_cdse", lambda *a, **kw: downloads.append((a, kw)) or a[1]
    )
    client = CdseClient(auth=auth, session=session)
    with client:
        assert client.search_sentinel2_l2a(
            bbox=(0, 0, 1, 1),
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 2, tzinfo=UTC),
        ) == {"value": []}
        assert client.stac.landing_page()["conformsTo"] == ["native"]
        assert client.download("uuid", tmp_path / "out", max_bytes=100) == tmp_path / "out"
    assert downloads[0][1]["access_token"] == "my-token"
    assert all(request.get_header("Authorization") is None for request in requests)
    assert auth.get_token() == "my-token"  # Borrowed auth remains open.
    assert session.request("https://example.test/") == {}
    with pytest.raises(RuntimeError):
        client.download("uuid", tmp_path / "out", max_bytes=100)
    with pytest.raises(RuntimeError):
        client.__enter__()
    with pytest.raises(RuntimeError):
        client.stac.landing_page()
    with CdseClient() as anonymous:
        with pytest.raises(ValueError, match="authentication"):
            anonymous.download("uuid", tmp_path / "out", max_bytes=100)


def test_earthdata_supplier_client_and_trust(monkeypatch, tmp_path):
    requests, downloads, tokens = [], [], iter(["first", "second"])
    session = transport([{"feed": {"entry": []}}, {}], requests)
    auth = EarthdataAuth(lambda: next(tokens))
    monkeypatch.setattr(
        earthdata_client, "download_earthdata", lambda *a, **kw: downloads.append((a, kw)) or a[1]
    )
    with EarthdataClient(auth=auth, trusted_hosts=("data.test",), session=session) as client:
        assert client.search(product="AG100.003", bbox=(0, 0, 1, 1), temporal=None) == {
            "feed": {"entry": []}
        }
        for name in ("one", "two"):
            assert (
                client.download("https://data.test/a", tmp_path / name, max_bytes=100)
                == tmp_path / name
            )
    assert [call[1]["access_token"] for call in downloads] == ["first", "second"]
    assert all(call[1]["trusted_hosts"] == ("data.test",) for call in downloads)
    assert session.request("https://example.test/") == {}
    assert requests[0].get_header("Authorization") is None
    with pytest.raises(RuntimeError):
        client.__enter__()
    with EarthdataClient() as anonymous:
        with pytest.raises(ValueError):
            anonymous.download("https://data.test/a", tmp_path / "out", max_bytes=10)
    with EarthdataAuth("static-token") as static:
        assert static.get_token() == "static-token"
        assert "static-token" not in repr(static)
    with pytest.raises(RuntimeError):
        static.get_token()
    with pytest.raises(RuntimeError):
        static.__enter__()
    with pytest.raises(ValueError):
        EarthdataAuth("bad token").get_token()
