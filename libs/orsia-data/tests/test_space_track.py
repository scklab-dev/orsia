"""Mocked Space-Track transport policies, never an authenticated live-provider check."""

import io
import json
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest
from orsia_data.services.space_track import SpaceTrackClient
from test_orbits import END, EPOCH, OMM, tle_payload


class Response(io.BytesIO):
    def __init__(self, payload, url):
        super().__init__(payload)
        self.url = url

    def geturl(self):
        return self.url


class Transport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.now = 0.0
        self.waits = []

    def __call__(self, request, *, timeout):
        self.calls.append(request)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return Response(value, request.full_url)

    def clock(self):
        return self.now

    def sleep(self, delay):
        self.waits.append(delay)
        self.now += delay

    def client(self, **kwargs):
        return SpaceTrackClient(
            lambda: ("secret-identity", "secret-password"),
            open_url=self,
            clock=self.clock,
            sleep=self.sleep,
            **kwargs,
        )


def error(status, headers=None):
    return HTTPError(
        "https://www.space-track.org/private",
        status,
        "secret-password",
        headers or {},
        io.BytesIO(b"secret-password"),
    )


def test_download_bundles_ids_and_keeps_payload_receipt_free_of_credentials(tmp_path):
    native = json.dumps([OMM, dict(OMM, NORAD_CAT_ID=123)]).encode()
    transport = Transport(b"{}", native)
    path = tmp_path / "native.json"
    with transport.client() as client:
        assert client.download((25544, 123), path) == path
        assert "secret" not in repr(client)
        with pytest.raises(FileExistsError):
            client.download((25544, 123), path)
    assert path.read_bytes() == native
    assert len(transport.calls) == 2
    login, query = transport.calls
    assert login.full_url == "https://www.space-track.org/ajaxauth/login"
    assert b"identity=secret-identity" in login.data
    assert query.data is None
    assert "/NORAD_CAT_ID/123,25544/" in query.full_url and query.full_url.endswith("/format/json")
    receipt_text = path.with_name(path.name + ".provenance.json").read_text()
    receipt = json.loads(receipt_text)
    assert (
        receipt["provider"] == "Space-Track"
        and receipt["element_epochs"] == [EPOCH.isoformat()] * 2
    )
    assert "secret-" not in receipt_text and "password" not in receipt_text
    assert not client._cookies and client._credentials is None and client._open is None
    with pytest.raises(RuntimeError, match="closed"):
        client.download((25544,), tmp_path / "other")
    with pytest.raises(RuntimeError, match="closed"):
        client.__enter__()


def test_gp_hourly_spacing_and_history_once_per_selection(tmp_path):
    transport = Transport(b"{}", json.dumps(OMM).encode(), json.dumps(OMM).encode())
    with transport.client() as client:
        client.download((25544,), tmp_path / "first")
        with pytest.raises(RuntimeError, match="waiting budget"):
            client.download((25544,), tmp_path / "second")
        client.download((25544,), tmp_path / "history", history_utc=(EPOCH, END))
        with pytest.raises(RuntimeError, match="already retrieved"):
            client.download((25544,), tmp_path / "history-again", history_utc=(EPOCH, END))
    assert "/class/gp_history/" in transport.calls[-1].full_url
    assert "2026-01-01T00:00:00--2026-01-02T00:00:00" in transport.calls[-1].full_url


def test_tle_download_and_no_implicit_login_for_local_reuse(tmp_path):
    transport = Transport(b"", tle_payload())
    with transport.client() as client:
        client.download((25544,), tmp_path / "native.tle", format="tle")
    assert transport.calls[-1].full_url.endswith("/format/tle")


@pytest.mark.parametrize("reply", [b"", b"{}", b'""', b'  ""\n'])
def test_empty_login_success_responses_allow_the_element_query(tmp_path, reply):
    transport = Transport(reply, json.dumps(OMM).encode())
    with transport.client() as client:
        client.download((25544,), tmp_path / "native.json")
    assert len(transport.calls) == 2
    assert "/class/gp/" in transport.calls[-1].full_url


@pytest.mark.parametrize(
    "reply",
    [
        b'{"Login":"Failed"}',
        b'{"unexpected":"value"}',
        b'"unexpected"',
        b"[]",
        b"null",
        b"false",
        b"0",
        b"login html",
        b"<html>login</html>",
        error(401),
        error(403),
    ],
)
def test_authentication_failures_are_redacted_and_write_nothing(tmp_path, reply):
    transport = Transport(reply)
    with (
        transport.client() as client,
        pytest.raises(RuntimeError, match="authentication") as caught,
    ):
        client.download((25544,), tmp_path / "native")
    assert "secret" not in str(caught.value)
    assert len(transport.calls) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "reply",
    [
        error(302),
        error(404),
        error(429),
        error(503),
        URLError("secret-password"),
        TimeoutError("secret-password"),
    ],
)
def test_service_failures_are_explicit_redacted_and_bounded(tmp_path, reply):
    transport = Transport(b"{}", reply)
    with transport.client() as client, pytest.raises(RuntimeError) as caught:
        client.download((25544,), tmp_path / "native")
    assert "secret" not in str(caught.value) and len(transport.calls) == 2
    assert list(tmp_path.iterdir()) == []


def test_retry_after_and_gp_retry_respect_wait_bound(tmp_path):
    transport = Transport(error(429, {"Retry-After": "4"}), b"{}", json.dumps(OMM).encode())
    with transport.client(attempts=2, max_wait_s=5) as client:
        client.download((25544,), tmp_path / "native")
    assert 4 in transport.waits
    transport = Transport(b"{}", error(503), json.dumps(OMM).encode())
    with transport.client(attempts=2, max_wait_s=5) as client:
        with pytest.raises(RuntimeError, match="waiting budget"):
            client.download((25544,), tmp_path / "failed")
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    "payload", [b"[]", b"login html", json.dumps(dict(OMM, NORAD_CAT_ID=1)).encode()]
)
def test_empty_invalid_unrequested_elements_not_published(tmp_path, payload):
    transport = Transport(b"{}", payload)
    with transport.client() as client, pytest.raises(ValueError):
        client.download((25544,), tmp_path / "native")
    assert list(tmp_path.iterdir()) == []


def test_missing_objects_invalid_selection_and_existing_receipt(tmp_path):
    transport = Transport(b"{}", json.dumps(OMM).encode())
    with transport.client() as client:
        with pytest.raises(ValueError, match="incomplete"):
            client.download((25544, 999), tmp_path / "native")
        for ids in ((), (1, 1), (0,)):
            with pytest.raises(ValueError, match="identifiers"):
                client.download(ids, tmp_path / "invalid")
        with pytest.raises(ValueError, match="format"):
            client.download((25544,), tmp_path / "invalid", format=cast(Any, "xml"))
        for interval in (
            (END, EPOCH),
            (EPOCH.replace(tzinfo=None), END),
            (EPOCH.replace(microsecond=1), END),
        ):
            with pytest.raises(ValueError, match="history"):
                client.download((25544,), tmp_path / "invalid", history_utc=interval)
        (tmp_path / "exists.provenance.json").write_text("{}")
        with pytest.raises(FileExistsError):
            client.download((25544,), tmp_path / "exists")


def test_request_bounds_credentials_and_response_limits(tmp_path):
    cases: list[dict[str, Any]] = [
        {"timeout_s": 0},
        {"max_bytes": 0},
        {"attempts": 4},
        {"max_wait_s": -1},
    ]
    for kwargs in cases:
        with pytest.raises(ValueError, match="bounds"):
            SpaceTrackClient(lambda: ("a", "b"), **kwargs)
    transport = Transport(b"1234")
    with transport.client(max_bytes=3) as client, pytest.raises(ValueError, match="byte bound"):
        client.download((25544,), tmp_path / "oversize")
    with SpaceTrackClient(lambda: ("", "")) as client, pytest.raises(ValueError, match="identity"):
        client.download((25544,), tmp_path / "empty")

    def failed():
        raise RuntimeError("secret-password")

    with (
        SpaceTrackClient(failed) as client,
        pytest.raises(RuntimeError, match="supplier") as caught,
    ):
        client.download((25544,), tmp_path / "failed")
    assert "secret" not in str(caught.value)


def test_default_transport_refuses_redirect_and_clears_cookie_state():
    from orsia_data.services.space_track import _NoRedirect

    with SpaceTrackClient(lambda: ("unused", "unused")) as client:
        assert client._open is not None
    assert (
        _NoRedirect().redirect_request(
            Request("https://www.space-track.org"), None, 302, "", {}, "https://evil.example"
        )
        is None
    )


def test_minute_and_hour_request_limits_use_bounded_wait():
    transport = Transport(b"{}")
    with transport.client(max_wait_s=1) as client:
        client._requests = [0.0] * 29
        with pytest.raises(RuntimeError, match="waiting budget"):
            client._login()
        client._requests = [0.0] * 299
        with pytest.raises(RuntimeError, match="waiting budget"):
            client._login()
    assert not transport.calls


def test_broken_satkit_installation_fails_before_credentials_or_network(tmp_path, monkeypatch):
    import sys

    transport = Transport()

    def credentials():
        pytest.fail("credentials must not be requested after a required-parser import failure")

    # Simulate a broken required installation, preserving the pre-authentication boundary.
    monkeypatch.setitem(sys.modules, "satkit", None)
    with SpaceTrackClient(credentials, open_url=transport) as client:
        with pytest.raises(ImportError):
            client.download((25544,), tmp_path / "native")
    assert not transport.calls and list(tmp_path.iterdir()) == []
