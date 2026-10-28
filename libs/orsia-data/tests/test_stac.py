"""Offline native STAC pages, explicit traversal and credential confinement."""

import io
import json
from email.message import Message
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest
from orsia_data.http import JsonSession, _NoRedirect
from orsia_data.stac import StacClient, asset_href, read_document, select_assets

ROOT = "https://catalogue.test/v1/"
EMPTY = {"type": "FeatureCollection", "features": [], "links": []}


def session_for(pages, calls, **kwargs):
    replies = iter(pages)

    def open_url(request, timeout):
        calls.append(request)
        value = next(replies)
        return io.BytesIO(value if isinstance(value, bytes) else json.dumps(value).encode())

    return JsonSession(open_url=open_url, **kwargs)


def landing(method="GET"):
    return {
        "conformsTo": ["https://api.stacspec.org/v1.0.0/item-search"],
        "links": [{"rel": "search", "href": "search", "method": method}],
    }


def test_get_pages_relative_links_metadata_and_assets():
    calls = []
    item = {
        "id": "native",
        "assets": {
            "B01": {
                "href": "data/a.tif",
                "type": "image/tiff",
                "roles": ["data"],
                "vendor:extension": {"x": 1},
            },
            "preview": {"href": "https://assets.test/preview", "roles": ["thumbnail"]},
        },
    }
    page = EMPTY | {
        "features": [item],
        "vendor:quality": [1],
        "links": [{"rel": "next", "href": "?page=2"}],
    }
    session = session_for([landing(), page, EMPTY], calls)
    with StacClient(ROOT, session=session) as client:
        parameters = {
            "bbox": [1, 2, 3, 4],
            "collections": ["a", "b"],
            "limit": 2,
            "query": {"eo:cloud_cover": {"lt": 10}},
        }
        pages = list(client.pages(parameters, max_pages=4))
    assert pages == [page, EMPTY]
    assert calls[2].full_url == ROOT + "search?page=2"
    query = parse_qs(urlsplit(calls[1].full_url).query)
    assert query["bbox"] == ["1,2,3,4"] and query["collections"] == ["a,b"]
    assert json.loads(query["query"][0]) == parameters["query"]
    selected = select_assets(item, role="data", media_type="image/tiff")
    assert list(selected) == ["B01"]
    selected["B01"]["vendor:extension"]["x"] = 7
    assert item["assets"]["B01"]["vendor:extension"]["x"] == 1
    assert asset_href(item, "B01", document_url=ROOT + "item.json") == ROOT + "data/a.tif"
    assert select_assets({}) == {}
    with pytest.raises(RuntimeError, match="closed"):
        client.landing_page()


def test_post_merge_replacement_page_cap_and_no_input_mutation():
    calls = []
    page1 = EMPTY | {
        "links": [
            {
                "rel": "next",
                "href": "search",
                "method": "POST",
                "body": {"token": "p2"},
                "merge": True,
            }
        ]
    }
    page2 = EMPTY | {
        "links": [{"rel": "next", "href": "search", "method": "POST", "body": {"only": "cursor"}}]
    }
    session = session_for([landing("POST"), page1, page2, EMPTY], calls)
    parameters = {"collections": ["native"], "limit": 1}
    client = StacClient(ROOT, session=session)
    iterator = client.pages(parameters, method="POST", max_pages=3)
    first = next(iterator)
    # Caller mutation of a yielded page cannot alter the retained continuation.
    first["links"][0]["body"]["token"] = "changed"
    assert len(list(iterator)) == 2
    assert json.loads(calls[1].data) == parameters
    assert json.loads(calls[2].data) == parameters | {"token": "p2"}
    assert json.loads(calls[3].data) == {"only": "cursor"}
    assert parameters == {"collections": ["native"], "limit": 1}
    calls = []
    client = StacClient(ROOT, session=session_for([landing(), page1], calls))
    assert client.search({}) == page1
    assert len(calls) == 2  # No prefetch, even though a next link exists.


@pytest.mark.parametrize(
    "link",
    [
        {"rel": "next", "href": "search?", "method": "GET"},
        {"rel": "next", "href": "http://elsewhere.test/a"},
        {"rel": "next", "href": "https://user:secret@elsewhere.test/a"},
        {"rel": "next", "href": "next", "headers": {"Authorization": "stolen"}},
        {"rel": "next", "href": "next", "headers": {"Host": "elsewhere.test"}},
        {"rel": "next", "href": "next", "method": "DELETE"},
    ],
)
def test_unsafe_or_repeated_pagination_fails(link):
    calls = []
    page = EMPTY | {"links": [link]}
    client = StacClient(ROOT, session=session_for([landing(), page, page], calls))
    with pytest.raises(ValueError):
        list(client.pages({}, max_pages=5))


def test_empty_capabilities_invalid_pages_and_ambiguous_links():
    for document in ({}, {"links": [*landing()["links"], *landing()["links"]]}):
        client = StacClient(ROOT, session=session_for([document], []))
        with pytest.raises(ValueError, match="advertise"):
            client.search({})
    for page in ({}, EMPTY | {"features": None}):
        with pytest.raises(ValueError, match="ItemCollection"):
            StacClient(ROOT, session=session_for([landing(), page], [])).search({})
    page = EMPTY | {"links": [{"rel": "next", "href": "a"}, {"rel": "next", "href": "b"}]}
    client = StacClient(ROOT, session=session_for([landing(), page], []))
    with pytest.raises(ValueError, match="ambiguous"):
        list(client.pages({}, max_pages=2))
    for max_pages, method in ((0, "GET"), (1, "PUT")):
        with pytest.raises(ValueError):
            list(client.pages({}, max_pages=max_pages, method=method))
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.search({"bbox": [], "intersects": {}})


def test_search_ignores_other_advertised_media_types():
    root = landing()
    root["links"].append({"rel": "search", "href": "html", "type": "text/html"})
    calls = []
    client = StacClient(ROOT, session=session_for([root, EMPTY], calls))
    assert client.search({}) == EMPTY
    assert "/html" not in calls[-1].full_url


def test_json_session_trust_bounds_lifecycle_and_no_credentials_lookup(monkeypatch):
    monkeypatch.setenv("CDSE_PASSWORD", "not-to-be-read")
    calls, token_calls = [], []
    session = session_for(
        [{}, {}, {}],
        calls,
        token=lambda: token_calls.append(1) or "secret",
        trusted_hosts=("catalogue.test",),
    )
    with session:
        session.request(ROOT)
        session.request("https://catalogue.test.attacker.test/path")
        session.request("https://elsewhere.test/path")
    assert token_calls == [1]
    assert calls[0].get_header("Authorization") == "Bearer secret"
    assert all(call.get_header("Authorization") is None for call in calls[1:])
    assert "secret" not in repr(session)
    with pytest.raises(RuntimeError, match="closed"):
        session.request(ROOT)
    with pytest.raises(RuntimeError):
        session.__enter__()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_s": 0},
        {"timeout_s": float("nan")},
        {"max_bytes": 0},
        {"token": lambda: "token"},
        {"trusted_hosts": ("*.test",)},
    ],
)
def test_invalid_session_policy(kwargs):
    with pytest.raises(ValueError):
        JsonSession(**kwargs)


@pytest.mark.parametrize(
    "url,kwargs",
    [
        ("http://example.test/", {}),
        ("https://example.test:444/", {}),
        ("https://user:pw@example.test/", {}),
        ("https:///path", {}),
        (ROOT, {"method": "GET", "body": {}}),
        (ROOT, {"method": "POST", "form": b"a=b", "body": {}}),
        (ROOT, {"headers": {"Accept": "json\r\ninjected"}}),
    ],
)
def test_invalid_json_request(url, kwargs):
    session = session_for([], [])
    with pytest.raises(ValueError):
        session.request(url, **kwargs)


@pytest.mark.parametrize("content,max_bytes", [(b"[]", 10), (b"bad", 10), (b"{}", 1)])
def test_malformed_and_oversize_json(content, max_bytes):
    with pytest.raises(ValueError):
        session_for([content], [], max_bytes=max_bytes).request(ROOT)


def test_json_errors_redirects_and_borrowed_session():
    calls = []
    session = session_for([{}, {}], calls)
    client = StacClient(ROOT, session=session)
    client.close()
    assert session.request(ROOT) == {}  # Borrowed resource remains usable.
    with pytest.raises(RuntimeError):
        client.__enter__()
    owned = StacClient(ROOT)
    owned.close()
    assert (
        _NoRedirect().redirect_request(Request(ROOT), None, 302, "", {}, "https://other.test")
        is None
    )

    def denied(*args, **kwargs):
        raise HTTPError(ROOT, 401, "denied", Message(), None)

    with pytest.raises(HTTPError):
        JsonSession(open_url=denied).request(ROOT)
    with pytest.raises(ValueError, match="token"):
        session_for([], [], token=lambda: "bad token", trusted_hosts=("catalogue.test",)).request(
            ROOT
        )


@pytest.mark.parametrize("kind", ["Catalog", "Collection", "Feature", "FeatureCollection"])
def test_local_stac_documents(tmp_path, kind):
    path = tmp_path / "stac.json"
    document = {
        "type": kind,
        "stac_version": "1.1.0",
        "features": [],
        "links": [],
        "vendor:anything": {"nested": [1]},
    }
    path.write_text(json.dumps(document))
    assert read_document(path) == document
    with pytest.raises(ValueError, match="bound"):
        read_document(path, max_bytes=2)
    with pytest.raises(ValueError, match="bound"):
        read_document(path, max_bytes=0)


@pytest.mark.parametrize(
    "document",
    [
        [],
        {},
        {"type": "Catalog"},
        {"type": "FeatureCollection"},
        {"type": "Catalog", "links": "invalid"},
    ],
)
def test_invalid_local_document(tmp_path, document):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        read_document(path)
