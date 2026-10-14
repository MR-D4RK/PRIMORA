"""Phase 3 Step 1: the secure page fetcher, exercised without the network.

Every test injects a fake DNS resolver and/or a fake transport, so the whole
security policy - scheme blocking, address blocking, DNS-rebinding pinning,
redirect validation, size and time caps, header hygiene, HTML extraction, and
sentence selection - runs with zero live network access. No test in this file
may reach the network, the database, or the model.
"""

import ast
import gzip
import http.server
import ipaddress
import os
import socket
import ssl
import sys
import threading
import zlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.research.fetcher import (
    ACCEPT_HEADER,
    MAX_EXTRACTED_CHARS,
    MAX_RESPONSE_BYTES,
    REQUEST_TIMEOUT_S,
    STATUS_BLOCKED_ADDRESS,
    STATUS_BLOCKED_SCHEME,
    STATUS_BLOCKED_USERINFO,
    STATUS_FETCH_ERROR,
    STATUS_HTTP_ERROR,
    STATUS_OK,
    STATUS_TIMEOUT,
    STATUS_TOO_LARGE,
    STATUS_TOO_MANY_REDIRECTS,
    STATUS_UNSUPPORTED_CONTENT,
    USER_AGENT,
    FetchResult,
    PageFetcher,
    _default_transport,
    _destination_allowed,
    _html_to_text,
    _is_blocked_ip,
    _select_sentences,
    _try_ip,
)


# ======================================================================
# Fakes: DNS resolver and transport, both fully scripted
# ======================================================================


def ips(*strings):
    return [ipaddress.ip_address(s) for s in strings]


PUBLIC = ips("93.184.216.34")[0]
PUBLIC_V6 = ips("2606:4700:4700::1111")[0]


class _FakeResponse:
    """Minimal response: status, headers, and an incrementally readable body."""

    def __init__(self, status=200, headers=None, body=b""):
        self.status = status
        self.headers = dict(headers or {})
        self._body = body
        self._pos = 0
        self.closed = False

    def read(self, size=-1):
        if size is None or size < 0:
            chunk = self._body[self._pos:]
            self._pos = len(self._body)
            return chunk
        chunk = self._body[self._pos:self._pos + size]
        self._pos += len(chunk)
        return chunk

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class _FakeTransport:
    """Records every call and returns scripted responses in order."""

    def __init__(self, *responses, error=None):
        self.calls = []
        self._responses = list(responses)
        self._error = error

    def __call__(self, host, port, path, *, use_tls, server_hostname,
                 pinned_ip, timeout, headers):
        self.calls.append({
            "host": host,
            "port": port,
            "path": path,
            "use_tls": use_tls,
            "server_hostname": server_hostname,
            "pinned_ip": pinned_ip,
            "timeout": timeout,
            "headers": dict(headers),
        })
        if self._error is not None:
            raise self._error
        if not self._responses:
            raise AssertionError("fake transport ran out of responses")
        return self._responses.pop(0)


class _FakeResolver:
    """Returns scripted address lists; records every host it is asked about.

    IP literals short-circuit to themselves, exactly like the real
    ``_default_resolver`` - so a URL like ``http://127.0.0.1/`` exercises
    the address policy on the literal itself, not on a scripted answer.
    """

    def __init__(self, *answers):
        self.calls = []
        self._answers = list(answers)

    def __call__(self, host):
        self.calls.append(host)
        literal = _try_ip(host)
        if literal is not None:
            return [literal]
        if not self._answers:
            raise AssertionError("fake resolver ran out of answers")
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return list(answer)


def _html_page(paragraphs, *, script="", style="", nav="", footer=""):
    parts = [
        "<!DOCTYPE html><html><head><title>Doc</title>",
        f"<style>{style}</style><script>{script}</script></head><body>",
        f"<nav>{nav}</nav>",
    ]
    for para in paragraphs:
        parts.append(f"<p>{para}</p>")
    parts.append(f"<footer>{footer}</footer>")
    parts.append("</body></html>")
    return "".join(parts).encode("utf-8")


def _ok_page(text="Python 3.13 is the latest stable release."):
    return _html_page([text])


def _html_response(body=None, *, text="Python 3.13 is the latest stable release.",
                   headers=None, **kwargs):
    """A 200 response that looks like a real HTML page (Content-Type set)."""
    if body is None:
        body = _ok_page(text)
    if headers is None:
        headers = {"Content-Type": "text/html; charset=utf-8"}
    return _FakeResponse(headers=headers, body=body, **kwargs)


def _fetch(url, *, resolver_answers=(ips("93.184.216.34"),),
           responses=None, query="", **kwargs):
    resolver = _FakeResolver(*resolver_answers)
    if responses is None:
        responses = (_html_response(),)
    transport = _FakeTransport(*responses)
    fetcher = PageFetcher(resolver=resolver, transport=transport, **kwargs)
    return fetcher, resolver, transport, fetcher.fetch(url, query=query)


# ======================================================================
# Construction: configuration only, no I/O
# ======================================================================


class TestConstructionIsIOFree:
    def test_construction_performs_no_io(self, monkeypatch):
        def _no_io(*args, **kwargs):
            raise AssertionError("fetcher construction reached the network")

        monkeypatch.setattr("socket.getaddrinfo", _no_io)
        monkeypatch.setattr("socket.create_connection", _no_io)
        fetcher = PageFetcher()
        assert fetcher.timeout_s == REQUEST_TIMEOUT_S

    def test_the_default_timeout_is_the_documented_ceiling(self):
        assert PageFetcher().timeout_s == 5.0

    def test_an_explicit_timeout_is_honoured(self):
        assert PageFetcher(timeout_s=2.0).timeout_s == 2.0


# ======================================================================
# Scheme rejection
# ======================================================================


class TestSchemeRejection:
    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "ftp://example.com/x",
        "javascript:alert(1)",
        "data:text/html,<script>x</script>",
        "gopher://example.com/",
        "FILE:///etc/passwd",
        "JAVASCRIPT:alert(1)",
        "ws://example.com/",
        "wss://example.com/",
    ])
    def test_non_http_schemes_are_blocked_before_any_io(self, url):
        fetcher, resolver, transport, result = _fetch(url)
        assert result.status == STATUS_BLOCKED_SCHEME
        assert resolver.calls == []
        assert transport.calls == []

    def test_a_schemeless_url_is_blocked(self):
        _, _, _, result = _fetch("example.com/page")
        assert result.status == STATUS_BLOCKED_SCHEME

    def test_the_reason_names_the_allowed_schemes(self):
        _, _, _, result = _fetch("file:///etc/passwd")
        assert "http" in result.error and "https" in result.error


# ======================================================================
# Address policy: unit level
# ======================================================================


class TestAddressPolicy:
    @pytest.mark.parametrize("literal", [
        "127.0.0.1", "127.1.2.3", "10.0.0.1", "172.16.0.1", "172.31.255.255",
        "192.168.0.1", "169.254.1.1", "100.64.0.1", "0.0.0.0", "240.0.0.1",
        "255.255.255.255", "224.0.0.1",
    ])
    def test_blocked_ipv4_literals(self, literal):
        assert _is_blocked_ip(ipaddress.ip_address(literal)) is True

    @pytest.mark.parametrize("literal", [
        "::1", "::", "fe80::1", "fc00::1", "fd00::1", "ff02::1",
        "2001:db8::1",                      # documentation range
        "::ffff:127.0.0.1",                 # IPv4-mapped loopback
        "::ffff:10.0.0.1",                  # IPv4-mapped private
        "::ffff:8.8.8.8",                   # IPv4-mapped (any mapped form)
        "::127.0.0.1",                      # IPv4-compatible loopback
        "64:ff9b::7f00:1",                  # NAT64 loopback
        "64:ff9b:1::7f00:1",                # NAT64 local-use loopback
        "2002:7f00:1::",                    # 6to4 loopback
        "2001:0:7f00:1::",                  # Teredo-form loopback
    ])
    def test_blocked_ipv6_literals(self, literal):
        assert _is_blocked_ip(ipaddress.ip_address(literal)) is True

    @pytest.mark.parametrize("literal", [
        "8.8.8.8", "1.1.1.1", "93.184.216.34",
    ])
    def test_global_ipv4_is_allowed(self, literal):
        assert _is_blocked_ip(ipaddress.ip_address(literal)) is False

    def test_global_ipv6_is_allowed(self):
        assert _is_blocked_ip(PUBLIC_V6) is False

    def test_every_v4_embedding_prefix_is_refused_wholesale(self):
        # Even an embedded *global* IPv4 is refused: the prefix itself is the
        # risk, because it can equally embed a blocked one.
        assert _is_blocked_ip(ipaddress.ip_address("::ffff:8.8.8.8")) is True
        assert _is_blocked_ip(ipaddress.ip_address("64:ff9b::808:808")) is True
        assert _is_blocked_ip(ipaddress.ip_address("2002:808:808::")) is True


class TestUnusualIPLiterals:
    @pytest.mark.parametrize("literal", [
        "2130706433",       # decimal for 127.0.0.1
        "0177.0.0.1",       # octal
        "0x7f.0.0.1",       # hex
        "127.1",            # shortened
        "127.0.1",          # shortened
        "017700000001",     # octal dword
    ])
    def test_they_do_not_parse_as_ip_literals(self, literal):
        # Which is the point: they fall through to DNS, where the address
        # policy is applied to whatever the resolver returns.
        assert _try_ip(literal) is None

    @pytest.mark.parametrize("literal", [
        "2130706433", "0177.0.0.1", "0x7f.0.0.1", "127.1", "127.0.1",
    ])
    def test_a_resolver_mapping_them_to_loopback_is_blocked(self, literal):
        url = f"http://{literal}/"
        _, resolver, transport, result = _fetch(
            url, resolver_answers=(ips("127.0.0.1"),)
        )
        assert result.status == STATUS_BLOCKED_ADDRESS
        assert resolver.calls == [literal]
        assert transport.calls == []

    def test_a_resolver_mapping_one_to_a_global_address_proceeds(self):
        _, resolver, transport, result = _fetch(
            "http://2130706433/", resolver_answers=(ips("93.184.216.34"),)
        )
        assert result.status == STATUS_OK
        assert transport.calls[0]["pinned_ip"] == PUBLIC


# ======================================================================
# Blocked destinations through the fetch path
# ======================================================================


class TestBlockedDestinations:
    @pytest.mark.parametrize("literal", [
        "127.0.0.1", "[::1]", "10.0.0.1", "192.168.1.1", "169.254.1.1",
        "[fc00::1]", "[fe80::1]", "[::ffff:127.0.0.1]", "0.0.0.0",
    ])
    def test_ip_literal_urls_are_blocked_without_reaching_the_transport(
        self, literal
    ):
        _, resolver, transport, result = _fetch(f"http://{literal}/")
        assert result.status == STATUS_BLOCKED_ADDRESS
        # A literal short-circuits DNS to itself; the address policy then
        # refuses it before the transport is ever constructed.
        assert transport.calls == []

    def test_a_hostname_resolving_to_loopback_is_blocked(self):
        _, resolver, transport, result = _fetch(
            "http://evil.test/", resolver_answers=(ips("127.0.0.1"),)
        )
        assert result.status == STATUS_BLOCKED_ADDRESS
        assert transport.calls == []

    def test_a_mixed_public_and_private_answer_is_refused(self):
        # A hostname that resolves to both is the DNS-rebinding signature;
        # refusing the whole answer is the safe call.
        _, resolver, transport, result = _fetch(
            "http://rebind.test/",
            resolver_answers=(ips("93.184.216.34", "127.0.0.1"),),
        )
        assert result.status == STATUS_BLOCKED_ADDRESS
        assert transport.calls == []

    def test_an_ipv6_literal_url_with_a_bracketed_loopback_is_blocked(self):
        _, _, transport, result = _fetch("http://[::1]/")
        assert result.status == STATUS_BLOCKED_ADDRESS
        assert transport.calls == []

    def test_the_error_names_the_offending_address(self):
        _, _, _, result = _fetch(
            "http://evil.test/", resolver_answers=(ips("10.0.0.1"),)
        )
        assert "10.0.0.1" in result.error


# ======================================================================
# The local-proxy (fake-IP) range: HTTPS admitted, HTTP refused
# ======================================================================


class TestLocalProxyRange:
    """The benchmarking range a transparent proxy answers every name with.

    A TUN-mode client answers DNS with an address from 198.18.0.0/15 and
    routes the connection back out by SNI/Host. Refusing that range outright
    makes every real page unfetchable on such a machine, so HTTPS is
    admitted - the certificate check still binds the connection to the exact
    hostname requested - while plain HTTP, which has no such guarantee,
    stays refused like every other non-global address.
    """

    FAKE = ips("198.18.0.154")[0]

    def test_https_to_the_range_reaches_the_transport(self):
        _, _, transport, result = _fetch(
            "https://real.test/page",
            resolver_answers=(ips("198.18.0.154"),),
        )
        assert result.status == STATUS_OK
        assert transport.calls[0]["pinned_ip"] == self.FAKE
        assert transport.calls[0]["use_tls"] is True

    def test_http_to_the_range_is_still_refused(self):
        _, _, transport, result = _fetch(
            "http://real.test/page",
            resolver_answers=(ips("198.18.0.154"),),
        )
        assert result.status == STATUS_BLOCKED_ADDRESS
        assert transport.calls == []

    def test_the_range_edges_are_exact(self):
        # Inside the range: admitted over HTTPS, refused over HTTP.
        for literal in ("198.18.0.0", "198.18.0.154", "198.19.255.255"):
            ip = ipaddress.ip_address(literal)
            assert _destination_allowed(ip, use_tls=True) is True
            assert _destination_allowed(ip, use_tls=False) is False
        # Outside it, a non-global address is refused over HTTPS too: the
        # admission is one range and nothing else.
        for literal in ("100.64.0.1", "240.0.0.1", "169.254.1.1"):
            assert _destination_allowed(
                ipaddress.ip_address(literal), use_tls=True
            ) is False

    def test_private_and_loopback_stay_blocked_over_https(self):
        # The admission is one range only; every other non-global address
        # is refused over HTTPS exactly as before.
        for literal in ("127.0.0.1", "10.0.0.1", "169.254.169.254"):
            _, _, transport, result = _fetch(
                "https://internal.test/",
                resolver_answers=(ips(literal),),
            )
            assert result.status == STATUS_BLOCKED_ADDRESS
            assert transport.calls == []

    def test_a_mixed_answer_is_still_refused_over_https(self):
        # One admitted address beside a private one is the rebinding
        # signature; the whole answer is refused, as it always was.
        _, _, transport, result = _fetch(
            "https://rebind.test/",
            resolver_answers=(ips("198.18.0.154", "10.0.0.1"),),
        )
        assert result.status == STATUS_BLOCKED_ADDRESS
        assert transport.calls == []

    def test_an_https_redirect_downgraded_to_http_is_blocked(self):
        redirect = _FakeResponse(
            status=302,
            headers={
                "Location": "http://internal.test/",
                "Content-Type": "text/html",
            },
            body=b"",
        )
        _, _, transport, result = _fetch(
            "https://real.test/page",
            resolver_answers=(ips("198.18.0.154"), ips("10.0.0.1")),
            responses=(redirect, _html_response()),
        )
        assert result.status == STATUS_BLOCKED_ADDRESS
        assert len(transport.calls) == 1  # the second hop never ran


# ======================================================================
# Compressed responses
# ======================================================================


class TestCompressedResponses:
    """A cached response can arrive compressed even though we never ask.

    Varnish and friends serve a stored ``content-encoding: gzip`` body to a
    request that sent no Accept-Encoding header at all. Reading those bytes
    as HTML silently yields garbage rather than an honest failure, so they
    are inflated first - under the same size cap, so a stream that is tiny
    on the wire cannot expand into memory.
    """

    def _gzipped(self, text, *, extra_headers=None):
        headers = {
            "Content-Type": "text/html; charset=utf-8",
            "Content-Encoding": "gzip",
        }
        headers.update(extra_headers or {})
        return _html_response(body=gzip.compress(_ok_page(text)),
                              headers=headers)

    def test_a_gzip_body_is_inflated_before_extraction(self):
        response = self._gzipped(
            "Python 3.13.6 has been superseded by Python 3.13.16."
        )
        _, _, _, result = _fetch("https://real.test/", responses=(response,))
        assert result.status == STATUS_OK
        assert "superseded" in result.text

    def test_an_identity_encoding_is_left_alone(self):
        response = _html_response(headers={
            "Content-Type": "text/html",
            "Content-Encoding": "identity",
        })
        _, _, _, result = _fetch("https://real.test/", responses=(response,))
        assert result.status == STATUS_OK
        assert "latest stable release" in result.text

    def test_no_encoding_header_is_untouched(self):
        _, _, _, result = _fetch("https://real.test/")
        assert result.status == STATUS_OK
        assert "latest stable release" in result.text

    def test_the_inflated_size_is_what_the_cap_applies_to(self):
        # Far below the cap on the wire, far above it once inflated.
        bomb = gzip.compress(b"A " * (MAX_RESPONSE_BYTES + 4096))
        assert len(bomb) < MAX_RESPONSE_BYTES
        response = _html_response(
            body=bomb,
            headers={
                "Content-Type": "text/html",
                "Content-Encoding": "gzip",
            },
        )
        _, _, _, result = _fetch("https://real.test/", responses=(response,))
        assert result.status == STATUS_TOO_LARGE
        assert "exceeds" in result.error

    def test_an_unsupported_encoding_is_reported_not_guessed(self):
        response = _html_response(
            body=b"\x00\x01\x02not-actually-brotli",
            headers={
                "Content-Type": "text/html",
                "Content-Encoding": "br",
            },
        )
        _, _, _, result = _fetch("https://real.test/", responses=(response,))
        assert result.status == STATUS_UNSUPPORTED_CONTENT
        assert "br" in result.error

    def test_a_corrupt_stream_is_reported_not_raised(self):
        response = _html_response(
            body=b"\x1f\x8b\x08\x00truncated-garbage",
            headers={
                "Content-Type": "text/html",
                "Content-Encoding": "gzip",
            },
        )
        _, _, _, result = _fetch("https://real.test/", responses=(response,))
        assert result.status == STATUS_FETCH_ERROR
        assert "decoded" in result.error

    def test_a_deflate_body_is_inflated(self):
        response = _html_response(
            body=zlib.compress(_ok_page("Deflated prose survives intact.")),
            headers={
                "Content-Type": "text/html",
                "Content-Encoding": "deflate",
            },
        )
        _, _, _, result = _fetch("https://real.test/", responses=(response,))
        assert result.status == STATUS_OK
        assert "Deflated prose" in result.text


# ======================================================================
# DNS rebinding
# ======================================================================


class TestDNSRebinding:
    def test_the_connection_is_pinned_to_the_validated_address(self):
        _, resolver, transport, result = _fetch(
            "http://rebind.test/",
            resolver_answers=(ips("93.184.216.34"),),
        )
        assert result.status == STATUS_OK
        call = transport.calls[0]
        # The socket target is the resolved, validated IP - never the name.
        assert call["pinned_ip"] == PUBLIC
        # The hostname survives only as the Host/SNI identity.
        assert call["host"] == "rebind.test"
        assert call["server_hostname"] == "rebind.test"

    def test_the_hostname_is_never_used_as_the_connect_target(self):
        # The fake transport receives pinned_ip; asserting it is an IP object
        # (not the string hostname) demonstrates no implicit re-resolution.
        _, _, transport, _ = _fetch("http://rebind.test/")
        pinned = transport.calls[0]["pinned_ip"]
        assert isinstance(pinned, (ipaddress.IPv4Address,
                                   ipaddress.IPv6Address))

    def test_the_resolver_is_consulted_once_per_hop(self):
        _, resolver, _, _ = _fetch("http://one.test/")
        assert resolver.calls == ["one.test"]

    def test_an_ipv6_hostname_pins_the_ipv6_address(self):
        _, _, transport, result = _fetch(
            "http://v6.test/", resolver_answers=(ips("2606:4700:4700::1111"),)
        )
        assert result.status == STATUS_OK
        assert transport.calls[0]["pinned_ip"] == PUBLIC_V6

    def test_a_dns_failure_is_reported_not_raised(self):
        resolver = _FakeResolver(socket.gaierror("name or service unknown"))
        transport = _FakeTransport()
        fetcher = PageFetcher(resolver=resolver, transport=transport)
        result = fetcher.fetch("http://dead.test/")
        assert result.status == STATUS_FETCH_ERROR
        assert "dns failed" in result.error
        assert transport.calls == []

    def test_an_empty_dns_answer_is_reported(self):
        _, _, transport, result = _fetch("http://empty.test/",
                                         resolver_answers=((),))
        assert result.status == STATUS_FETCH_ERROR
        assert "no addresses" in result.error
        assert transport.calls == []


# ======================================================================
# Redirect validation
# ======================================================================


class TestRedirectValidation:
    def test_a_redirect_into_a_private_range_is_blocked(self):
        responses = (
            _FakeResponse(status=301, headers={"Location":
                                               "http://127.0.0.1/"}),
        )
        _, resolver, transport, result = _fetch(
            "http://open.test/", responses=responses,
            resolver_answers=(ips("93.184.216.34"),),
        )
        assert result.status == STATUS_BLOCKED_ADDRESS
        assert transport.calls, "the first hop ran; the redirect target did not"

    def test_a_redirect_to_a_blocked_scheme_is_blocked(self):
        responses = (
            _FakeResponse(status=302, headers={"Location":
                                               "file:///etc/passwd"}),
        )
        _, _, _, result = _fetch("http://open.test/", responses=responses)
        assert result.status == STATUS_BLOCKED_SCHEME

    def test_a_redirect_to_a_new_hostname_is_re_resolved_and_validated(self):
        responses = (
            _FakeResponse(status=301, headers={"Location":
                                               "http://other.test/"}),
            _html_response(text="Redirected content here."),
        )
        _, resolver, transport, result = _fetch(
            "http://open.test/",
            responses=responses,
            resolver_answers=(ips("93.184.216.34"), ips("93.184.216.34")),
        )
        assert result.status == STATUS_OK
        assert resolver.calls == ["open.test", "other.test"]
        assert transport.calls[1]["host"] == "other.test"

    def test_a_relative_redirect_is_resolved_against_the_current_url(self):
        responses = (
            _FakeResponse(status=301, headers={"Location": "/next"}),
            _html_response(text="At the next page."),
        )
        # Two hops to the same hostname, so the resolver needs two answers.
        answers = (ips("93.184.216.34"), ips("93.184.216.34"))
        _, _, transport, result = _fetch(
            "http://open.test/start", responses=responses,
            resolver_answers=answers,
        )
        assert result.status == STATUS_OK
        assert transport.calls[1]["path"] == "/next"

    def test_exactly_three_redirects_are_followed(self):
        responses = (
            _FakeResponse(status=301, headers={"Location": "http://h1.test/"}),
            _FakeResponse(status=301, headers={"Location": "http://h2.test/"}),
            _FakeResponse(status=301, headers={"Location": "http://h3.test/"}),
            _html_response(text="Third hop content."),
        )
        answers = (ips("93.184.216.34"),) * 4
        _, resolver, transport, result = _fetch(
            "http://h0.test/", responses=responses, resolver_answers=answers
        )
        assert result.status == STATUS_OK
        assert len(transport.calls) == 4  # original + three redirects

    def test_a_fourth_redirect_is_refused(self):
        responses = (
            _FakeResponse(status=301, headers={"Location": "http://h1.test/"}),
            _FakeResponse(status=301, headers={"Location": "http://h2.test/"}),
            _FakeResponse(status=301, headers={"Location": "http://h3.test/"}),
            _FakeResponse(status=301, headers={"Location": "http://h4.test/"}),
        )
        answers = (ips("93.184.216.34"),) * 4
        _, resolver, transport, result = _fetch(
            "http://h0.test/", responses=responses, resolver_answers=answers
        )
        assert result.status == STATUS_TOO_MANY_REDIRECTS
        assert len(transport.calls) == 4  # the fourth target is never opened

    def test_a_redirect_without_a_location_is_an_http_error(self):
        responses = (_FakeResponse(status=301, headers={}),)
        _, _, _, result = _fetch("http://open.test/", responses=responses)
        assert result.status == STATUS_HTTP_ERROR
        assert "Location" in result.error

    def test_final_url_records_where_the_body_came_from(self):
        responses = (
            _FakeResponse(status=302, headers={"Location":
                                               "http://moved.test/"}),
            _html_response(text="Moved content."),
        )
        answers = (ips("93.184.216.34"),) * 2
        _, _, _, result = _fetch(
            "http://open.test/", responses=responses, resolver_answers=answers
        )
        assert result.status == STATUS_OK
        assert result.final_url == "http://moved.test/"
        assert result.url == "http://open.test/"


# ======================================================================
# Response-size limit
# ======================================================================


class TestResponseSizeLimit:
    def test_a_body_exactly_at_the_cap_is_accepted(self):
        body = b"<p>" + b"a" * (MAX_RESPONSE_BYTES - 7) + b"</p>"
        assert len(body) == MAX_RESPONSE_BYTES
        responses = (_html_response(body=body),)
        _, _, _, result = _fetch("http://big.test/", responses=responses)
        assert result.status == STATUS_OK

    def test_one_byte_over_the_cap_aborts(self):
        body = b"<p>" + b"a" * (MAX_RESPONSE_BYTES - 6) + b"</p>"
        assert len(body) == MAX_RESPONSE_BYTES + 1
        responses = (_html_response(body=body),)
        _, _, _, result = _fetch("http://big.test/", responses=responses)
        assert result.status == STATUS_TOO_LARGE
        assert str(MAX_RESPONSE_BYTES) in result.error

    def test_the_cap_is_512_kibibytes(self):
        assert MAX_RESPONSE_BYTES == 512 * 1024

    def test_an_oversized_body_is_read_in_chunks_not_all_at_once(self):
        # A response whose read() would return everything at once still has
        # to be streamed through _read_capped; a body just over the cap
        # proves the running total is checked, not a single read's result.
        body = b"x" * (MAX_RESPONSE_BYTES + 1)
        responses = (_html_response(body=body),)
        _, _, _, result = _fetch("http://big.test/", responses=responses)
        assert result.status == STATUS_TOO_LARGE


# ======================================================================
# Timeouts and transport errors
# ======================================================================


class TestTimeoutsAndTransportErrors:
    def test_a_timeout_is_reported_as_timeout(self):
        transport = _FakeTransport(error=TimeoutError("timed out"))
        fetcher = PageFetcher(
            resolver=_FakeResolver(ips("93.184.216.34")), transport=transport
        )
        result = fetcher.fetch("http://slow.test/")
        assert result.status == STATUS_TIMEOUT
        assert "timed out" in result.error

    def test_socket_timeout_is_also_a_timeout(self):
        transport = _FakeTransport(error=socket.timeout("timed out"))
        fetcher = PageFetcher(
            resolver=_FakeResolver(ips("93.184.216.34")), transport=transport
        )
        result = fetcher.fetch("http://slow.test/")
        assert result.status == STATUS_TIMEOUT

    def test_a_connection_refusal_is_a_fetch_error(self):
        transport = _FakeTransport(error=ConnectionRefusedError("refused"))
        fetcher = PageFetcher(
            resolver=_FakeResolver(ips("93.184.216.34")), transport=transport
        )
        result = fetcher.fetch("http://down.test/")
        assert result.status == STATUS_FETCH_ERROR
        assert "refused" in result.error

    def test_an_unexpected_transport_exception_never_escapes(self):
        transport = _FakeTransport(error=RuntimeError("boom"))
        fetcher = PageFetcher(
            resolver=_FakeResolver(ips("93.184.216.34")), transport=transport
        )
        result = fetcher.fetch("http://boom.test/")
        assert result.status == STATUS_FETCH_ERROR
        assert "boom" in result.error

    def test_a_non_200_status_is_an_http_error(self):
        responses = (_FakeResponse(status=404, body=b"missing"),)
        _, _, _, result = _fetch("http://gone.test/", responses=responses)
        assert result.status == STATUS_HTTP_ERROR
        assert "404" in result.error

    def test_a_non_html_content_type_is_unsupported(self):
        responses = (
            _FakeResponse(headers={"Content-Type": "application/pdf"},
                          body=b"%PDF-1.4"),
        )
        _, _, _, result = _fetch("http://doc.test/", responses=responses)
        assert result.status == STATUS_UNSUPPORTED_CONTENT
        assert "pdf" in result.error

    def test_content_type_parameters_are_ignored(self):
        responses = (_html_response(),)
        _, _, _, result = _fetch("http://ok.test/", responses=responses)
        assert result.status == STATUS_OK

    def test_a_missing_content_type_is_unsupported(self):
        # No Content-Type header at all: the body type is unknown, so the
        # fetcher refuses to parse it rather than guessing.
        responses = (_FakeResponse(body=_ok_page()),)
        _, _, _, result = _fetch("http://bare.test/", responses=responses)
        assert result.status == STATUS_UNSUPPORTED_CONTENT


# ======================================================================
# Header hygiene: no cookies, no credentials, nothing from the environment
# ======================================================================


class TestHeaderHygiene:
    def test_only_the_two_documented_headers_are_sent(self):
        _, _, transport, _ = _fetch("http://headers.test/")
        headers = transport.calls[0]["headers"]
        assert set(headers) == {"User-Agent", "Accept"}

    def test_no_cookie_authorization_or_proxy_header_is_sent(self):
        _, _, transport, _ = _fetch("http://headers.test/")
        headers = transport.calls[0]["headers"]
        forbidden = ("cookie", "authorization", "proxy-authorization",
                     "proxy-connection", "x-api-key")
        for name in headers:
            assert name.lower() not in forbidden, f"forbidden header: {name}"

    def test_the_user_agent_is_the_fixed_primora_string(self):
        _, _, transport, _ = _fetch("http://headers.test/")
        assert transport.calls[0]["headers"]["User-Agent"] == USER_AGENT

    def test_the_accept_header_names_html_only(self):
        _, _, transport, _ = _fetch("http://headers.test/")
        assert transport.calls[0]["headers"]["Accept"] == ACCEPT_HEADER

    def test_userinfo_in_the_url_is_rejected_before_any_io(self):
        _, resolver, transport, result = _fetch(
            "http://user:secret@example.test/"
        )
        assert result.status == STATUS_BLOCKED_USERINFO
        assert resolver.calls == []
        assert transport.calls == []

    def test_an_at_only_url_is_also_rejected(self):
        _, _, transport, result = _fetch("http://@example.test/")
        assert result.status == STATUS_BLOCKED_USERINFO
        assert transport.calls == []

    def test_the_timeout_forwarded_to_the_transport_is_the_configured_one(self):
        _, _, transport, _ = _fetch("http://t.test/", timeout_s=2.5)
        assert transport.calls[0]["timeout"] == 2.5


# ======================================================================
# HTML cleanup: untrusted content, stripped to prose
# ======================================================================


class TestHtmlCleanup:
    def _page(self):
        return _html_page(
            [
                "Python 3.13 is the latest stable release.",
                "It ships with a new incremental garbage collector.",
            ],
            script="alert('pwned')",
            style="body { display: none }",
            nav="Home | About | Login",
            footer="Copyright notice and tracking pixels.",
        )

    def test_script_content_is_removed(self):
        text = _html_to_text(self._page())
        assert "alert" not in text
        assert "pwned" not in text

    def test_style_content_is_removed(self):
        text = _html_to_text(self._page())
        assert "display" not in text

    def test_navigation_is_removed(self):
        text = _html_to_text(self._page())
        assert "Login" not in text

    def test_footer_is_removed(self):
        text = _html_to_text(self._page())
        assert "Copyright" not in text

    def test_paragraph_prose_survives(self):
        text = _html_to_text(self._page())
        assert "latest stable release" in text
        assert "incremental garbage collector" in text

    def test_entities_are_decoded_to_plain_text(self):
        body = b"<p>A &amp; B &lt; C &gt; D &quot;E&quot; &#39;F&#39;</p>"
        text = _html_to_text(body)
        assert "A & B < C > D \"E\" 'F'" in text

    def test_tags_are_stripped(self):
        text = _html_to_text(b"<p><b>bold</b> and <i>italic</i></p>")
        assert text == "bold and italic"

    def test_a_naked_script_without_a_closing_tag_does_not_leak(self):
        # Hostile markup: the parser may desync, but the script body must
        # not be emitted as prose.
        body = b"<p>visible</p><script>var secret = 1;"
        text = _html_to_text(body)
        assert "visible" in text
        assert "secret" not in text

    def test_malformed_html_does_not_raise(self):
        _html_to_text(b"<p><div><span>unclosed")

    def test_an_empty_body_is_empty_text(self):
        assert _html_to_text(b"") == ""


# ======================================================================
# Sentence selection: relevance and the hard budget
# ======================================================================


class TestSentenceSelection:
    def test_the_output_never_exceeds_the_budget(self):
        text = " ".join(f"Sentence number {i} about nothing in particular."
                        for i in range(200))
        out = _select_sentences(text, "unrelated query", MAX_EXTRACTED_CHARS)
        assert len(out) <= MAX_EXTRACTED_CHARS

    def test_query_relevant_sentences_are_preferred(self):
        text = (
            "The weather in Paris is warm this week. "
            "Python 3.13 introduces a new JIT compiler. "
            "Bananas are yellow and soft to the touch."
        )
        out = _select_sentences(text, "Python 3.13 compiler", 800)
        assert "JIT compiler" in out
        assert "Bananas" not in out

    def test_relevance_beats_position(self):
        text = (
            "An unrelated opening sentence about the sea. "
            "Another unrelated sentence about mountains. "
            "Python 3.13 is the latest stable release."
        )
        out = _select_sentences(text, "Python release", 800)
        assert "latest stable release" in out
        assert "unrelated opening" not in out
        assert "unrelated sentence about mountains" not in out

    def test_no_match_falls_back_to_leading_sentences(self):
        text = "First sentence here. Second sentence follows. Third ends it."
        out = _select_sentences(text, "zzzz qqqq", 800)
        assert out.startswith("First sentence here.")

    def test_an_empty_query_keeps_the_leading_text(self):
        text = "First sentence here. Second sentence follows."
        out = _select_sentences(text, "", 800)
        assert out.startswith("First sentence here.")

    def test_the_budget_is_enforced_on_the_final_string(self):
        # Many short relevant sentences: the joined excerpt is truncated to
        # the cap, not merely limited by sentence count.
        text = " ".join(["Python is great."] * 100)
        out = _select_sentences(text, "python", MAX_EXTRACTED_CHARS)
        assert len(out) <= MAX_EXTRACTED_CHARS
        assert out

    def test_a_zero_budget_returns_empty(self):
        assert _select_sentences("Anything at all.", "anything", 0) == ""

    def test_stopwords_do_not_dominate_relevance(self):
        text = (
            "The release of the compiler was the event of the year for the "
            "team. "
            "Python 3.13 shipped a new garbage collector."
        )
        out = _select_sentences(text, "the of and for", 800)
        # With only stopwords as terms, no sentence scores > 0, so the
        # leading sentence is kept rather than a random match.
        assert out.startswith("The release of the compiler")

    def test_the_documented_maximum_is_800(self):
        assert MAX_EXTRACTED_CHARS == 800


# ======================================================================
# End-to-end happy path through the fetcher
# ======================================================================


class TestHappyPath:
    def test_a_normal_page_is_fetched_and_extracted(self):
        page = _html_page(
            ["Python 3.13 is the latest stable release.",
             "Download it from the official site."],
            script="evil()",
        )
        responses = (_html_response(body=page),)
        _, _, transport, result = _fetch(
            "https://python.test/downloads", responses=responses,
            query="python release",
        )
        assert result.status == STATUS_OK
        assert "latest stable release" in result.text
        assert "evil" not in result.text
        assert result.content_type == "text/html"
        assert result.final_url == "https://python.test/downloads"

    def test_https_sets_use_tls_and_the_sni_name(self):
        _, _, transport, _ = _fetch("https://secure.test/page")
        call = transport.calls[0]
        assert call["use_tls"] is True
        assert call["server_hostname"] == "secure.test"
        assert call["port"] == 443

    def test_http_defaults_to_port_80_without_tls(self):
        _, _, transport, _ = _fetch("http://plain.test/")
        call = transport.calls[0]
        assert call["use_tls"] is False
        assert call["port"] == 80

    def test_an_explicit_port_is_honoured(self):
        _, _, transport, _ = _fetch("http://odd.test:8080/x")
        assert transport.calls[0]["port"] == 8080
        assert transport.calls[0]["path"] == "/x"

    def test_the_query_string_is_preserved_in_the_request_path(self):
        _, _, transport, _ = _fetch("http://q.test/search?a=1&b=2")
        assert transport.calls[0]["path"] == "/search?a=1&b=2"

    def test_the_returned_result_is_a_fetched_result(self):
        _, _, _, result = _fetch("http://r.test/")
        assert isinstance(result, FetchResult)


# ======================================================================
# Real TLS against a local server: no external network, no SSRF weakening
# ======================================================================
#
# PageFetcher.fetch refuses loopback (the SSRF policy, proven above). The
# TLS code itself lives in _default_transport - the injectable transport
# seam - so these tests drive that real function against a real local TLS
# server bound to 127.0.0.1. The layers compose: fetch passes
# server_hostname=target.host (asserted by the fake-transport tests above)
# and _default_transport feeds that name to SNI and certificate hostname
# verification (asserted below). No SSRF check is relaxed for any of this.


class _TLSRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<p>TLS test page.</p>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # keep test output clean


class _QuietTLSServer(http.server.HTTPServer):
    def handle_error(self, request, client_address):
        # The negative tests abort mid-handshake; that surfaces here as a
        # handler error and is expected.
        pass


def _new_ec_key():
    return ec.generate_private_key(ec.SECP256R1())


def _cert_name(common_name):
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])


def _make_ca(common_name):
    key = _new_ec_key()
    name = _cert_name(common_name)
    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None), critical=True
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _make_leaf(ca_key, ca_cert, common_name, *, dns_names=(), ip_literals=()):
    key = _new_ec_key()
    now = datetime.now(timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(_cert_name(common_name))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
    )
    entries = [x509.DNSName(name) for name in dns_names]
    entries.extend(
        x509.IPAddress(ipaddress.ip_address(literal))
        for literal in ip_literals
    )
    if entries:
        builder = builder.add_extension(
            x509.SubjectAlternativeName(entries), critical=False
        )
    cert = builder.sign(ca_key, hashes.SHA256())
    return key, cert


def _cert_pem(cert):
    return cert.public_bytes(serialization.Encoding.PEM)


def _key_pem(key):
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


@pytest.fixture
def tls_env(monkeypatch, tmp_path):
    """Ephemeral CA + leaf certificates + a local TLS-server factory.

    Replaces ``ssl.create_default_context`` with a context that has the
    same posture as the real one (``PROTOCOL_TLS_CLIENT``:
    ``check_hostname=True``, ``CERT_REQUIRED``, TLS >= 1.2) but trusts only
    this test's CA, so a locally generated chain stands in for a publicly
    trusted one. The production trust store and the network are untouched.
    """
    ca_key, ca_cert = _make_ca("PRIMORA Fetch Test CA")
    good_key, good_cert = _make_leaf(
        ca_key, ca_cert, "tls.test", dns_names=("tls.test",)
    )
    other_key, other_cert = _make_leaf(
        ca_key, ca_cert, "other.example", dns_names=("other.example",)
    )
    ip_only_key, ip_only_cert = _make_leaf(
        ca_key, ca_cert, "127.0.0.1", ip_literals=("127.0.0.1",)
    )
    rogue_ca_key, rogue_ca_cert = _make_ca("PRIMORA Rogue CA")
    rogue_key, rogue_cert = _make_leaf(
        rogue_ca_key, rogue_ca_cert, "tls.test", dns_names=("tls.test",)
    )

    certs = {
        "good": (good_key, good_cert),
        "other": (other_key, other_cert),
        "ip_only": (ip_only_key, ip_only_cert),
        "rogue": (rogue_key, rogue_cert),
    }
    for label, (_key, cert) in certs.items():
        (tmp_path / f"{label}.pem").write_bytes(_cert_pem(cert))
    for label, (key, _cert) in certs.items():
        (tmp_path / f"{label}_key.pem").write_bytes(_key_pem(key))

    ca_pem = tmp_path / "ca.pem"
    ca_pem.write_bytes(_cert_pem(ca_cert))

    def _client_context(*args, **kwargs):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        # PROTOCOL_TLS_CLIENT already sets check_hostname=True and
        # verify_mode=CERT_REQUIRED - the same posture as the real default
        # context. Only the trust anchor differs.
        context.load_verify_locations(cafile=str(ca_pem))
        return context

    monkeypatch.setattr(ssl, "create_default_context", _client_context)

    servers = []

    def start_server(cert_label):
        """Start a TLS server presenting ``cert_label``; return (port, snis)."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(
            certfile=str(tmp_path / f"{cert_label}.pem"),
            keyfile=str(tmp_path / f"{cert_label}_key.pem"),
        )
        snis: list = []
        context.sni_callback = (
            lambda sock, servername, ctx: snis.append(servername)
        )
        httpd = _QuietTLSServer(("127.0.0.1", 0), _TLSRequestHandler)
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        servers.append((httpd, thread))
        return httpd.server_address[1], snis

    yield SimpleNamespace(start_server=start_server, tmp_path=tmp_path)

    for httpd, thread in servers:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()


def _real_transport_over_tls(port, *, server_hostname, path="/"):
    """Drive the production transport against the local TLS server.

    The pinned IP is 127.0.0.1 because that is where the test server lives;
    ``fetch``'s address policy still refuses loopback, and is exercised by
    the TestBlockedDestinations suite above.
    """
    return _default_transport(
        server_hostname,
        port,
        path,
        use_tls=True,
        server_hostname=server_hostname,
        pinned_ip=ipaddress.ip_address("127.0.0.1"),
        timeout=5.0,
        headers={"User-Agent": USER_AGENT, "Accept": ACCEPT_HEADER},
    )


class TestRealTLSHandshake:
    """The production TLS path against a real local TLS server."""

    def test_valid_certificate_completes_a_real_handshake(self, tls_env):
        port, _snis = tls_env.start_server("good")
        with _real_transport_over_tls(
            port, server_hostname="tls.test"
        ) as resp:
            assert resp.status == 200
            assert b"TLS test page" in resp.read()

    def test_hostname_mismatch_is_rejected(self, tls_env):
        # The chain is trusted (leaf signed by our CA), but the leaf is for
        # other.example while the client asks for tls.test.
        port, _snis = tls_env.start_server("other")
        with pytest.raises(ssl.SSLCertVerificationError) as excinfo:
            _real_transport_over_tls(port, server_hostname="tls.test")
        assert "hostname" in str(excinfo.value).lower()

    def test_untrusted_certificate_is_rejected(self, tls_env):
        # The leaf claims tls.test but is signed by a CA the client does
        # not trust - chain validation must refuse it.
        port, _snis = tls_env.start_server("rogue")
        with pytest.raises(ssl.SSLCertVerificationError):
            _real_transport_over_tls(port, server_hostname="tls.test")

    def test_verification_uses_the_hostname_not_the_pinned_ip(self, tls_env):
        # The leaf is valid only for the IP literal 127.0.0.1 - the same
        # address the socket is pinned to. If certificate verification
        # checked the pinned IP instead of the URL hostname, this would
        # succeed; because it checks server_hostname ("tls.test"), it fails.
        port, _snis = tls_env.start_server("ip_only")
        with pytest.raises(ssl.SSLCertVerificationError):
            _real_transport_over_tls(port, server_hostname="tls.test")

    def test_a_hostname_only_certificate_succeeds_despite_a_pinned_ip(
        self, tls_env
    ):
        # The converse: the leaf carries only DNS:tls.test and no IP SAN,
        # yet the connection is pinned to 127.0.0.1. Success proves the
        # pinned IP is not fed into the hostname check.
        port, _snis = tls_env.start_server("good")
        with _real_transport_over_tls(
            port, server_hostname="tls.test"
        ) as resp:
            assert resp.status == 200

    def test_sni_preserves_the_original_hostname(self, tls_env):
        port, snis = tls_env.start_server("good")
        with _real_transport_over_tls(port, server_hostname="tls.test"):
            pass
        # The client socket was pinned to 127.0.0.1, yet the server saw the
        # original URL hostname in SNI - never the IP.
        assert snis == ["tls.test"]
        assert "127.0.0.1" not in snis

    def test_the_production_path_uses_create_default_context(self, monkeypatch):
        # A bare TCP listener accepts and closes immediately, so the
        # handshake fails - but only after create_default_context has run,
        # which is what this spy proves.
        calls = []
        real = ssl.create_default_context

        def spy(*args, **kwargs):
            calls.append(True)
            return real(*args, **kwargs)

        monkeypatch.setattr(ssl, "create_default_context", spy)
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def accept_and_close():
            conn, _addr = listener.accept()
            conn.close()

        threading.Thread(target=accept_and_close, daemon=True).start()
        try:
            with pytest.raises(OSError):
                _real_transport_over_tls(port, server_hostname="tls.test")
        finally:
            listener.close()
        assert calls, "the transport never reached create_default_context"

    def test_a_tls_verification_failure_is_reported_not_raised(self):
        # Composition: fetch maps a transport-level verification failure to
        # an honest STATUS_FETCH_ERROR rather than letting it escape.
        error = ssl.SSLCertVerificationError(
            "certificate verify failed: Hostname mismatch"
        )
        transport = _FakeTransport(error=error)
        fetcher = PageFetcher(
            resolver=_FakeResolver(ips("93.184.216.34")), transport=transport
        )
        result = fetcher.fetch("https://rebind.test/")
        assert result.status == STATUS_FETCH_ERROR
        assert "certificate" in result.error.lower()


# ======================================================================
# Module hygiene: no coupling to shell, database, environment, or eval
# ======================================================================


class TestModuleHygiene:
    PATH = os.path.join("PRIMORA", "research", "fetcher.py")

    def _source(self):
        with open(self.PATH, encoding="utf-8") as handle:
            return handle.read()

    def test_it_imports_only_standard_library_modules(self):
        tree = ast.parse(self._source())
        top_level = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(
                    (alias.name or "").split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_level.add(node.module.split(".")[0])
        allowed = {
            "__future__", "http", "ipaddress", "re", "socket", "ssl",
            "urllib", "dataclasses", "html", "typing", "zlib",
        }
        assert top_level <= allowed, f"unexpected imports: {top_level}"

    def test_it_never_touches_the_environment_or_subprocesses(self):
        source = self._source().lower()
        for banned in ("os.environ", "os.getenv", "subprocess", "sqlite",
                       "shutil", "pickle", "marshal"):
            assert banned not in source, f"forbidden coupling: {banned}"

    def test_it_never_evaluates_or_executes(self):
        tree = ast.parse(self._source())
        forbidden_calls = {"eval", "exec", "compile", "__import__"}
        found = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not (found & forbidden_calls), f"dynamic execution: {found}"

    def test_it_does_not_import_the_model_loop_memory_or_capabilities(self):
        source = self._source().lower()
        for banned in ("aurora", "prime", "capability", "specialist",
                       "research.provider", "research.service",
                       "research.searxng"):
            assert banned not in source, f"coupling leaked: {banned}"
