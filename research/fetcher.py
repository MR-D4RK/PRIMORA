"""Secure single-page fetcher for research evidence.

Milestone 3, Phase 3 Step 1. The search providers return ranked URLs and
snippets; this module is the only place PRIMORA opens a result page. It sits
below the provider seam and beside ``searxng``: the provider answers "what
are the candidate sources", the fetcher answers "what does this one page
actually say".

Why the security policy lives here and not in the provider
----------------------------------------------------------
A provider request goes to one fixed, operator-controlled endpoint. A page
fetch goes wherever the open web points - including, adversarially, back at
this machine. That is a different threat model, so it is its own module with
its own guards rather than something bolted onto ``ResearchProvider``.

The policy, in order:

1. Only ``http``/``https``. No ``file``, ``ftp``, ``javascript``, ``data``.
2. No credentials in the URL: userinfo is rejected, never sent.
3. Every hop - the original URL and every redirect target - is resolved and
   validated independently. A hostname is resolved once; every returned
   address is checked; the connection is pinned to a validated address, so a
   second DNS answer (the DNS-rebinding trick) can never be reached.
4. Only globally routable unicast destinations. Loopback, private,
   link-local, reserved, multicast, unspecified, and every IPv6 form that
   embeds an IPv4 address (IPv4-mapped, IPv4-compatible, NAT64, 6to4,
   Teredo) are refused. One narrow exception: the IANA benchmarking range
   198.18.0.0/15 is admitted over HTTPS, because a local transparent proxy
   answers every hostname with an address from that range and still routes
   the connection by SNI - and the certificate check keeps the connection
   bound to the exact hostname requested. Plain HTTP to the range is
   refused like every other non-global address.
5. Unusual IPv4 literals (decimal, octal, hex, shortened) do not parse as IP
   addresses, so they fall through to DNS where the same address check is
   applied to whatever the resolver returns.
6. No cookies, authorization headers, or environment-derived credentials are
   ever constructed or sent. The request carries a fixed User-Agent and an
   Accept header; nothing else.
7. Response size is capped at 512 KB, read incrementally: a body that
   exceeds the cap aborts the fetch rather than being truncated into memory.
   A body that arrives compressed is inflated first, and the *inflated* size
   is held to the same cap, so a small compressed stream cannot expand into
   memory under the limit's nose.
8. Request duration is capped at 5 seconds per hop; at most three redirects
   are followed.
9. Only HTML (and XHTML) bodies are accepted for extraction; anything else
   is reported as unsupported, not parsed.
10. Extracted text is query-filtered to a strict 800-character budget, so a
    page can never overwhelm the context window.

Untrusted by construction
-------------------------
Every byte that returns is treated as data: parsed by a lenient HTML parser
that discards script/style/navigation, reduced to plain text, and returned
in a ``FetchResult`` field. Nothing here executes, evaluates, imports,
persists, or interprets fetched content as instructions. A future caller
that pastes this text into a prompt owns the prompt-injection surface; this
module owns only the network and extraction surface.

Deliberately absent
-------------------
No crawling (one URL, one response), no recursion, no robots.txt fetch, no
compression *negotiation* (this module never asks for a compressed body, so
a size cap means what it says), no cookie jar, no proxy configuration, and
no coupling to the model or to memory. A server that compresses a response
anyway is inflated under the same cap; any other content-encoding is
reported as unsupported rather than guessed at.
``resolver`` and ``transport`` are injected, which is why the
entire policy is exercised in tests with zero live network access.
"""

from __future__ import annotations

import http.client
import ipaddress
import re
import socket
import ssl
import urllib.parse
import zlib
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable, Optional

# Hard ceilings. Each is a deliberate refusal point, not a soft target:
# exceeding any of them ends the fetch with a named status.
MAX_RESPONSE_BYTES = 512 * 1024
REQUEST_TIMEOUT_S = 5.0
MAX_REDIRECTS = 3
MAX_EXTRACTED_CHARS = 800

ALLOWED_SCHEMES = frozenset({"http", "https"})
HTML_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})

# Content-encodings this module will inflate: only what the standard library
# can decode under a size cap. This module never *requests* compression, but a
# cached response can still arrive compressed (Varnish and friends do this),
# and treating compressed bytes as HTML silently yields garbage instead of an
# honest failure. Anything else - brotli, zstd - is reported as unsupported
# rather than guessed at.
DECODABLE_ENCODINGS = frozenset({"gzip", "x-gzip", "deflate"})

# The only two headers this module ever puts on the wire. Neither carries a
# credential, and the transport forwards the mapping verbatim.
USER_AGENT = "PRIMORA/1.0 (+research fetcher)"
ACCEPT_HEADER = "text/html,application/xhtml+xml"

# Fetch outcomes. A fetch always returns exactly one of these; there is no
# silent-empty path and no exception escapes ``PageFetcher.fetch``.
STATUS_OK = "ok"
STATUS_BLOCKED_SCHEME = "blocked_scheme"
STATUS_BLOCKED_USERINFO = "blocked_userinfo"
STATUS_BLOCKED_ADDRESS = "blocked_address"
STATUS_TOO_MANY_REDIRECTS = "too_many_redirects"
STATUS_TIMEOUT = "timeout"
STATUS_TOO_LARGE = "too_large"
STATUS_UNSUPPORTED_CONTENT = "unsupported_content"
STATUS_HTTP_ERROR = "http_error"
STATUS_FETCH_ERROR = "fetch_error"


@dataclass(frozen=True)
class FetchResult:
    """One fetch attempt, reported honestly.

    ``text`` is non-empty only when ``status`` is ``STATUS_OK``. Error
    statuses carry a human-readable ``error`` and never fabricate content.
    ``final_url`` records the URL that actually returned the body; it differs
    from ``url`` only when redirects were followed.
    """

    url: str
    status: str
    text: str = ""
    final_url: str = ""
    content_type: str = ""
    error: str = ""


# ----------------------------------------------------------------------
# Address policy
# ----------------------------------------------------------------------

# IPv6 networks that embed an IPv4 address. A research fetcher has no reason
# to reach the internet through a transition gateway, and each of these can
# smuggle a blocked IPv4 target (loopback, for instance) past a naive IPv6
# check, so the whole prefixes are refused rather than decoded.
_V4_EMBEDDING_V6_PREFIXES = (
    ipaddress.ip_network("::ffff:0:0/96"),   # IPv4-mapped
    ipaddress.ip_network("::/96"),           # IPv4-compatible (deprecated)
    ipaddress.ip_network("64:ff9b::/96"),    # NAT64 well-known prefix
    ipaddress.ip_network("64:ff9b:1::/48"),  # NAT64 local-use prefix
    ipaddress.ip_network("2002::/16"),       # 6to4
    ipaddress.ip_network("2001::/32"),       # Teredo
)


def _try_ip(host: str):
    """Parse ``host`` as an IP literal, or return None.

    Unusual IPv4 forms (decimal, octal, hex, shortened) are *not* accepted:
    ``ipaddress`` rejects them, so they fall through to DNS where the same
    address policy is applied to whatever the resolver returns. That is the
    point - the check is on the address we will connect to, not on the
    spelling in the URL.
    """
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _is_blocked_ip(ip) -> bool:
    """Whether connecting to ``ip`` would reach a non-global destination.

    Fails closed: anything not demonstrably global unicast is blocked.
    Multicast is refused explicitly because this Python's ``is_global``
    still reports multicast as global (it is routable, after all) even
    though a page can never come back from a multicast group. IPv4-embedding
    IPv6 prefixes are refused wholesale because each can carry a blocked
    IPv4 address inside an apparently-global IPv6 one.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        for network in _V4_EMBEDDING_V6_PREFIXES:
            if ip in network:
                return True
    if ip.is_multicast:
        return True
    return not ip.is_global


# The IANA "benchmarking" range (RFC 2544). It is not globally routable, so
# the policy above refuses it like any other non-global destination. A local
# transparent proxy that answers *every* hostname with an address from this
# range - the fake-IP scheme used by common TUN-mode clients - routes the
# connection back out by SNI/Host, so refusing the range outright makes every
# real page unfetchable on such a machine. It is therefore admitted for HTTPS
# only: the certificate check still binds the connection to the exact hostname
# requested, so pointing a name at one of these addresses can never
# impersonate another site, and a redirect into the range fails the same
# verification. Plain HTTP has no such guarantee and stays refused, which is
# what keeps the address policy's protection intact where it matters.
_BENCHMARK_V4 = ipaddress.ip_network("198.18.0.0/15")


def _destination_allowed(ip, *, use_tls: bool) -> bool:
    """Whether the connection may go to ``ip`` for a URL of the given scheme.

    ``use_tls`` reports whether the request will be wrapped in TLS, which is
    the only reason the benchmarking range can be admitted at all. Every
    other address answers exactly as :func:`_is_blocked_ip` does, so the
    existing policy is unchanged for everything that is not that one range.
    """
    if use_tls and ip.version == 4 and ip in _BENCHMARK_V4:
        return True
    return not _is_blocked_ip(ip)


# ----------------------------------------------------------------------
# URL parsing
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    scheme: str
    host: str
    port: int
    path: str  # path + query, exactly as sent in the request line
    is_ip: bool


def _parse_target(url: str) -> tuple[Optional[_Target], Optional[str]]:
    """Structurally validate a URL. Returns ``(target, blocked_status)``.

    ``blocked_status`` is None only when the URL is structurally usable;
    otherwise it names the policy that rejected it. DNS and address checks
    happen later, after resolution, because a hostname is not yet an
    address at this point.
    """
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return None, STATUS_FETCH_ERROR
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        return None, STATUS_BLOCKED_SCHEME
    # Userinfo would become an Authorization header on some transports; it
    # is refused here so no credential-shaped data ever reaches the wire.
    if parts.username is not None or parts.password is not None:
        return None, STATUS_BLOCKED_USERINFO
    try:
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None, STATUS_FETCH_ERROR
    if not host:
        return None, STATUS_FETCH_ERROR
    if port is None:
        port = 443 if scheme == "https" else 80
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    return (
        _Target(
            scheme=scheme,
            host=host,
            port=port,
            path=path,
            is_ip=_try_ip(host) is not None,
        ),
        None,
    )


# ----------------------------------------------------------------------
# DNS resolution (injectable)
# ----------------------------------------------------------------------


def _default_resolver(host: str) -> list:
    """Resolve ``host`` to a de-duplicated list of addresses.

    An IP literal short-circuits to itself. Everything else goes through
    ``getaddrinfo``; unparseable address strings are dropped so the caller
    sees only addresses the policy can evaluate.
    """
    literal = _try_ip(host)
    if literal is not None:
        return [literal]
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    addresses: list = []
    seen: set = set()
    for info in infos:
        raw = info[4][0]
        parsed = _try_ip(raw)
        if parsed is None or raw in seen:
            continue
        seen.add(raw)
        addresses.append(parsed)
    return addresses


# ----------------------------------------------------------------------
# Transport (injectable)
# ----------------------------------------------------------------------


class _TransportResponse:
    """Wraps an ``http.client`` response so the fetcher can cap the read."""

    def __init__(self, response, connection) -> None:
        self._response = response
        self._connection = connection

    @property
    def status(self) -> int:
        return self._response.status

    @property
    def headers(self):
        return self._response.headers

    def read(self, size: int = -1) -> bytes:
        return self._response.read(size)

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


def _host_header_value(host: str) -> str:
    """Bracket an IPv6 literal for the Host header; pass hostnames through."""
    if _try_ip(host) is not None and ":" in host:
        return f"[{host}]"
    return host


def _default_transport(
    host: str,
    port: int,
    path: str,
    *,
    use_tls: bool,
    server_hostname: str,
    pinned_ip,
    timeout: float,
    headers: dict,
):
    """Open one HTTP(S) request to ``pinned_ip`` and return the response.

    The socket is created against the *validated* address, never against the
    hostname - this is the DNS-rebinding defense. TLS, when requested, wraps
    that same socket and sends ``server_hostname`` for SNI and certificate
    verification, so a pinned connection still presents the correct name.
    ``headers`` is forwarded verbatim; nothing is added here, which is what
    keeps credentials off the wire by construction.
    """
    sock = socket.create_connection((str(pinned_ip), port), timeout=timeout)
    try:
        if use_tls:
            context = ssl.create_default_context()
            sock = context.wrap_socket(sock, server_hostname=server_hostname)
        connection = http.client.HTTPConnection(
            _host_header_value(host), port, timeout=timeout
        )
        connection.sock = sock
        connection.request("GET", path, headers=headers)
        return _TransportResponse(connection.getresponse(), connection)
    except BaseException:
        sock.close()
        raise


# ----------------------------------------------------------------------
# Response handling helpers
# ----------------------------------------------------------------------


def _header(headers, name: str) -> Optional[str]:
    """Case-insensitive header lookup that works on HTTPMessage and dicts."""
    getter = getattr(headers, "get", None)
    if getter is not None:
        value = getter(name)
        if value is not None:
            return str(value)
    items = getattr(headers, "items", None)
    if items is None:
        return None
    lowered = name.lower()
    for key, value in items():
        if str(key).lower() == lowered:
            return str(value)
    return None


def _read_capped(response, cap: int) -> Optional[bytes]:
    """Read up to ``cap`` bytes; return None when the body exceeds the cap.

    Reading stops the moment the running total passes ``cap`` - a hostile
    body cannot force more than ``cap`` bytes into memory.
    """
    chunks: list = []
    total = 0
    while True:
        chunk = response.read(65536)
        if not chunk:
            break
        total += len(chunk)
        if total > cap:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _inflate(body: bytes, wbits: int, cap: int) -> tuple[Optional[bytes], str]:
    """Inflate ``body`` with one zlib window, refusing anything past ``cap``.

    Feeding the decompressor in fixed-size slices is what keeps the cap
    meaningful: the *inflated* size is bounded as it grows, so a body that is
    tiny on the wire cannot expand into memory before anyone notices.
    """
    try:
        inflater = zlib.decompressobj(wbits)
        chunks: list = []
        total = 0
        for start in range(0, len(body), 65536):
            chunk = inflater.decompress(body[start:start + 65536])
            total += len(chunk)
            if total > cap:
                return None, f"decompressed response body exceeds {cap} bytes"
            chunks.append(chunk)
        tail = inflater.flush()
        total += len(tail)
        if total > cap:
            return None, f"decompressed response body exceeds {cap} bytes"
        chunks.append(tail)
    except zlib.error:
        return None, "the compressed response could not be decoded"
    return b"".join(chunks), ""


def _decompress_capped(
    body: bytes, encoding: str, cap: int
) -> tuple[Optional[bytes], str]:
    """Inflate a body under ``content-encoding`` up to ``cap`` bytes.

    Returns ``(data, "")`` on success, or ``(None, reason)`` when the stream
    is unsupported, corrupt, or larger than ``cap`` once inflated. The caller
    reports ``reason`` verbatim, so the failure is named rather than turned
    into an empty page.
    """
    if encoding in ("gzip", "x-gzip"):
        attempts = (zlib.MAX_WBITS | 16,)
    elif encoding == "deflate":
        # Some servers label raw deflate (no zlib header) as "deflate", so
        # both spellings are tried before a stream is called undecodable.
        attempts = (zlib.MAX_WBITS, -zlib.MAX_WBITS)
    else:
        return None, f"content-encoding {encoding!r} is not supported"
    reason = ""
    for wbits in attempts:
        data, why = _inflate(body, wbits, cap)
        if data is not None:
            return data, ""
        reason = why
    return None, reason


# ----------------------------------------------------------------------
# HTML extraction
# ----------------------------------------------------------------------

# Elements whose contents are presentation, not page text. Skipped wholly: a
# script's body is JavaScript, a nav's body is chrome, and none of it is
# page prose the evidence pipeline should ever see. Only elements with end
# tags are listed - a void element here would raise the skip depth and never
# lower it.
_NON_CONTENT_TAGS = frozenset({
    "script", "style", "noscript", "template", "svg", "math",
    "nav", "header", "footer", "aside", "form", "iframe", "object",
    "embed", "canvas", "head", "title",
})


class _TextExtractor(HTMLParser):
    """Lenient HTML-to-text: keep visible prose, drop everything else.

    ``convert_charrefs=True`` decodes entities inside ``handle_data``, so the
    output is plain text rather than markup. Malformed HTML cannot crash the
    caller - ``_html_to_text`` swallows any parser exception and returns
    whatever was collected before the crash.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: list = []

    def handle_starttag(self, tag, attrs) -> None:
        if tag in _NON_CONTENT_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag) -> None:
        if tag in _NON_CONTENT_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def text(self) -> str:
        return " ".join(self._parts)


def _decode_body(body: bytes) -> str:
    try:
        return body.decode("utf-8")
    except UnicodeDecodeError:
        # latin-1 maps every byte value, so this always succeeds.
        return body.decode("latin-1")


def _html_to_text(body: bytes) -> str:
    """Reduce a raw HTML body to plain prose text.

    The body is untrusted: it is decoded leniently, fed to a parser that
    discards non-content elements, and returned as a string. No part of the
    body influences control flow beyond being parsed as markup.
    """
    extractor = _TextExtractor()
    try:
        extractor.feed(_decode_body(body))
        extractor.close()
    except Exception:
        # A parser crash on hostile markup yields whatever was collected
        # before the crash, never an exception to the caller.
        pass
    return extractor.text()


# ----------------------------------------------------------------------
# Sentence selection
# ----------------------------------------------------------------------

# Sentence boundary: punctuation followed by whitespace. Naive - it splits on
# abbreviations and decimals - but deterministic, dependency-free, and good
# enough to rank paragraphs of prose for an 800-character excerpt.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Word-ish query terms of at least two characters.
_QUERY_TERM = re.compile(r"[a-z0-9]{2,}")

# A minimal stoplist so common function words do not dominate relevance. Not
# a language model - just enough that "the" cannot match every sentence.
_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "any", "can",
    "her", "was", "one", "our", "out", "has", "have", "had", "how", "its",
    "may", "new", "now", "old", "see", "way", "who", "did", "get", "got",
    "let", "say", "she", "too", "use", "that", "with", "this", "they",
    "them", "from", "were", "been", "will", "would", "there", "what",
    "when", "your", "about", "into", "than", "then",
})


def _split_sentences(text: str) -> list:
    parts = _SENTENCE_BOUNDARY.split(text)
    return [part.strip() for part in parts if part.strip()]


def _query_terms(query: str) -> list:
    terms: list = []
    for token in _QUERY_TERM.findall((query or "").lower()):
        if token not in _STOPWORDS and token not in terms:
            terms.append(token)
    return terms


def _select_sentences(text: str, query: str, limit: int) -> str:
    """Pick query-relevant sentences within a hard character budget.

    Sentences are scored by how many distinct query terms they contain,
    greedily kept best-first (ties prefer earlier text), then reassembled in
    reading order and hard-truncated to ``limit``. When no sentence matches,
    the leading sentences are kept instead, so an unrelated page still yields
    honest text rather than an empty string.
    """
    if limit <= 0:
        return ""
    sentences = _split_sentences(text)
    if not sentences:
        return text[:limit]
    picked: list = []
    terms = _query_terms(query)
    if terms:
        scored = []
        for index, sentence in enumerate(sentences):
            lowered = sentence.lower()
            score = sum(
                1
                for term in terms
                if re.search(r"\b" + re.escape(term) + r"\b", lowered)
            )
            scored.append((score, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        running = 0
        for score, index in scored:
            if score <= 0:
                break
            sentence = sentences[index]
            if running + len(sentence) + 1 > limit:
                continue
            picked.append(index)
            running += len(sentence) + 1
    if not picked:
        # No relevant sentence (or none that fit): keep the leading text so
        # the caller still receives the page's actual opening rather than a
        # silent empty string.
        running = 0
        for index, sentence in enumerate(sentences):
            if running + len(sentence) + 1 > limit:
                break
            picked.append(index)
            running += len(sentence) + 1
    picked.sort()
    excerpt = " ".join(sentences[index] for index in picked)
    return excerpt[:limit]


# ----------------------------------------------------------------------
# The fetcher
# ----------------------------------------------------------------------


class PageFetcher:
    """Fetch and extract one web page under the security policy above.

    Dependency-injected by design: ``resolver`` and ``transport`` are
    replaceable, which is why the entire policy - scheme blocking, address
    blocking, DNS-rebinding pinning, redirect validation, size and time caps,
    header hygiene, and HTML extraction - is exercised in tests with zero
    live network access.

    Construction performs no I/O. The first network touch is inside
    :meth:`fetch`, and every failure mode is reported as a ``FetchResult``
    status rather than raised.
    """

    def __init__(
        self,
        *,
        resolver: Optional[Callable[[str], list]] = None,
        transport: Optional[Callable[..., object]] = None,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ) -> None:
        self._resolver = (
            resolver if resolver is not None else _default_resolver
        )
        self._transport = (
            transport if transport is not None else _default_transport
        )
        self._timeout_s = float(timeout_s)

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def fetch(self, url: str, *, query: str = "") -> FetchResult:
        """Fetch ``url`` and return extracted text, or an honest failure.

        Follows at most ``MAX_REDIRECTS`` redirects, re-validating every hop
        independently. Never raises: transport, DNS, TLS, size, and parsing
        failures all become a named status on the returned ``FetchResult``.
        """
        original = str(url or "")
        current = original
        for _hop in range(MAX_REDIRECTS + 1):
            target, blocked = _parse_target(current)
            if target is None:
                reason = blocked or STATUS_FETCH_ERROR
                return FetchResult(
                    url=original,
                    status=reason,
                    error=self._rejection_reason(reason, current),
                )
            # Resolve once, validate every answer, then pin the connection.
            try:
                addresses = self._resolver(target.host)
            except Exception as exc:
                return FetchResult(
                    url=original,
                    status=STATUS_FETCH_ERROR,
                    error=f"dns failed: {exc}",
                )
            if not addresses:
                return FetchResult(
                    url=original,
                    status=STATUS_FETCH_ERROR,
                    error="dns returned no addresses",
                )
            for address in addresses:
                if not _destination_allowed(
                    address, use_tls=(target.scheme == "https")
                ):
                    return FetchResult(
                        url=original,
                        status=STATUS_BLOCKED_ADDRESS,
                        error=(
                            f"{target.host} resolves to a non-global "
                            f"address ({address})"
                        ),
                    )
            pinned = addresses[0]
            headers = {"User-Agent": USER_AGENT, "Accept": ACCEPT_HEADER}
            body: Optional[bytes] = None
            content_type = ""
            try:
                with self._transport(
                    target.host,
                    target.port,
                    target.path,
                    use_tls=(target.scheme == "https"),
                    server_hostname=target.host,
                    pinned_ip=pinned,
                    timeout=self._timeout_s,
                    headers=headers,
                ) as response:
                    status_code = int(response.status)
                    if status_code in (301, 302, 303, 307, 308):
                        location = _header(response.headers, "Location")
                        if not location:
                            return FetchResult(
                                url=original,
                                status=STATUS_HTTP_ERROR,
                                error=(
                                    f"HTTP {status_code} without a Location "
                                    "header"
                                ),
                            )
                        # urljoin resolves relative Locations against the
                        # URL that produced this redirect, which is then
                        # re-parsed and re-validated on the next hop.
                        current = urllib.parse.urljoin(current, location)
                        continue
                    if status_code != 200:
                        return FetchResult(
                            url=original,
                            status=STATUS_HTTP_ERROR,
                            error=f"HTTP {status_code}",
                        )
                    content_type = (
                        _header(response.headers, "Content-Type") or ""
                    ).split(";")[0].strip().lower()
                    if content_type not in HTML_CONTENT_TYPES:
                        return FetchResult(
                            url=original,
                            status=STATUS_UNSUPPORTED_CONTENT,
                            error=(
                                f"content-type {content_type!r} is not html"
                            ),
                        )
                    body = _read_capped(response, MAX_RESPONSE_BYTES)
                    encoding = (
                        _header(response.headers, "Content-Encoding") or ""
                    ).strip().lower()
            except TimeoutError:
                return FetchResult(
                    url=original,
                    status=STATUS_TIMEOUT,
                    error=f"request timed out after {self._timeout_s:g}s",
                )
            except Exception as exc:
                return FetchResult(
                    url=original,
                    status=STATUS_FETCH_ERROR,
                    error=str(exc),
                )
            if body is None:
                return FetchResult(
                    url=original,
                    status=STATUS_TOO_LARGE,
                    error=(
                        f"response body exceeds {MAX_RESPONSE_BYTES} bytes"
                    ),
                )
            if encoding and encoding != "identity":
                # A cached response can arrive compressed even though this
                # module never asks for it. Inflating under the same cap is
                # what keeps "read this page" from silently becoming garbage.
                if encoding not in DECODABLE_ENCODINGS:
                    return FetchResult(
                        url=original,
                        status=STATUS_UNSUPPORTED_CONTENT,
                        error=(
                            f"content-encoding {encoding!r} is not supported"
                        ),
                    )
                inflated, reason = _decompress_capped(
                    body, encoding, MAX_RESPONSE_BYTES
                )
                if inflated is None:
                    too_large = "exceeds" in reason
                    return FetchResult(
                        url=original,
                        status=STATUS_TOO_LARGE if too_large
                        else STATUS_FETCH_ERROR,
                        error=reason,
                    )
                body = inflated
            text = _select_sentences(
                _html_to_text(body), query, MAX_EXTRACTED_CHARS
            )
            return FetchResult(
                url=original,
                status=STATUS_OK,
                text=text,
                final_url=current,
                content_type=content_type,
            )
        return FetchResult(
            url=original,
            status=STATUS_TOO_MANY_REDIRECTS,
            error=f"more than {MAX_REDIRECTS} redirects",
        )

    @staticmethod
    def _rejection_reason(status: str, url: str) -> str:
        """A human-readable reason for a structurally rejected URL."""
        if status == STATUS_BLOCKED_SCHEME:
            return "only http and https URLs may be fetched"
        if status == STATUS_BLOCKED_USERINFO:
            return "URLs with embedded credentials are not fetched"
        return f"url could not be parsed: {url!r}"
