import sys
sys.path.insert(0, "C:\\PRIMORA PROJECT")

from PRIMORA.prime.router import Router
from PRIMORA.prime.contract import PrimeRequest
from PRIMORA.prime.capability import CapabilityRegistry


def test_router_returns_action():
    registry = CapabilityRegistry()
    router = Router(registry)
    req = PrimeRequest(action="calculate", payload={})
    assert router.route(req) == "calculate"


def test_router_returns_empty_string_for_empty_action():
    registry = CapabilityRegistry()
    router = Router(registry)
    req = PrimeRequest(action="", payload={})
    assert router.route(req) == ""


if __name__ == "__main__":
    test_router_returns_action()
    test_router_returns_empty_string_for_empty_action()
    print("All Router tests passed!")