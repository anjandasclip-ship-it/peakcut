"""Tests for multi-API key rotation, automatic failover when limited, and keys management."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tempfile
from unittest.mock import patch, MagicMock
import httpx
from fastapi.testclient import TestClient

from studio import config, ai, server

ORIGIN = {"origin": "http://127.0.0.1:8765"}

def run_tests():
    with tempfile.TemporaryDirectory() as tmp:
        config.DATA_DIR = Path(tmp)
        ai._saved["at"] = None
        ai._saved["data"] = {}
        ai.clear_limit()

        # 1. Start with no keys
        ai.save("gemini", ["gemini-2.5-flash"], new_key="initial-key-1234567890ab")
        pub = ai.keys_public()
        assert len(pub["keys"]) == 1, f"Expected 1 key, got {len(pub['keys'])}"
        assert pub["keys"][0]["label"] == "Key 1"
        assert pub["keys"][0]["is_active"] is True
        print("PASS initial key saved and labeled Key 1")

        # 2. Add Key 2 and Key 3
        pub = ai.add_key("second-key-0987654321cd")
        pub = ai.add_key("third-key-1122334455ef")
        assert len(pub["keys"]) == 3, f"Expected 3 keys, got {len(pub['keys'])}"
        assert pub["keys"][1]["label"] == "Key 2"
        assert pub["keys"][2]["label"] == "Key 3"
        k2_id = pub["keys"][1]["id"]
        k3_id = pub["keys"][2]["id"]
        print("PASS multiple keys added and numbered Key 1, Key 2, Key 3")

        # 3. Test activate_key
        pub = ai.activate_key(k2_id)
        assert pub["active_key_id"] == k2_id
        assert pub["keys"][1]["is_active"] is True
        assert pub["keys"][0]["is_active"] is False
        print("PASS activate key switches active key correctly")

        # 4. Test rate-limit tracking and get_usable_keys
        ai.mark_limited("second-key-0987654321cd", "Rate limit exceeded (HTTP 429)")
        pub = ai.keys_public()
        assert len(pub["limited_keys"]) == 1
        assert pub["limited_keys"][0]["label"] == "Key 2"
        assert pub["keys"][1]["is_limited"] is True

        usable = ai.get_usable_keys()
        assert "second-key-0987654321cd" not in usable
        assert len(usable) == 2
        print("PASS limited key excluded from usable keys and listed in limited_keys")

        # 5. Test reset limit
        pub = ai.reset_key_limit(k2_id)
        assert len(pub["limited_keys"]) == 0
        assert pub["keys"][1]["is_limited"] is False
        print("PASS reset limit restores key to available state")

        # 6. Test automatic failover in call()
        # k2 is active. Let's make k2 fail with 429, and k3 succeed!
        ai.activate_key(k2_id)
        ai.clear_limit()

        call_keys_seen = []
        def fake_post(url, *args, **kwargs):
            k_used = (kwargs.get("headers", {}).get("x-goog-api-key") or
                      kwargs.get("headers", {}).get("authorization", "").replace("Bearer ", "").strip() or
                      kwargs.get("params", {}).get("key"))
            call_keys_seen.append(k_used)
            req = httpx.Request("POST", url)
            if k_used == "second-key-0987654321cd":
                # Return HTTP 429 response so _gemini triggers _fail
                return httpx.Response(429, request=req, text='{"error": {"code": 429, "message": "RESOURCE_EXHAUSTED"}}')
            # Success
            return httpx.Response(200, request=req, json={"candidates": [{"content": {"parts": [{"text": '{"clips": []}'}]}}]})

        with patch("httpx.post", side_effect=fake_post):
            data, tin, tout = ai.call("gemini-2.5-flash", "Find clips", {"type": "OBJECT", "properties": {"clips": {"type": "ARRAY"}}})
            assert data == {"clips": []}
            assert len(call_keys_seen) >= 2
            # Key 2 should have been marked limited
            pub = ai.keys_public()
            assert any(lk["key_hint"] == "21cd" for lk in pub["limited_keys"])
            print("PASS call() automatically failed over from 429 limited key to next key")

        # 7. Test HTTP endpoints
        config.ACCOUNTS = False
        c = TestClient(server.app, base_url="http://127.0.0.1:8765")
        r = c.get("/api/ai/keys", headers=ORIGIN)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        keys_data = r.json()
        assert "keys" in keys_data
        assert "limited_keys" in keys_data

        # Add key via POST
        r = c.post("/api/ai/keys", json={"key": "fourth-key-9988776655zz"}, headers=ORIGIN)
        assert r.status_code == 200
        pub = r.json()
        assert len(pub["keys"]) == 4
        new_k = pub["keys"][-1]
        assert new_k["label"] == "Key 4"

        # Delete key via DELETE
        r = c.delete(f"/api/ai/keys/{new_k['id']}", headers=ORIGIN)
        assert r.status_code == 200
        assert len(r.json()["keys"]) == 3

        print("PASS HTTP API endpoints /api/ai/keys work as expected")

    print("\nALL MULTI-KEY TESTS PASSED!")

if __name__ == "__main__":
    run_tests()
