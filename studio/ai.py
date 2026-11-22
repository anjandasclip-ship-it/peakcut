"""Talking to the AI service. You bring your own key, and choose the service and model in the dashboard
(Settings, AI) or in .env. What is saved from the dashboard is kept in ai.json in the data folder and wins.

    PEAKCUT_AI_PROVIDER   gemini | openai | xai | groq | openrouter | anthropic | ollama | custom
    PEAKCUT_AI_MODELS     one or more model ids, comma separated, tried in this order
    PEAKCUT_AI_KEY        your key (or PEAKCUT_AI_KEY_FILE, a file holding it)
    PEAKCUT_AI_BASE_URL   only for "custom" (any service that speaks the OpenAI chat format), or to override

Every service is asked for the same thing: JSON that follows a small schema. Gemini takes the schema directly.
The others are told the schema in the prompt and asked for a JSON object, which works with any chat model.

One picture can also be shown (`see`), which is how the webcam is looked for on a gaming stream. A model that
cannot look at pictures answers with something else, and that is simply taken as "it did not look"."""
import base64
import json
import os
import re
import threading
import time

import httpx

from . import config

# name -> (label shown on the pages, address, how it is spoken to, where to get a key, an example model id)
PROVIDERS = {
    "gemini": ("Gemini", "https://generativelanguage.googleapis.com/v1beta/models", "gemini", "https://aistudio.google.com/apikey", "gemini-2.5-flash"),
    "openai": ("OpenAI", "https://api.openai.com/v1", "openai", "https://platform.openai.com/api-keys", "gpt-4o-mini"),
    "xai": ("Grok", "https://api.x.ai/v1", "openai", "https://console.x.ai", "grok-4"),
    "groq": ("Groq", "https://api.groq.com/openai/v1", "openai", "https://console.groq.com/keys", "llama-3.3-70b-versatile"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "openai", "https://openrouter.ai/keys", "google/gemini-2.5-flash"),
    "anthropic": ("Claude", "https://api.anthropic.com/v1", "anthropic", "https://console.anthropic.com", ""),
    "ollama": ("Ollama (on this machine)", "http://127.0.0.1:11434/v1", "openai", "https://ollama.com", "llama3.1"),
    "custom": ("Custom (OpenAI-compatible)", "", "openai", "", ""),
}
ALIASES = {"google": "gemini", "grok": "xai", "claude": "anthropic", "local": "ollama"}
NO_KEY = {"ollama", "custom"}       # a model running on your own machine needs no key

# ---------------------------------------------------------------- the settings in use
_lock = threading.Lock()
_saved = {"at": None, "data": {}}
_limited_keys: dict = {}  # key_string -> {"time": float, "date": str, "reason": str}


def _today_str(t: float | None = None) -> str:
    import time
    return time.strftime("%Y-%m-%d", time.localtime(t or time.time()))


def is_limited(k: str) -> bool:
    info = _limited_keys.get(k)
    if not info:
        return False
    if info.get("date") != _today_str():
        _limited_keys.pop(k, None)
        return False
    return True


def mark_limited(k: str, reason: str = "Quota or rate limit reached"):
    import time
    _limited_keys[k] = {"time": time.time(), "date": _today_str(), "reason": reason}


def clear_limit(k: str | None = None):
    if k is None:
        _limited_keys.clear()
    else:
        _limited_keys.pop(k, None)


def _file():
    return config.DATA_DIR / "ai.json"


def _from_dashboard() -> dict:
    """What was saved from the dashboard, read again only when the file changes."""
    f = _file()
    try:
        at = f.stat().st_mtime_ns
    except OSError:
        at = 0
    if _saved["at"] != at:
        try:
            data = json.loads(f.read_text(encoding="utf-8")) if at else {}
        except (OSError, ValueError):
            data = {}
        _saved.update(at=at, data=data if isinstance(data, dict) else {})
    return _saved["data"]


def current() -> dict:
    """Provider, models, address and keys: what the dashboard saved, or else what .env says."""
    d = _from_dashboard()
    p = str(d.get("provider") or config.AI_PROVIDER or "").strip().lower()
    dash_has_provider = bool(d.get("provider"))

    raw_keys = d.get("keys")
    keys_list = []
    if isinstance(raw_keys, list):
        for item in raw_keys:
            if isinstance(item, dict) and item.get("key"):
                keys_list.append({"id": str(item.get("id") or len(keys_list) + 1),
                                  "key": str(item["key"]).strip(),
                                  "added_at": float(item.get("added_at") or 0.0)})
            elif isinstance(item, str) and item.strip():
                keys_list.append({"id": str(len(keys_list) + 1),
                                  "key": item.strip(),
                                  "added_at": 0.0})
    single_key = str(d["key"]).strip() if dash_has_provider and "key" in d else (config.ai_key() if not dash_has_provider else "")
    if single_key and not any(x["key"] == single_key for x in keys_list):
        if not dash_has_provider or not keys_list:
            keys_list.insert(0, {"id": "1", "key": single_key, "added_at": 0.0})
    
    active_id = str(d.get("active_key_id") or "")
    active_key = ""
    for k_item in keys_list:
        if k_item["id"] == active_id:
            active_key = k_item["key"]
            break
    if not active_key and keys_list:
        active_key = keys_list[0]["key"]
        active_id = keys_list[0]["id"]
    if not active_key:
        active_key = single_key

    return {"provider": ALIASES.get(p, p),
            "models": list(d["models"]) if d.get("provider") else list(config.AI_MODELS),
            "base_url": (str(d.get("base_url") or "") if d.get("provider") else config.AI_BASE_URL).rstrip("/"),
            "key": active_key,
            "keys": keys_list,
            "active_key_id": active_id}


def provider() -> str:
    return current()["provider"]


def models() -> list:
    return current()["models"]


def key() -> str:
    return current()["key"]


def get_usable_keys() -> list:
    c = current()
    all_keys = [x["key"] for x in c.get("keys", []) if x.get("key")]
    if not all_keys and c.get("key"):
        all_keys = [c["key"]]
    if not all_keys:
        return []
    act = c.get("key")
    ordered = [act] + [k for k in all_keys if k != act] if act in all_keys else all_keys
    usable = [k for k in ordered if not is_limited(k)]
    return usable or ordered


def label() -> str:
    return PROVIDERS.get(provider(), ("",))[0]


def base_url() -> str:
    c = current()
    return c["base_url"] or PROVIDERS.get(c["provider"], ("", ""))[1]


def ready() -> bool:
    c = current()
    return bool(c["provider"] in PROVIDERS and c["models"] and base_url() and (c["key"] or c["provider"] in NO_KEY))


def public() -> dict:
    """What the settings page may see. The key itself never leaves the server: only whether there is one, and
    its last four characters so the owner can tell which key it is."""
    c = current()
    return {"provider": c["provider"], "models": c["models"], "base_url": c["base_url"], "has_key": bool(c["key"]),
            "key_hint": c["key"][-4:] if len(c["key"]) >= 12 else "", "ready": ready(), "label": label(),
            "keys_count": len(c.get("keys", [])),
            "saved_here": bool(_from_dashboard().get("provider")),
            "providers": [{"id": k, "label": v[0], "address": v[1], "needs_key": k not in NO_KEY, "key_url": v[3], "example": v[4]}
                          for k, v in PROVIDERS.items()]}


def _save_keys_data(new_keys_list: list, new_active_id: str | None = None):
    d = _from_dashboard()
    p = str(d.get("provider") or config.AI_PROVIDER or "").strip().lower()
    p = ALIASES.get(p, p)
    ms = list(d.get("models") or config.AI_MODELS)
    base = str(d.get("base_url") or config.AI_BASE_URL or "").rstrip("/")
    active_key = ""
    if new_active_id:
        for item in new_keys_list:
            if item["id"] == new_active_id:
                active_key = item["key"]
                break
    if not active_key and new_keys_list:
        active_key = new_keys_list[0]["key"]
        new_active_id = new_keys_list[0]["id"]
    with _lock:
        f = _file()
        tmp = f.with_suffix(".tmp")
        payload = {
            "provider": p,
            "models": ms,
            "base_url": base,
            "key": active_key,
            "keys": [{"id": x["id"], "key": x["key"], "added_at": x.get("added_at", 0.0)} for x in new_keys_list],
            "active_key_id": new_active_id or ""
        }
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(f)
        _saved["at"] = None


def keys_public() -> dict:
    import time
    c = current()
    raw_keys = c.get("keys", [])
    active_id = c.get("active_key_id")
    formatted_keys = []
    limited_keys = []
    for idx, item in enumerate(raw_keys):
        k_val = item["key"]
        k_id = str(item["id"])
        is_act = (k_id == active_id) or (not active_id and idx == 0)
        lim = is_limited(k_val)
        lim_info = _limited_keys.get(k_val) if lim else None
        lim_time_str = ""
        if lim_info and lim_info.get("time"):
            lim_time_str = time.strftime("%H:%M", time.localtime(lim_info["time"]))
        hint = k_val[-4:] if len(k_val) >= 8 else k_val
        key_obj = {
            "id": k_id,
            "index": idx + 1,
            "name": f"Key {idx + 1}",
            "label": f"Key {idx + 1}",
            "hint": hint,
            "key_hint": hint,
            "active": is_act,
            "is_active": is_act,
            "limited": lim,
            "is_limited": lim,
            "limited_at": lim_time_str,
            "limited_reason": lim_info.get("reason", "") if lim_info else "",
            "time": lim_info.get("time", 0.0) if lim_info else 0.0,
            "date": lim_info.get("date", "") if lim_info else "",
            "reason": lim_info.get("reason", "") if lim_info else "",
            "added_at": item.get("added_at") or 0.0
        }
        formatted_keys.append(key_obj)
        if lim:
            limited_keys.append(key_obj)
    return {
        "provider": c["provider"],
        "label": label(),
        "ready": ready(),
        "total": len(formatted_keys),
        "keys": formatted_keys,
        "limited_keys": limited_keys,
        "active_key_id": active_id,
        "active_index": next((k["index"] for k in formatted_keys if k["active"]), 1 if formatted_keys else 0)
    }


def add_key(new_key_str: str) -> dict:
    import time
    new_key_str = str(new_key_str or "").strip()[:400]
    if not new_key_str:
        raise ValueError("Paste an API key.")
    c = current()
    keys_list = list(c.get("keys", []))
    if any(item.get("key") == new_key_str for item in keys_list):
        raise ValueError("This API key is already in the list.")
    new_id = str(int(time.time() * 1000) % 1000000)
    existing_ids = {str(item.get("id")) for item in keys_list}
    counter = 1
    while new_id in existing_ids:
        new_id = str(len(keys_list) + counter)
        counter += 1
    keys_list.append({"id": new_id, "key": new_key_str, "added_at": time.time()})
    active_id = c.get("active_key_id") or new_id
    _save_keys_data(keys_list, active_id)
    clear_limit(new_key_str)
    return keys_public()


def remove_key(key_id: str) -> dict:
    c = current()
    keys_list = [item for item in c.get("keys", []) if str(item.get("id")) != str(key_id)]
    active_id = c.get("active_key_id")
    if active_id == str(key_id):
        active_id = str(keys_list[0]["id"]) if keys_list else ""
    _save_keys_data(keys_list, active_id)
    return keys_public()


def activate_key(key_id: str) -> dict:
    c = current()
    found = False
    for item in c.get("keys", []):
        if str(item.get("id")) == str(key_id):
            found = True
            break
    if not found:
        raise ValueError("Key not found.")
    _save_keys_data(c.get("keys", []), str(key_id))
    return keys_public()


def reset_key_limit(key_id: str) -> dict:
    c = current()
    for item in c.get("keys", []):
        if str(item.get("id")) == str(key_id):
            clear_limit(item["key"])
            break
    return keys_public()


def test_single_key(key_id: str) -> dict:
    c = current()
    raw_keys = c.get("keys", [])
    target = None
    for item in raw_keys:
        if str(item.get("id")) == str(key_id):
            target = item
            break
    if not target or not target.get("key"):
        raise ValueError("Key not found.")
    ms = models()
    if not ms:
        raise ValueError("No model configured.")
    k_val = target["key"]
    how = PROVIDERS[provider()][2]
    fn = {"gemini": _gemini, "openai": _openai, "anthropic": _anthropic}[how]
    out, sent, wrote = fn(ms[0], 'Answer with a JSON array that holds the single string "ok".',
                          {"type": "ARRAY", "items": {"type": "STRING"}}, k=k_val)
    clear_limit(k_val)
    return {"ok": True, "model": ms[0], "tokens": sent + wrote, "answer": str(out)[:80]}


def _set_active_key_silent(key_id: str):
    try:
        activate_key(key_id)
    except Exception:
        pass


def save(provider_id: str, model_ids, address: str = "", new_key=None) -> dict:
    """Save the choice made in the dashboard. `new_key` None keeps the key that is already there (the page never
    has it to send back); an empty string removes it."""
    p = ALIASES.get(str(provider_id or "").strip().lower(), str(provider_id or "").strip().lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    if isinstance(model_ids, str):
        model_ids = model_ids.split(",")
    ms = [str(m).strip()[:120] for m in (model_ids or []) if str(m).strip()][:5]
    if not ms:
        raise ValueError("Enter the model to use.")
    if any(not re.fullmatch(r"[\w.\-:/@]+", m) for m in ms):
        raise ValueError("A model id has letters, digits and . - : / only.")
    address = str(address or "").strip().rstrip("/")[:300]
    if address and not re.fullmatch(r"https?://[^\s]+", address):
        raise ValueError("The address has to start with http:// or https://")
    if p == "custom" and not address:
        raise ValueError("A custom service needs its address, for example http://127.0.0.1:1234/v1")
    old = current()
    k = old["key"] if new_key is None and old["provider"] == p else str(new_key or "").strip()[:400]
    if not k and p not in NO_KEY:
        raise ValueError("Paste the key for this service.")
    
    dash = _from_dashboard()
    dash_has = bool(dash.get("provider"))
    keys_list = [dict(x) for x in dash.get("keys", [])] if (dash_has and dash.get("provider") == p) else []
    active_id = str(dash.get("active_key_id") or "") if (dash_has and dash.get("provider") == p) else ""
    if k:
        matched = False
        for item in keys_list:
            if item.get("key") == k:
                active_id = item["id"]
                matched = True
                break
        if not matched:
            new_id = str(len(keys_list) + 1)
            keys_list.insert(0, {"id": new_id, "key": k, "added_at": time.time()})
            active_id = new_id

    with _lock:
        f = _file()
        tmp = f.with_suffix(".tmp")
        payload = {
            "provider": p,
            "models": ms,
            "base_url": address,
            "key": k,
            "keys": [{"id": x["id"], "key": x["key"], "added_at": x.get("added_at", 0.0)} for x in keys_list],
            "active_key_id": active_id or ""
        }
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)        # the key is only for whoever runs the server
        except OSError:
            pass
        tmp.replace(f)
        _saved["at"] = None
    return public()


def list_models(provider_id=None, address=None, with_key=None) -> list:
    """The model ids a service offers, so the settings page can show a list to pick from."""
    c = current()
    p = ALIASES.get(str(provider_id or c["provider"]).lower(), str(provider_id or c["provider"]).lower())
    if p not in PROVIDERS:
        raise ValueError("Choose one of the AI services in the list.")
    k = str(with_key or "").strip() or (c["key"] if c["provider"] == p else "")
    base = str(address or "").strip().rstrip("/") or (c["base_url"] if c["provider"] == p else "") or PROVIDERS[p][1]
    if not base:
        raise ValueError("Enter the service's address first.")
    how = PROVIDERS[p][2]
    if how == "gemini":
        r = httpx.get(base, params={"pageSize": 200}, headers={"x-goog-api-key": k}, timeout=20)
    elif how == "anthropic":
        r = httpx.get(f"{base}/models", params={"limit": 200}, headers={"x-api-key": k, "anthropic-version": "2023-06-01"}, timeout=20)
    else:
        r = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {k}"} if k else {}, timeout=20)
    if r.status_code != 200:
        raise AIError(f"HTTP {r.status_code} {r.text[:200].replace(k or chr(0), '<key>')}")
    data = r.json()
    if how == "gemini":
        found = [m.get("name", "").split("/")[-1] for m in data.get("models", [])
                 if "generateContent" in (m.get("supportedGenerationMethods") or [])]
    else:
        found = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
    return sorted({m for m in found if m})[:400]


def test() -> dict:
    """One tiny request with the settings in use, to prove the key, the address and the model all work."""
    ms = models()
    if not ready():
        raise ValueError("Save a service, a model and a key first.")
    out, sent, wrote = call(ms[0], 'Answer with a JSON array that holds the single string "ok".', {"type": "ARRAY", "items": {"type": "STRING"}})
    return {"ok": True, "model": ms[0], "tokens": sent + wrote, "answer": str(out)[:80]}


class AIError(RuntimeError):
    pass


problem = {"text": "", "at": 0.0}       # why the AI last failed, for the dashboard; cleared when it answers again


def explain(e) -> str:
    """What went wrong with an AI call, in words the owner can act on."""
    text, low = str(e), str(e).lower()
    name = (models() or ["?"])[0]
    if "404" in text or "not found" in low or "does not exist" in low or "unknown model" in low or "not_found" in low:
        return f"{label()} doesn't know the model \"{name}\". Open Settings, AI, press Find models and pick one from the list."
    if "401" in text or "403" in text or "api key" in low or "api_key" in low or "unauthorized" in low or "permission" in low:
        return f"{label()} refused the key. Paste a working key under Settings, AI."
    if "429" in text or "quota" in low or "rate limit" in low or "exhausted" in low:
        return f"{label()} says the key has hit its limit or has no credit left. Check the account with {label()}, or try again later."
    if "connect" in low or "timed out" in low or "timeout" in low or "getaddrinfo" in low or "refused" in low or "nodename" in low or "servname" in low or "nameresolutionerror" in low or "name resolution" in low:
        return f"PeakCut couldn't reach {label()}. Check the internet connection" + (" and that Ollama is running." if provider() == "ollama" else ".")
    return ""


def note_problem(e):
    import time
    why = explain(e)
    if why:
        problem.update(text=why, at=time.time())


def clear_problem():
    problem.update(text="", at=0.0)


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e).lower()
    if "429" in msg or "resource_exhausted" in msg or "resourceexhausted" in msg:
        return True
    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
        return True
    if "rate limit" in msg or "rate_limit" in msg or "ratelimit" in msg or "quota" in msg:
        return True
    return False


def _fail(r, model, k=None):
    use_k = k or key()
    text = r.text[:300].replace(use_k or "\0", "<key>")
    raise AIError(f"HTTP {r.status_code} {text}")


# ---------------------------------------------------------------- the schema, said in words for chat models
def _json_schema(s):
    """Gemini's schema spelling (ARRAY, OBJECT...) as ordinary JSON Schema."""
    if isinstance(s, dict):
        return {k: (v.lower() if k == "type" and isinstance(v, str) else _json_schema(v)) for k, v in s.items()}
    if isinstance(s, list):
        return [_json_schema(x) for x in s]
    return s


def _wrapped(prompt, schema):
    return (prompt + "\n\nAnswer with JSON only, no other text and no code fences: one object with a single key "
            "\"items\", whose value follows this JSON Schema:\n" + json.dumps(_json_schema(schema)))


def _json_chunks(text: str):
    """Every balanced {...} or [...] in the text, in the order they appear."""
    out, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start:i + 1])
                start = None
            elif depth < 0:
                depth = 0
    return out


def _parse(text):
    """The JSON in an answer, even if the model wrapped it in a code fence or a sentence.

    A model that is not careful sometimes puts a sentence in front of the JSON,
    or a second object after it, or forgets a comma. The JSON is looked for
    inside the answer rather than demanded of it."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    out = None
    try:
        out = json.loads(text)
    except ValueError:
        for chunk in _json_chunks(text):
            try:
                out = json.loads(chunk)
                break
            except ValueError:
                continue
        if out is None:
            m = re.search(r"[\[{].*[\]}]", text, re.S)
            if not m:
                raise AIError("The AI did not answer with JSON.")
            try:
                out = json.loads(m.group(0))
            except ValueError:
                raise AIError("The AI did not answer with JSON.")
    if isinstance(out, dict) and "items" in out:
        out = out["items"]
    return out


# ---------------------------------------------------------------- one call
def call(model: str, prompt: str, schema: dict, timeout: float | None = None):
    """Ask one model. Returns (the parsed JSON answer, tokens sent, tokens written back).
    Automatically switches to backup keys if a key hits its daily limit/quota."""
    how = PROVIDERS[provider()][2]
    fn = {"gemini": _gemini, "openai": _openai, "anthropic": _anthropic}[how]
    usable_keys = get_usable_keys()
    if not usable_keys:
        return fn(model, prompt, schema, timeout, k="")
    last_err = None
    for k_val in usable_keys:
        try:
            res = fn(model, prompt, schema, timeout, k=k_val)
            c = current()
            if c.get("key") != k_val:
                for item in c.get("keys", []):
                    if item.get("key") == k_val:
                        _set_active_key_silent(item["id"])
                        break
            return res
        except Exception as e:
            last_err = e
            if _is_rate_limit(e):
                hint = k_val[-4:] if len(k_val) >= 8 else k_val
                mark_limited(k_val, explain(e) or "Quota or rate limit reached")
                print(f"  PeakCut: API key (ends in ...{hint}) reached limit. Switching to backup key...", flush=True)
                continue
            raise e
    if last_err:
        raise last_err


def _dispatch_see(how, model, prompt, b64, timeout=None, k=None):
    use_k = k if k is not None else key()
    if how == "gemini":
        body = {"contents": [{"role": "user", "parts": [
            {"text": prompt},
            {"inline_data": {"mimeType": "image/jpeg", "data": b64}}]}]}
        r = httpx.post(f"{base_url()}/{model}:generateContent", json=body,
                       headers={"x-goog-api-key": use_k}, timeout=timeout or 120)
        if r.status_code != 200:
            _fail(r, model, k=use_k)
        data = r.json()
        text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        u = data.get("usageMetadata", {})
        sent = int(u.get("promptTokenCount") or 0)
        return text, sent, max(0, int(u.get("totalTokenCount") or 0) - sent)
    if how == "anthropic":
        body = {"model": model, "max_tokens": 300, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": prompt}]}]}
        r = httpx.post(f"{base_url()}/messages", json=body, timeout=timeout or 120,
                       headers={"x-api-key": use_k, "anthropic-version": "2023-06-01"})
        if r.status_code != 200:
            _fail(r, model, k=use_k)
        data = r.json()
        text = "".join(p.get("text", "") for p in data.get("content") or [] if p.get("type") == "text")
        u = data.get("usage") or {}
        return text, int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
    headers = {"Authorization": f"Bearer {use_k}"} if use_k else {}
    body = {"model": model, "stream": False, "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}]}]}
    r = httpx.post(f"{base_url()}/chat/completions", json=body, headers=headers, timeout=timeout or 120)
    if r.status_code != 200:
        _fail(r, model, k=use_k)
    data = _openai_body(r)
    text = (data["choices"][0]["message"].get("content") or "")
    u = data.get("usage") or {}
    return text, int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def see(prompt: str, image: bytes, timeout: float | None = None) -> tuple[str, int, int]:
    """Ask one model about one picture: (the answer as text, tokens in, tokens out).
    Automatically switches to backup keys if a key hits its daily limit/quota."""
    if not ready() or not image or not models():
        return "", 0, 0
    model = models()[0]
    b64 = base64.b64encode(image).decode()
    how = PROVIDERS[provider()][2]
    usable_keys = get_usable_keys()
    if not usable_keys:
        return _dispatch_see(how, model, prompt, b64, timeout, k="")
    last_err = None
    for k_val in usable_keys:
        try:
            return _dispatch_see(how, model, prompt, b64, timeout, k=k_val)
        except Exception as e:
            last_err = e
            if _is_rate_limit(e):
                mark_limited(k_val, explain(e) or "Quota or rate limit reached")
                continue
            raise e
    if last_err:
        raise last_err
    return "", 0, 0


def _gemini(model, prompt, schema, timeout=None, k=None):
    use_k = k if k is not None else key()
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json", "responseSchema": schema}}
    r = httpx.post(f"{base_url()}/{model}:generateContent", json=body, headers={"x-goog-api-key": use_k}, timeout=timeout or 120)
    if r.status_code != 200:
        _fail(r, model, k=use_k)
    data = r.json()
    text = "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
    u = data.get("usageMetadata", {})
    sent = int(u.get("promptTokenCount") or 0)
    return json.loads(text), sent, max(0, int(u.get("totalTokenCount") or 0) - sent)   # the answer, and any thinking billed with it


def _openai_body(r):
    """The answer, whether the service sent plain JSON or an event stream.

    Some OpenAI-compatible servers answer with a stream of events unless the
    request says otherwise, so the answer is read from whichever shape arrived.
    A model that answers with a tool call instead of a message leaves the JSON in
    the call's arguments, so that is used too."""
    try:
        data = r.json()
    except ValueError:
        data = _from_events(r.text)
    for ch in data.get("choices") or []:
        msg = ch.get("message") or {}
        if (msg.get("content") or "").strip():
            continue
        for call in msg.get("tool_calls") or []:
            args = (call.get("function") or {}).get("arguments")
            if args:
                msg["content"] = args
                break
    return data


def _from_events(text: str) -> dict:
    """Read an answer out of a server-sent event stream."""
    parts, usage, calls = [], {}, {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if not chunk or chunk == "[DONE]":
            continue
        try:
            d = json.loads(chunk)
        except ValueError:
            continue
        usage = d.get("usage") or usage
        for ch in d.get("choices") or []:
            for piece in ((ch.get("message") or {}), (ch.get("delta") or {})):
                if piece.get("content"):
                    parts.append(piece["content"])
                for call in piece.get("tool_calls") or []:
                    i = call.get("index", 0)
                    calls[i] = calls.get(i, "") + ((call.get("function") or {}).get("arguments") or "")
    message = {"content": "".join(parts)}
    if not message["content"] and calls:
        message["content"] = "".join(calls.values())
    return {"choices": [{"message": message}], "usage": usage}


def _openai(model, prompt, schema, timeout=None, k=None):
    use_k = k if k is not None else key()
    headers = {"Authorization": f"Bearer {use_k}"} if use_k else {}
    body = {"model": model, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}], "stream": False}
    # Most services accept these two extras; a few models refuse one of them, so they are dropped on a refusal.
    extras = [{"temperature": 0.4, "response_format": {"type": "json_object"}}, {"response_format": {"type": "json_object"}}, {}]
    r = None
    for extra in extras:
        r = httpx.post(f"{base_url()}/chat/completions", json={**body, **extra}, headers=headers, timeout=timeout or 180)
        if r.status_code not in (400, 422):
            break
    if r.status_code != 200:
        _fail(r, model, k=use_k)
    data = _openai_body(r)
    u = data.get("usage") or {}
    return _parse(data["choices"][0]["message"].get("content")), int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)


def _anthropic(model, prompt, schema, timeout=None, k=None):
    use_k = k if k is not None else key()
    body = {"model": model, "max_tokens": 8192, "messages": [{"role": "user", "content": _wrapped(prompt, schema)}]}
    r = httpx.post(f"{base_url()}/messages", json=body, timeout=timeout or 180,
                   headers={"x-api-key": use_k, "anthropic-version": "2023-06-01"})
    if r.status_code != 200:
        _fail(r, model, k=use_k)
    data = r.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage") or {}
    return _parse(text), int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
