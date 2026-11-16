import glob
import os
import re
import tempfile
import time
import pytest
from utils import *

# --cache-shared-prefix: the prompt before the first user message of a chat request (the system prompt and the tools of
# a harness) is kept as a prompt cache entry of its own, so a new conversation that starts with it is warm
#
# the model is hybrid (recurrent and attention layers, as production's): with --swa-full its attention has no sliding
# window, so a restored state is continued from its end without a context checkpoint

server = ServerPreset.tinylfm3omni()

MIN_TOKENS = 64


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0
    def drain(self):
        with open(self.path, errors="replace") as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content
    # the server logs from a thread of its own: a line may lag behind the response
    def wait_for(self, text, timeout=5.0):
        content = ""
        t0 = time.time()
        while text not in content and time.time() - t0 < timeout:
            content += self.drain()
            if text not in content:
                time.sleep(0.05)
        return content


@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinylfm3omni()
    server.n_ctx = 4096
    server.n_slots = 1
    server.n_predict = 4
    server.temperature = 0.0
    server.swa_full = True
    server.jinja = True
    server.cache_ram = 100
    server.cache_shared_prefix = MIN_TOKENS
    server.cache_dir = str(tmp_path / "cache")
    server.cache_dir_delay = 0
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather of a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The name of the city"}},
            "required": ["city"],
        },
    },
}]

SYSTEM = {"role": "system", "content": "You are a helpful assistant who answers questions about animals. " * 12}
SYSTEM_OTHER = {"role": "system", "content": "You are a terse assistant who answers questions about plants. " * 12}

# long enough that restoring a copy of it for its system prompt alone is skipped (less than an eighth of it is used)
LONG_STORY = "Tell me a long story about a fox who lives in the forest and meets a bear. " * 120

OTHER = [{"role": "user", "content": "The quick brown fox jumps over the lazy dog."}]


def chat(messages, tools=TOOLS, **kwargs):
    res = server.make_request("POST", "/chat/completions", data={
        "messages": messages,
        "max_tokens": 4,
        **({"tools": tools} if tools else {}),
        **kwargs,
    })
    assert res.status_code == 200, res.body
    return res.body


def cached(body):
    return body["usage"]["prompt_tokens_details"]["cached_tokens"]


SAVED = re.compile(r"saved the shared prefix ([0-9a-f]{16}) \((\d+) tokens\)")


def saved(log_text):
    return [(m.group(1), int(m.group(2))) for m in SAVED.finditer(log_text)]


# the number of tokens of the prompt before the first user message, from the rendered prompt
def n_before_user(messages, tools=TOOLS):
    res = server.make_request("POST", "/apply-template", data={"messages": messages, **({"tools": tools} if tools else {})})
    assert res.status_code == 200
    prompt = res.body["prompt"]
    at = prompt.index("<|im_start|>user")
    res = server.make_request("POST", "/tokenize", data={"content": prompt[:at], "add_special": True, "parse_special": True})
    assert res.status_code == 200
    return len(res.body["tokens"])


def entry_files():
    return sorted(glob.glob(os.path.join(server.cache_dir, "*", "*.bin")))


def restart():
    server.stop()
    os.replace(server.log_path, server.log_path + ".prev")
    server.start()


# a new conversation with the same system prompt and tools starts from the shared prefix, which stays cached
def test_new_conversation_reuses_prefix():
    global server
    server.start()
    log = LogReader(server.log_path)

    first = [SYSTEM, {"role": "user", "content": LONG_STORY}]
    chat(first)
    entries = saved(log.wait_for("saved the shared prefix"))
    assert len(entries) == 1
    id_prefix, n_prefix = entries[0]
    assert n_prefix >= MIN_TOKENS
    assert n_prefix == n_before_user(first)

    chat(OTHER)
    log.drain()

    second = [SYSTEM, {"role": "user", "content": "Which animals live in the desert?"}]
    body = chat(second)
    out = log.wait_for("__TEST_TAG_PROMPT_CACHE_KEEP__")
    assert f"restored the shared prefix {id_prefix} ({n_prefix} tokens)" in out
    assert "__TEST_TAG_PROMPT_CACHE_KEEP__" in out
    assert cached(body) == n_prefix
    assert saved(out) == []

    # the same answer as without the cache
    cold = chat(second, cache_prompt=False)
    assert cached(cold) == 0
    assert body["choices"][0]["message"] == cold["choices"][0]["message"]

    # the entry is kept: a third conversation uses it too
    chat(OTHER)
    log.drain()
    body = chat([SYSTEM, {"role": "user", "content": "Do foxes sleep at night?"}])
    assert f"restored the shared prefix {id_prefix}" in log.wait_for("restored the shared prefix")
    assert cached(body) == n_prefix


# another system prompt does not use the entry: it gets one of its own
def test_other_system_prompt_does_not_reuse():
    global server
    server.start()
    log = LogReader(server.log_path)

    chat([SYSTEM, {"role": "user", "content": "Which animals live in the desert?"}])
    entries = saved(log.wait_for("saved the shared prefix"))
    assert len(entries) == 1
    chat(OTHER)
    log.drain()

    body = chat([SYSTEM_OTHER, {"role": "user", "content": "Which animals live in the desert?"}])
    out = log.wait_for("saved the shared prefix")
    assert "restored the shared prefix" not in out
    assert cached(body) < entries[0][1]
    other = saved(out)
    assert len(other) == 1 and other[0][0] != entries[0][0]

    # without tools the prompt before the first user message differs too
    chat(OTHER)
    log.drain()
    body = chat([SYSTEM, {"role": "user", "content": "Which animals live in the desert?"}], tools=None)
    out = log.wait_for("saved the shared prefix")
    assert "restored the shared prefix" not in out
    assert len(saved(out)) == 1


# a prefix below the threshold, a prompt without a user message or without chat messages makes no entry
def test_no_entry_below_threshold():
    global server
    server.cache_shared_prefix = 100000
    server.start()
    log = LogReader(server.log_path)

    chat([SYSTEM, {"role": "user", "content": "Which animals live in the desert?"}])
    chat(OTHER)
    res = server.make_request("POST", "/completion", data={"prompt": SYSTEM["content"] + " Which animals live in the desert?"})
    assert res.status_code == 200
    time.sleep(0.2)
    out = log.drain()
    assert "shared prefix" not in out


def test_no_entry_without_user_message_or_template():
    global server
    server.start()
    log = LogReader(server.log_path)

    # no user message: no boundary
    chat([SYSTEM])
    # a raw prompt has no message spans
    res = server.make_request("POST", "/completion", data={"prompt": "<|im_start|>system\n" + SYSTEM["content"] + "<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"})
    assert res.status_code == 200
    time.sleep(0.2)
    assert "shared prefix" not in log.drain()


# a state that a request could not continue from its end is not saved: with a sliding window, the recurrent layers'
# position marks the state as lacking the window (see server_prompt_cache::n_usable)
def test_no_entry_when_not_restorable():
    global server
    server.swa_full = False
    server.start()
    log = LogReader(server.log_path)

    chat([SYSTEM, {"role": "user", "content": "Which animals live in the desert?"}])
    out = log.wait_for("could not be restored")
    assert "could not be restored, not saved" in out
    assert saved(out) == []


# a request that continues a long conversation restores the conversation, not the shared prefix
def test_continuation_restores_conversation_not_prefix():
    global server
    server.start()
    log = LogReader(server.log_path)

    first = [SYSTEM, {"role": "user", "content": LONG_STORY}]
    body = chat(first)
    n_first = body["usage"]["prompt_tokens"]
    _, n_prefix = saved(log.wait_for("saved the shared prefix"))[0]

    chat(OTHER)
    log.drain()

    body = chat(first + [
        {"role": "assistant", "content": "Once upon a time."},
        {"role": "user", "content": "And then what happened?"},
    ])
    out = log.wait_for("__TEST_TAG_PROMPT_CACHE_RESTORE__")
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in out
    assert "restored the shared prefix" not in out
    assert cached(body) > n_prefix
    assert cached(body) >= n_first // 2


# the entry is written to disk and survives a restart; a graceful stop writes it even before its delay
@pytest.mark.parametrize("delay", [0, 60])
def test_prefix_survives_restart(delay):
    global server
    server.cache_dir_delay = delay
    server.start()
    log = LogReader(server.log_path)

    chat([SYSTEM, {"role": "user", "content": LONG_STORY}])
    id_prefix, n_prefix = saved(log.wait_for("saved the shared prefix"))[0]

    restart() # SIGTERM
    assert any(os.path.basename(f) == f"{id_prefix}.bin" for f in entry_files()), entry_files()

    log = LogReader(server.log_path)
    body = chat([SYSTEM, {"role": "user", "content": "Which animals live in the desert?"}])
    out = log.wait_for("restored the shared prefix")
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" in out
    assert f"restored the shared prefix {id_prefix} ({n_prefix} tokens)" in out
    assert cached(body) == n_prefix

    # the copy read from disk is still a shared prefix: not superseded by the conversation that used it
    chat(OTHER)
    restart()
    assert any(os.path.basename(f) == f"{id_prefix}.bin" for f in entry_files()), entry_files()
    log = LogReader(server.log_path)
    body = chat([SYSTEM, {"role": "user", "content": "Do foxes sleep at night?"}])
    assert f"restored the shared prefix {id_prefix}" in log.wait_for("restored the shared prefix")
    assert cached(body) == n_prefix


# the entry is counted in the bounds and evicted like the others, least recently used first: a prefix that new
# conversations keep using stays, the conversations used less recently go first
def test_eviction_order_under_bound():
    global server
    server.cache_dir_max = 3 # MiB: the prefix, about five of these stories and a few small entries
    server.start()
    log = LogReader(server.log_path)

    # each entry is written before the next request: the order of the writes is the order of use
    n_staged = 0
    def chat_written(messages, **kwargs):
        nonlocal n_staged
        body = chat(messages, **kwargs)
        n_staged += 1
        t0 = time.time()
        while (st := server.make_request("GET", "/cache").body)["disk"]["persists"] < n_staged:
            assert time.time() - t0 < 10, st
            time.sleep(0.02)
        return body

    first = [SYSTEM, {"role": "user", "content": "Tell me a story about a fox."}]
    chat_written(first) # the prefix
    id_prefix, n_prefix = saved(log.wait_for("saved the shared prefix"))[0]

    def story(i):
        return [{"role": "user", "content": f"Story number {i}. " + "A bear walks through the snow looking for honey. " * 40}]

    # older conversations, then the prefix is used, then newer conversations fill the bound
    for i in range(2):
        chat_written(story(i), tools=None) # the conversation before it
    chat_written([SYSTEM, {"role": "user", "content": "Which animals live in the desert?"}])
    for i in range(2, 6):
        chat_written(story(i), tools=None)
    chat_written(OTHER, tools=None)

    # nine entries written (the prefix and the conversations that left the slot), the oldest evicted
    st = server.make_request("GET", "/cache").body
    assert st["disk"]["persists"] == 9 and st["disk"]["failures"] == 0
    assert st["disk"]["bytes"] <= st["disk"]["limit_bytes"]
    assert st["disk"]["entries"] < 9
    assert any(os.path.basename(f) == f"{id_prefix}.bin" for f in entry_files()), entry_files()

    restart()
    log = LogReader(server.log_path)

    # the first conversation was used before the prefix and is gone: its continuation starts from the prefix
    body = chat(first + [
        {"role": "assistant", "content": "Once upon a time."},
        {"role": "user", "content": "And then what happened?"},
    ])
    out = log.wait_for("restored the shared prefix")
    assert f"restored the shared prefix {id_prefix}" in out
    assert cached(body) == n_prefix

    # the newest conversation is still there
    body = chat(story(5) + [
        {"role": "assistant", "content": "Once upon a time."},
        {"role": "user", "content": "And then?"},
    ], tools=None)
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_DISK_LOAD__")
    assert cached(body) > n_prefix


# --cache-shared-prefix 0: no entry; a new conversation does not start from a long conversation's system prompt
def test_disabled():
    global server
    server.cache_shared_prefix = 0
    server.start()
    log = LogReader(server.log_path)

    first = [SYSTEM, {"role": "user", "content": LONG_STORY}]
    chat(first)
    n_prefix = n_before_user(first)
    chat(OTHER)
    body = chat([SYSTEM, {"role": "user", "content": "Which animals live in the desert?"}])
    time.sleep(0.2)
    assert "shared prefix" not in log.drain()
    assert cached(body) < n_prefix
    assert server.make_request("GET", "/cache").body["memory"]["entries"] == 2
