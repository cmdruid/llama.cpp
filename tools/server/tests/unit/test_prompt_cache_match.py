import os
import tempfile
import time
import pytest
from utils import *

# the prompt cache keeps a conversation that a request does not continue, and restores an entry
# only for a request that continues it up to a usable restore point

server = ServerPreset.tinyllama2()


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
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_predict = 4
    server.temperature = 0.0
    server.cache_ram = 100
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)

# shares well over half of LONG_PROMPT, then goes elsewhere
SHORT_PROMPT = LONG_PROMPT[:len(LONG_PROMPT) * 3 // 4] + " The end."


def complete(prompt):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body


def n_tokens(prompt):
    res = server.make_request("POST", "/tokenize", data={"content": prompt, "add_special": True})
    assert res.status_code == 200
    return len(res.body["tokens"])


# a short request that shares most of a long conversation must not destroy its tail
def test_short_request_does_not_truncate_conversation():
    global server
    server.start()
    log = LogReader(server.log_path)

    n_long = n_tokens(LONG_PROMPT)

    body = complete(LONG_PROMPT)
    assert body["timings"]["cache_n"] == 0

    # the only slot holds the long conversation, the short request shares more than half of it
    body = complete(SHORT_PROMPT)
    assert body["timings"]["cache_n"] > n_long // 2

    # the long conversation is continued: it is restored from the cache
    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_RESTORE__")
    assert body["timings"]["cache_n"] >= n_long


# a request that branches off a cached conversation gets a copy, the conversation stays in the cache
def test_branch_keeps_cached_conversation():
    global server
    server.start()
    log = LogReader(server.log_path)

    n_long = n_tokens(LONG_PROMPT)

    complete(LONG_PROMPT)

    # an unrelated request takes the slot, the conversation goes to the cache
    complete("The quick brown fox jumps over the lazy dog.")
    log.drain()

    # branch: shares most of the conversation, then differs - restored from a copy of the entry
    body = complete(SHORT_PROMPT)
    out = log.wait_for("__TEST_TAG_PROMPT_CACHE_KEEP__")
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in out
    assert "__TEST_TAG_PROMPT_CACHE_KEEP__" in out
    assert body["timings"]["cache_n"] > n_long // 2

    # an unrelated request again: the branch goes to the cache, the original conversation is still there
    complete("The quick brown fox jumps over the lazy dog.")
    log.drain()

    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_RESTORE__")
    assert body["timings"]["cache_n"] >= n_long


# a request that shares only a small part of a large cached state does not restore it
def test_small_overlap_does_not_restore():
    global server
    server.start()
    log = LogReader(server.log_path)

    complete(LONG_PROMPT)
    complete("The quick brown fox jumps over the lazy dog.")
    log.drain()

    body = complete("Once upon a time there was a cat.")
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" not in log.drain()

    # the conversation is still cached and resumable
    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_RESTORE__")


# hybrid model (short-conv recurrent state + SWA attention): a branch (an edited last user message) is restored from
# the context checkpoint before the divergence point, and the conversation it branched from stays resumable
def test_hybrid_branch_restores_from_checkpoint():
    global server
    server = ServerPreset.tinylfm3omni()
    server.n_ctx = 4096
    server.n_slots = 1
    server.n_predict = 4
    server.temperature = 0.0
    server.cache_ram = 100
    server.jinja = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    server.start()
    log = LogReader(server.log_path)

    def chat(messages):
        res = server.make_request("POST", "/chat/completions", data={
            "messages": messages,
            "max_tokens": 4,
        })
        assert res.status_code == 200
        return res.body

    head = [
        {"role": "system", "content": "You are a helpful assistant. " * 20},
        {"role": "user", "content": "Tell me a long story about a fox. " * 40},
        {"role": "assistant", "content": "Once upon a time."},
    ]
    first  = head + [{"role": "user", "content": "And then what happened?"}]
    second = head + [{"role": "user", "content": "Was the fox hungry?"}]
    other  = [{"role": "user", "content": "Completely unrelated question about the weather."}]

    body = chat(first)
    n_first = body["usage"]["prompt_tokens"]

    chat(other)
    log.drain()

    # the branch restores a copy of the first conversation and rolls back to a checkpoint in the shared part
    body = chat(second)
    out = log.wait_for("__TEST_TAG_PROMPT_CACHE_KEEP__")
    assert "__TEST_TAG_PROMPT_CACHE_KEEP__" in out
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] > 0

    chat(other)
    log.drain()

    # the first conversation is still cached
    body = chat(first)
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_RESTORE__")
    assert 0 < body["usage"]["prompt_tokens_details"]["cached_tokens"] < n_first


def chat(messages):
    res = server.make_request("POST", "/chat/completions", data={
        "messages": messages,
        "max_tokens": 4,
    })
    assert res.status_code == 200
    return res.body


CHAT_HEAD = [
    {"role": "system", "content": "You are a helpful assistant who tells long stories."},
    {"role": "user", "content": LONG_PROMPT},
]
CHAT_OTHER = [{"role": "user", "content": "The quick brown fox jumps over the lazy dog."}]


# chat: the protected prefix of a conversation ends with its last message that is not the assistant's, so a request
# that sends the assistant turn back rendered differently still continues it: the entry moves into the slot
def test_chat_reply_rendered_differently_continues():
    global server
    server.jinja = True
    server.start()
    log = LogReader(server.log_path)

    chat(CHAT_HEAD)
    # the slots' status is in the detailed status of a test build (-DLLAMA_SERVER_TEST_HOOKS=ON)
    status = server.make_request("GET", "/cache").body
    n_protected = 0
    if "test" in status:
        slot = status["test"]["slots"][0]
        assert 0 < slot["n_protected"] < slot["n_tokens"]
        n_protected = slot["n_protected"]

    chat(CHAT_OTHER)
    log.drain()

    # the client sends back another text for the assistant turn than the one generated
    body = chat(CHAT_HEAD + [
        {"role": "assistant", "content": "A reply the model did not write."},
        {"role": "user", "content": "And then?"},
    ])
    out = log.wait_for("__TEST_TAG_PROMPT_CACHE_RESTORE__")
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in out
    assert "__TEST_TAG_PROMPT_CACHE_KEEP__" not in out
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] >= max(n_protected - 1, 1)

    # the entry moved into the slot: the cache holds only the other conversation
    assert server.make_request("GET", "/cache").body["memory"]["entries"] == 1


# chat: a request that edits the last user message does not continue the conversation: it branches off it, and the
# conversation stays in the cache
def test_chat_edited_user_message_branches():
    global server
    server.jinja = True
    server.start()
    log = LogReader(server.log_path)

    chat(CHAT_HEAD)
    chat(CHAT_OTHER)
    log.drain()

    edited = [CHAT_HEAD[0], {"role": "user", "content": LONG_PROMPT + " Make it short."}]
    chat(edited)
    out = log.wait_for("__TEST_TAG_PROMPT_CACHE_KEEP__")
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in out
    assert "__TEST_TAG_PROMPT_CACHE_KEEP__" in out
