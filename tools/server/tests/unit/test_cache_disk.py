import glob
import os
import tempfile
import time
import pytest
from utils import *

# disk tier of the prompt cache (--cache-dir): entries are written to disk, survive a restart, are bounded by
# --cache-dir-max and are only loaded by an engine with the same identity

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
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 2048
    server.n_slots = 1
    server.n_predict = 4
    server.temperature = 0.0
    server.cache_ram = 100
    server.cache_dir = str(tmp_path / "cache")
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

OTHER_PROMPT = "The quick brown fox jumps over the lazy dog."


def conversation(i, n_repeat=1):
    return f"Story number {i}. " + " ".join([LONG_PROMPT] * n_repeat)


def complete(prompt, **kwargs):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "cache_prompt": True,
        **kwargs,
    })
    assert res.status_code == 200
    return res.body


def n_tokens(prompt):
    res = server.make_request("POST", "/tokenize", data={"content": prompt, "add_special": True})
    assert res.status_code == 200
    return len(res.body["tokens"])


def entry_files():
    return sorted(glob.glob(os.path.join(server.cache_dir, "*", "*.bin")))


def wait_entries(n, timeout=10.0):
    t0 = time.time()
    while len(entry_files()) < n:
        assert time.time() - t0 < timeout, f"expected {n} entries on disk, found {entry_files()}"
        time.sleep(0.05)


def restart():
    server.stop()
    # keep the log of the previous run
    os.replace(server.log_path, server.log_path + ".prev")
    server.start()


# an entry written before a restart is reused after it
def test_entry_survives_restart():
    global server
    server.start()

    n_long = n_tokens(LONG_PROMPT)
    complete(LONG_PROMPT)

    # displacing the conversation from the only slot writes it to disk
    complete(OTHER_PROMPT)
    wait_entries(1)

    restart()
    log = LogReader(server.log_path)

    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    out = log.wait_for("__TEST_TAG_PROMPT_CACHE_RESTORE__")
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" in out
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in out
    assert body["timings"]["cache_n"] >= n_long


# conversations beyond the RAM limit are resumable from disk
def test_conversations_beyond_ram_limit():
    global server
    server.cache_ram = 1 # MiB, about two of these conversations
    server.start()
    log = LogReader(server.log_path)

    prompts = [conversation(i, 4) for i in range(4)]
    n = [n_tokens(p) for p in prompts]

    for p in prompts:
        complete(p)
    complete(OTHER_PROMPT)
    wait_entries(4)
    log.drain()

    # the oldest ones were dropped from RAM, they are read from disk
    for i in range(2):
        body = complete(prompts[i] + " The end.")
        assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_DISK_LOAD__")
        assert body["timings"]["cache_n"] >= n[i] - 1


# the bound evicts the least recently used entries and keeps the one a slot continues
def test_bound_evicts_oldest_keeps_live():
    global server
    server.n_slots = 2
    server.n_ctx = 4096
    server.cache_dir_max = 1 # MiB, about two of these conversations
    server.start()
    log = LogReader(server.log_path)

    live = conversation(0, 4)
    others = [conversation(i, 4) for i in range(1, 5)]

    # the live conversation stays in slot 0; launching tasks on slot 1 saves the idle slot 0 to the cache
    complete(live, id_slot=0)
    for p in others:
        complete(p, id_slot=1)
    complete(OTHER_PROMPT, id_slot=1)
    wait_entries(2)
    time.sleep(0.5)

    out = log.wait_for("evicting entry")
    assert "evicting entry" in out
    assert len(entry_files()) <= 2

    restart()
    log = LogReader(server.log_path)

    # the live conversation was the oldest entry, it is still there
    body = complete(live + " The end.")
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_DISK_LOAD__")
    assert body["timings"]["cache_n"] > n_tokens(live) // 2

    # the oldest of the others is gone
    body = complete(others[0] + " The end.")
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" not in log.drain()


@pytest.mark.parametrize("damage", ["truncate", "flip"])
def test_damaged_entry_is_removed(damage):
    global server
    server.start()

    complete(LONG_PROMPT)
    complete(OTHER_PROMPT)
    wait_entries(1)
    server.stop()

    # the shutdown persisted the slot as well: the long conversation is the largest entry
    path = max(entry_files(), key=os.path.getsize)
    size = os.path.getsize(path)
    if damage == "truncate":
        with open(path, "r+b") as f:
            f.truncate(size // 2)
    else:
        # a byte of the state, after the header and the tokens
        with open(path, "r+b") as f:
            f.seek(size - 64)
            b = f.read(1)
            f.seek(size - 64)
            f.write(bytes([b[0] ^ 0xff]))

    server.start()
    log = LogReader(server.log_path)

    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    expected = "removing invalid or truncated entry" if damage == "truncate" else "is corrupt, removing it"
    out = log.wait_for(expected)
    assert expected in out
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" not in out
    assert body["timings"]["cache_n"] == 0
    assert path not in entry_files()


# an entry is only loaded by an engine with the same model and KV configuration
@pytest.mark.parametrize("change", ["kv_type", "model"])
def test_identity_key(change):
    global server
    server.start()

    # the other model has a context of 128 tokens
    prompt = LONG_PROMPT[:100]

    complete(prompt)
    complete(OTHER_PROMPT)
    wait_entries(1)
    server.stop()

    if change == "kv_type":
        server.ctk = "f32"
    else:
        # the same architecture and shapes, another file
        server.model_hf_repo = "ggml-org-dev/test-model-stories260K-infill"
    server.start()
    log = LogReader(server.log_path)

    body = complete(prompt + " The knight finally reached the castle gates.")
    out = log.drain()
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" not in out
    assert body["timings"]["cache_n"] == 0

    # two configurations, two subdirectories
    assert len(glob.glob(os.path.join(server.cache_dir, "*", "IDENTITY"))) == 2


# with a unified KV cache and --cache-idle-slots, the idle slot cleared on a new task reaches the disk
def test_kv_unified_idle_slot_reaches_disk():
    global server
    server.n_slots = 2
    server.kv_unified = True
    server.start()
    log = LogReader(server.log_path)

    n_long = n_tokens(LONG_PROMPT)
    complete(LONG_PROMPT, id_slot=0)
    complete(OTHER_PROMPT, id_slot=1)
    assert "__TEST_TAG_CACHE_IDLE_SLOT__" in log.wait_for("__TEST_TAG_CACHE_IDLE_SLOT__")
    wait_entries(1)

    restart()
    log = LogReader(server.log_path)

    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_DISK_LOAD__")
    assert body["timings"]["cache_n"] >= n_long


def kill():
    # no graceful shutdown: only what is already on disk survives
    server.process.kill()
    server.process.wait()
    server.process = None
    server_instances.discard(server)
    os.replace(server.log_path, server.log_path + ".prev")


# POST /cache/persist writes the live slots and waits for the writer: the conversation survives a crash
def test_persist_endpoint():
    global server
    server.start()

    n_long = n_tokens(LONG_PROMPT)
    complete(LONG_PROMPT)
    assert entry_files() == []

    res = server.make_request("POST", "/cache/persist")
    assert res.status_code == 200
    assert res.body["n_saved"] == 1
    assert res.body["n_busy"] == 0
    assert res.body["disk"] is True
    assert res.body["flushed"] is True
    assert len(entry_files()) == 1

    # nothing new to save
    res = server.make_request("POST", "/cache/persist")
    assert res.status_code == 200
    assert res.body["n_saved"] == 0
    assert res.body["n_cached"] == 1
    assert len(entry_files()) == 1

    res = server.make_request("GET", "/cache")
    assert res.status_code == 200
    assert res.body["disk"]["entries"] == 1
    assert res.body["disk"]["backlog_entries"] == 0
    assert res.body["counters"]["persists"] == 2
    assert res.body["slots"][0]["n_tokens"] > 0

    kill()
    server.start()
    log = LogReader(server.log_path)

    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_DISK_LOAD__")
    assert body["timings"]["cache_n"] >= n_long

    res = server.make_request("GET", "/cache")
    assert res.body["counters"]["hits_disk"] == 1
    assert res.body["counters"]["hits_ram"] == 1


# a graceful shutdown (SIGTERM) persists the live slots
def test_sigterm_persists_live_slot():
    global server
    server.start()

    n_long = n_tokens(LONG_PROMPT)
    complete(LONG_PROMPT)
    assert entry_files() == []

    restart() # SIGTERM
    assert "persist before shutdown: 1 slots saved, written to disk" in open(server.log_path + ".prev", errors="replace").read()
    assert len(entry_files()) == 1

    log = LogReader(server.log_path)
    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_DISK_LOAD__")
    assert body["timings"]["cache_n"] >= n_long


# going to sleep discards the slots and the RAM cache: their conversations are persisted first
def test_sleep_persists():
    global server
    server.sleep_idle_seconds = 1
    server.start()
    log = LogReader(server.log_path)

    n_long = n_tokens(LONG_PROMPT)
    complete(LONG_PROMPT)

    t0 = time.time()
    while "persist before sleep" not in log.drain():
        assert time.time() - t0 < 10
        time.sleep(0.1)
    assert len(entry_files()) == 1

    res = server.make_request("GET", "/cache")
    assert res.status_code == 200
    assert res.body["sleeping"] is True
    assert res.body["disk"]["entries"] == 1

    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_DISK_LOAD__")
    assert body["timings"]["cache_n"] >= n_long


def test_metrics():
    global server
    server.server_metrics = True
    server.start()

    complete(LONG_PROMPT)
    complete(OTHER_PROMPT)
    wait_entries(1)

    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    metrics = {}
    for line in res.body.splitlines():
        if line.startswith("llamacpp:prompt_cache"):
            k, v = line.split(" ")
            metrics[k] = float(v)
    assert metrics["llamacpp:prompt_cache_saves_total"] >= 1
    assert metrics["llamacpp:prompt_cache_disk_entries"] == 1
    assert metrics["llamacpp:prompt_cache_disk_bytes"] > 0
    assert metrics["llamacpp:prompt_cache_disk_writes_total"] == 1


def test_disabled_without_cache_ram():
    global server
    server.cache_ram = 0
    server.start()
    log = LogReader(server.log_path)
    assert "--cache-dir requires --cache-ram" in log.wait_for("--cache-dir requires --cache-ram")

    complete(LONG_PROMPT)
    res = server.make_request("POST", "/cache/persist")
    assert res.status_code == 200
    assert res.body["disk"] is False
    assert entry_files() == []
