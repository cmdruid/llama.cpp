import glob
import os
import tempfile
import threading
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
    # write-through unless a test checks the write delay
    server.cache_dir_delay = 0
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
@pytest.mark.parametrize("change", ["kv_type", "model", "override_kv"])
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
    elif change == "override_kv":
        server.override_kv = "test.cache.identity=int:1"
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
    wait_entries(1)

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
    wait_disk("writes", 1)

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


def cache_status():
    res = server.make_request("GET", "/cache")
    assert res.status_code == 200
    return res.body


class Background:
    """a request in a thread of its own"""
    def __init__(self, fn, *args, **kwargs):
        self.result = None
        self.error = None
        def run():
            try:
                self.result = fn(*args, **kwargs)
            except Exception as e:
                self.error = e
        self.thread = threading.Thread(target=run)
        self.thread.start()
    def join(self, timeout=30):
        self.thread.join(timeout)
        assert not self.thread.is_alive()
        if self.error:
            raise self.error
        return self.result


READ_DELAY_ENV = "LLAMA_SERVER_TEST_CACHE_DISK_READ_DELAY_MS"


# an entry superseded while the reader loads it: the waiting request gets the loaded state, the entry is removed once
# the load ends, and a later eviction does not trip over it (it used to crash the server)
def test_superseded_while_loading(monkeypatch):
    global server
    server.start()
    complete(LONG_PROMPT)
    restart() # SIGTERM persists the slot
    assert len(entry_files()) == 1
    path_long = entry_files()[0]
    server.stop()

    monkeypatch.setenv(READ_DELAY_ENV, "3000")
    server.cache_dir_max = 1 # MiB, about two of the conversations below
    server.start()
    log = LogReader(server.log_path)
    n_long = n_tokens(LONG_PROMPT)

    # A waits for the disk read
    a = Background(complete, LONG_PROMPT + " A")
    assert "reading disk cache entry" in log.wait_for("reading disk cache entry")

    # B takes the slot meanwhile and is persisted: its entry supersedes the one being read
    complete(LONG_PROMPT + " B", cache_prompt=False)
    res = server.make_request("POST", "/cache/persist")
    assert res.status_code == 200 and res.body["flushed"]

    # removing the entry waits for the end of the load
    assert path_long in entry_files()
    assert cache_status()["disk"]["loading"] == 1

    body = a.join()
    assert body["timings"]["cache_n"] >= n_long
    assert path_long not in entry_files()

    # the superseded entry is gone from the index and from the directory
    st = cache_status()["disk"]
    assert st["removed_superseded"] == 1
    assert st["entries"] == len(entry_files())

    # enough new conversations to evict: the server survives and stays within the bound
    for i in range(5):
        complete(conversation(i, 4))
        res = server.make_request("POST", "/cache/persist")
        assert res.status_code == 200
    assert "evicting entry" in log.wait_for("evicting entry")
    st = cache_status()["disk"]
    assert st["entries"] == len(entry_files())
    assert st["bytes"] <= st["limit_bytes"]
    assert server.process.poll() is None


def wait_disk(key, value, timeout=10.0):
    t0 = time.time()
    while True:
        st = cache_status()["disk"]
        if st[key] >= value:
            return st
        assert time.time() - t0 < timeout, f"expected disk {key} >= {value}, got {st}"
        time.sleep(0.05)


# write-behind: a conversation resumed before its delay is not written, the next one waits in RAM
def test_resumed_before_delay_is_not_written():
    global server
    server.cache_dir_delay = 60
    server.start()

    complete(LONG_PROMPT)
    complete(OTHER_PROMPT) # LONG goes to the RAM cache
    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    assert body["timings"]["cache_n"] >= n_tokens(LONG_PROMPT)

    st = cache_status()["disk"]
    assert st["writes"] == 0
    assert st["writes_skipped"] == 1
    assert st["bytes_skipped"] > 0
    assert st["backlog_entries"] == 1 # OTHER waits for its delay
    assert entry_files() == []


# write-behind: an entry is written once it has stayed in RAM for the delay
def test_written_after_delay():
    global server
    server.cache_dir_delay = 2
    server.start()

    complete(LONG_PROMPT)
    complete(OTHER_PROMPT)
    t0 = time.time()
    assert cache_status()["disk"]["backlog_entries"] == 1
    assert entry_files() == []

    wait_disk("writes", 1)
    assert time.time() - t0 > 1.5
    assert len(entry_files()) == 1
    assert cache_status()["disk"]["backlog_entries"] == 0


# write-behind: an entry that the RAM cache drops to stay within its limit is written at once
def test_ram_eviction_writes():
    global server
    server.cache_dir_delay = 60
    server.cache_ram = 1 # MiB, about two of these conversations
    server.start()

    for i in range(4):
        complete(conversation(i, 4))
    complete(OTHER_PROMPT)

    # the two oldest left RAM, the others wait for their delay
    st = wait_disk("writes", 2)
    assert st["writes"] == 2
    assert st["backlog_entries"] == 2
    assert cache_status()["counters"]["evictions_ram"] == 2

    # they are read back from disk
    log = LogReader(server.log_path)
    log.drain()
    body = complete(conversation(0, 4) + " The end.")
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_DISK_LOAD__")
    assert body["timings"]["cache_n"] > 0


# write-behind: a persist writes everything not written yet, the most recently saved first
def test_persist_writes_everything():
    global server
    server.cache_dir_delay = 60
    server.start()
    log = LogReader(server.log_path)

    for i in range(3):
        complete(conversation(i, 1))
    slot = conversation(9, 2)
    complete(slot)
    assert cache_status()["disk"]["backlog_entries"] == 3
    log.drain()

    res = server.make_request("POST", "/cache/persist")
    assert res.status_code == 200
    assert res.body["flushed"] is True
    assert res.body["n_saved"] == 1
    assert len(entry_files()) == 4
    st = cache_status()["disk"]
    assert st["writes"] == 4 and st["backlog_entries"] == 0

    # the slot was saved last: it is written first
    wrote = [l for l in log.drain().splitlines() if "wrote entry" in l]
    assert len(wrote) == 4
    assert f"{n_tokens(slot) + 4} tokens" in wrote[0] or f"{n_tokens(slot) + 3} tokens" in wrote[0]


# --cache-dir-delay 0 writes an entry as soon as it is saved, a resumed conversation keeps its copy
def test_delay_zero_is_write_through():
    global server
    server.cache_dir_delay = 0
    server.start()

    complete(LONG_PROMPT)
    complete(OTHER_PROMPT)
    wait_disk("writes", 1)
    path = entry_files()[0]

    complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    st = cache_status()["disk"]
    assert st["writes_skipped"] == 0
    assert path in entry_files()


# a conversation read from disk and continued keeps its disk copy until the continuation is written
def test_disk_copy_kept_until_continuation_written():
    global server
    server.cache_dir_delay = 60
    server.start()

    complete(LONG_PROMPT)
    restart() # persists the slot
    path = entry_files()[0]

    complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    complete(OTHER_PROMPT) # the continuation goes to RAM, waiting for its delay
    assert entry_files() == [path]

    res = server.make_request("POST", "/cache/persist")
    assert res.status_code == 200 and res.body["flushed"]
    assert path not in entry_files()
    assert cache_status()["disk"]["removed_superseded"] == 1


def key_dirs():
    return sorted(os.path.dirname(p) for p in glob.glob(os.path.join(server.cache_dir, "*", "IDENTITY")))


def persist():
    res = server.make_request("POST", "/cache/persist")
    assert res.status_code == 200 and res.body["flushed"]
    return res.body


# --cache-dir-max bounds the directory as a whole: the entries of another key are counted and evicted, least recently
# used first, and the key directory they leave empty is removed
def test_bound_counts_other_keys():
    global server
    server.start()
    for i in range(2):
        complete(conversation(i, 4))
        persist()
    server.stop()
    assert len(entry_files()) == 2
    [dir_old] = key_dirs()

    # another key: f32 K and V, about twice the size per entry
    server.ctk = "f32"
    server.ctv = "f32"
    server.cache_dir_max = 2
    server.start()

    st = cache_status()["disk"]
    assert st["entries"] == 0
    assert st["other_entries"] == 2
    assert st["bytes"] == sum(os.path.getsize(p) for p in entry_files())

    complete(conversation(2, 4))
    persist()
    assert cache_status()["disk"]["other_entries"] == 2

    # no room for a second entry of this key without the entries of the other key
    complete(conversation(3, 4))
    persist()
    st = cache_status()["disk"]
    assert st["other_entries"] == 0
    assert st["entries"] == 2
    assert st["removed_evicted"] == 2
    assert st["bytes"] <= st["limit_bytes"]
    assert dir_old not in key_dirs()
    assert not os.path.exists(dir_old)


# a key directory without entries is removed at start-up, a directory that is not a key directory is left alone
def test_empty_key_dir_removed():
    global server
    stale = os.path.join(server.cache_dir, "llama-0123456789abcdef")
    other = os.path.join(server.cache_dir, "notes")
    os.makedirs(stale)
    os.makedirs(other)
    with open(os.path.join(stale, "IDENTITY"), "w") as f:
        f.write("stale\n")
    with open(os.path.join(other, "README"), "w") as f:
        f.write("keep\n")

    server.start()
    assert not os.path.exists(stale)
    assert os.path.exists(os.path.join(other, "README"))


# the entry that a growing conversation replaces makes room first: an unrelated entry is not evicted for it
def test_superseded_entry_makes_room():
    global server
    server.cache_dir_max = 1 # MiB
    server.start()

    unrelated = conversation(1, 4)
    grown = conversation(0, 2)
    complete(unrelated)
    persist()
    complete(grown)
    persist()
    assert len(entry_files()) == 2
    path_unrelated = min(entry_files(), key=os.path.getmtime)

    # the conversation grows: the new entry does not fit beside both, but it replaces the old one
    complete(grown + " " + LONG_PROMPT + " " + LONG_PROMPT)
    persist()
    st = cache_status()["disk"]
    assert st["removed_superseded"] == 1
    assert st["removed_evicted"] == 0
    assert path_unrelated in entry_files()
    assert st["bytes"] <= st["limit_bytes"]


# when the live entries leave no room, a write is skipped and counted, the entry stays in RAM, the bound holds
def test_no_room_skips_write():
    global server
    server.n_slots = 2
    server.n_ctx = 4096
    server.cache_dir_max = 1 # MiB
    server.start()

    live = conversation(0, 5)
    complete(live, id_slot=0)
    persist()
    assert len(entry_files()) == 1

    other = conversation(1, 5)
    complete(other, id_slot=1)
    complete(OTHER_PROMPT, id_slot=1) # `other` goes to RAM and is written at once (delay 0)

    st = wait_disk("writes_no_room", 1)
    assert st["writes"] == 1
    assert st["bytes"] <= st["limit_bytes"]
    assert len(entry_files()) == 1

    # still in RAM: restored without a disk read
    log = LogReader(server.log_path)
    log.drain()
    body = complete(other + " The end.", id_slot=1)
    out = log.wait_for("__TEST_TAG_PROMPT_CACHE_RESTORE__")
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in out
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" not in out
    assert body["timings"]["cache_n"] >= n_tokens(other) - 1


# an entry whose file is gone: the request waiting for it proceeds cold, the entry is dropped from the index and later
# requests do not wait for it again
def test_missing_file_is_dropped():
    global server
    server.start()
    complete(LONG_PROMPT)
    restart()
    [path] = entry_files()
    os.remove(path)
    log = LogReader(server.log_path)

    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    out = log.wait_for("dropping it from the index")
    assert "reading disk cache entry" in out
    assert "failed to read entry" in out
    assert body["timings"]["cache_n"] == 0
    st = cache_status()["disk"]
    assert st["entries"] == 0
    assert st["removed_missing"] == 1

    log.drain()
    complete(LONG_PROMPT + " The knight went home.")
    assert "reading disk cache entry" not in log.drain()


# the cache directory removed while the server runs: it is created again (0700) by the next write, and the entries
# that went with it are dropped
def test_removed_directory_is_recreated():
    global server
    import shutil, stat
    server.start()
    complete(LONG_PROMPT)
    complete(OTHER_PROMPT)
    wait_disk("writes", 1)
    [key_dir] = key_dirs()

    shutil.rmtree(server.cache_dir)

    complete(conversation(1)) # OTHER is written
    st = wait_disk("writes", 2)
    assert st["write_errors"] == 0
    assert st["entries"] == 1
    assert st["removed_missing"] == 1
    assert len(entry_files()) == 1
    assert os.path.exists(os.path.join(key_dir, "IDENTITY"))
    assert stat.S_IMODE(os.stat(key_dir).st_mode) == 0o700

    # the lost entry is not waited for (the conversation is still in RAM)
    log = LogReader(server.log_path)
    log.drain()
    complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    assert "reading disk cache entry" not in log.drain()


# a damaged entry is removed when it is read, and the same conversation is written again afterwards: the removal of
# the old file and the write of the new one happen in order
def test_corrupt_entry_rewritten():
    global server
    server.n_predict = 0
    server.start()
    complete(LONG_PROMPT)
    server.stop()

    [path] = entry_files()
    size = os.path.getsize(path)
    with open(path, "r+b") as f:
        f.seek(size - 64)
        b = f.read(1)
        f.seek(size - 64)
        f.write(bytes([b[0] ^ 0xff]))

    server.start()
    log = LogReader(server.log_path)
    complete(LONG_PROMPT)
    assert "is corrupt, removing it" in log.wait_for("is corrupt, removing it")
    persist()
    assert entry_files() == [path]
    assert cache_status()["disk"]["entries"] == 1

    restart()
    log = LogReader(server.log_path)
    complete(LONG_PROMPT)
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_DISK_LOAD__")


# the key names the build and the resolved flash attention, not the requested one
def test_identity_names_build_and_flash_attn():
    global server
    server.start()
    [key_dir] = key_dirs()
    identity = open(os.path.join(key_dir, "IDENTITY")).read()
    assert "\nbuild: " in identity
    assert "flash_attn on" in identity or "flash_attn off" in identity
    assert "flash_attn -1" not in identity


RESTORE_FAIL_ENV = "LLAMA_SERVER_TEST_CACHE_RESTORE_FAIL"


# a restore that fails does not prove the entry bad: the request proceeds cold, the disk copy is kept and is not
# offered again by this server; a later run restores it
def test_restore_failure_keeps_disk_entry(monkeypatch):
    global server
    server.start()
    complete(LONG_PROMPT)
    restart()
    [path] = entry_files()
    server.stop()

    monkeypatch.setenv(RESTORE_FAIL_ENV, "1")
    server.start()
    log = LogReader(server.log_path)
    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    assert "failed to restore" in log.wait_for("failed to restore")
    assert body["timings"]["cache_n"] == 0
    assert path in entry_files()
    st = cache_status()
    assert st["disk"]["quarantined"] == 1
    assert st["counters"]["failed_restores"] == 1

    # not read again
    log.drain()
    complete(LONG_PROMPT + " The knight went home.")
    assert "reading disk cache entry" not in log.drain()
    server.stop()

    monkeypatch.delenv(RESTORE_FAIL_ENV)
    server.start()
    log = LogReader(server.log_path)
    body = complete(LONG_PROMPT + " The knight finally reached the castle gates.")
    assert "__TEST_TAG_PROMPT_CACHE_RESTORE__" in log.wait_for("__TEST_TAG_PROMPT_CACHE_RESTORE__")
    assert body["timings"]["cache_n"] >= n_tokens(LONG_PROMPT)


# temporary files at start-up: one of a process that is gone, or older than an hour (its pid may be reused), is
# removed; one of a process that runs is kept and counted in the bound
def test_stale_temporary_files():
    global server
    import subprocess
    server.start()
    server.stop()
    [key_dir] = key_dirs()

    p = subprocess.Popen(["true"])
    p.wait()
    gone = os.path.join(key_dir, f".0000000000000001.{p.pid}.tmp")
    live = os.path.join(key_dir, f".0000000000000002.{os.getpid()}.tmp")
    old  = os.path.join(key_dir, f".0000000000000003.{os.getpid()}.tmp")
    for path in (gone, live, old):
        with open(path, "wb") as f:
            f.write(b"x" * 1000)
    t = time.time() - 2*3600
    os.utime(old, (t, t))

    server.start()
    assert not os.path.exists(gone)
    assert not os.path.exists(old)
    assert os.path.exists(live)
    st = cache_status()["disk"]
    assert st["bytes"] == 1000
    assert st["other_entries"] == 1


WRITE_DELAY_ENV = "LLAMA_SERVER_TEST_CACHE_DISK_WRITE_DELAY_MS"


# the entries that the RAM cache drops are held by the writer until they are written: at most one at a time, the
# server loop waits for the writer before it hands it another
def test_backlog_bounded_beyond_ram(monkeypatch):
    global server
    monkeypatch.setenv(WRITE_DELAY_ENV, "1500")
    server.cache_dir_delay = 60
    server.cache_ram = 1 # MiB, about two of these conversations
    server.start()

    seen = []
    stop = threading.Event()
    def watch():
        while not stop.is_set():
            seen.append(cache_status()["disk"]["released_entries"])
            time.sleep(0.05)
    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        for i in range(4):
            complete(conversation(i, 4))
        complete(OTHER_PROMPT)
    finally:
        stop.set()
        watcher.join()

    st = wait_disk("writes", 2)
    assert st["backlog_waits"] >= 1
    assert st["backlog_wait_ms"] > 500
    assert st["writes_dropped"] == 0
    assert max(seen) <= 1
    assert st["released_entries"] == 0


# a task waiting for a disk read is cancelled (the client went away): its row and the read result are released
def test_cancelled_waiting_task_releases_read(monkeypatch):
    global server
    import requests
    server.start()
    complete(LONG_PROMPT)
    restart()
    server.stop()

    monkeypatch.setenv(READ_DELAY_ENV, "2000")
    server.start()
    log = LogReader(server.log_path)
    try:
        requests.post(f"http://{server.server_host}:{server.server_port}/completion", json={
            "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
            "cache_prompt": True,
        }, timeout=0.5)
        assert False, "the request should have timed out"
    except requests.exceptions.Timeout:
        pass
    assert "reading disk cache entry" in log.wait_for("reading disk cache entry")

    t0 = time.time()
    while True:
        st = cache_status()
        if st["disk"]["loading"] == 0 and st["waiting"] == 0:
            break
        assert time.time() - t0 < 10, st
        time.sleep(0.1)
    assert st["disk"]["loaded"] == 0
    assert "__TEST_TAG_PROMPT_CACHE_DISK_LOAD__" not in log.drain()


# going to sleep does not wait for the disk writer with the task queue locked: HTTP requests are answered meanwhile
def test_sleep_does_not_wait_for_writer(monkeypatch):
    global server
    monkeypatch.setenv(WRITE_DELAY_ENV, "3000")
    server.sleep_idle_seconds = 1
    server.start()
    log = LogReader(server.log_path)

    complete(LONG_PROMPT)
    assert "persist before sleep" in log.wait_for("persist before sleep", timeout=10)

    t0 = time.time()
    st = cache_status()
    assert time.time() - t0 < 1.0
    assert st["sleeping"] is True
    assert st["disk"]["backlog_entries"] == 1

    st = wait_disk("writes", 1)
    assert len(entry_files()) == 1
