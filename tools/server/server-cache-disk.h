#pragma once

#include "server-task.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

// disk tier of the prompt cache (--cache-dir)
//
// every prompt cache entry is also written to <dir>/<key>/<id>.bin by a background writer thread, and entries found
// there at start-up are indexed (tokens, checkpoint positions and sizes, not the states), so that a conversation
// survives a restart and conversations beyond --cache-ram stay resumable
//
// - <key> names everything that shapes a saved state (model and draft files, context and KV settings), so a state is
//   never loaded into an engine that would not have computed it, see server_context_impl::cache_disk_identity()
// - <id> is a hash of the entry's tokens: an entry present both in RAM and on disk is one entry
// - files are written to a temporary name, synced and renamed; a torn, truncated or corrupt file is detected by its
//   exact length and checksums, removed and never loaded
// - the directory is bounded by size, least recently used first (file mtime), the entries of live slots are kept
// - an entry superseded or removed while it is being read is removed when the read ends: the read completes and the
//   request waiting for it gets the state
//
// threads: the server loop calls every public method; the writer and the reader threads do the file I/O
// all mutable fields of `state` are guarded by state::mtx, which is never held during file I/O
// the writer reads the entry data through shared pointers to immutable buffers (server_prompt_data and the
// checkpoints are not modified once they are in the prompt cache)

uint64_t server_cache_disk_hash(const void * data, size_t size, uint64_t seed = 0);

uint64_t server_cache_disk_hash_tokens(const llama_tokens & tokens);

// the tokens of a prompt; false if it has media chunks, such a prompt is never written to disk
bool server_cache_disk_text_tokens(const server_tokens & tokens, llama_tokens & out);

// cheap identity of a model file: its size and hashes of its first and last MiB
std::string server_cache_disk_file_identity(const std::string & path);

struct server_cache_disk_stats {
    size_t n_entries = 0;
    size_t n_bytes   = 0;
    size_t n_bytes_max = 0;

    size_t n_backlog       = 0; // writes queued or in progress
    size_t n_backlog_bytes = 0;
    size_t n_loading       = 0; // reads in progress

    uint64_t n_writes        = 0;
    uint64_t n_write_skipped = 0; // already on disk, or larger than the bound
    uint64_t n_write_errors  = 0;
    uint64_t n_reads         = 0;
    uint64_t n_read_errors   = 0;
    uint64_t n_removed_corrupt    = 0;
    uint64_t n_removed_evicted    = 0;
    uint64_t n_removed_superseded = 0;

    uint64_t n_bytes_written = 0;
    uint64_t n_bytes_read    = 0;
    double   t_write_ms = 0.0; // total
    double   t_read_ms  = 0.0; // total

    std::string last_error;
};

// an indexed entry, immutable once published
// `prompt` holds the tokens and the checkpoint positions, the checkpoints carry no data
struct server_cache_disk_entry {
    uint64_t      id      = 0;
    server_prompt prompt;
    llama_pos     pos_min = 0;
    size_t        n_bytes = 0;
};

struct server_cache_disk {
    using entry     = server_cache_disk_entry;
    using entry_ptr = std::shared_ptr<const entry>;

    enum load_result {
        LOAD_PENDING,
        LOAD_FAILED,
        LOAD_OK,
    };

    // returns nullptr if the directory cannot be used
    // on_loaded(id) is called from the reader thread when a load started by load_async() is done
    static std::unique_ptr<server_cache_disk> open(
            const std::string & dir_root,
            const std::string & key_name,
            const std::string & key,
            size_t n_bytes_max,
            std::function<void(uint64_t)> on_loaded);

    ~server_cache_disk();

    const std::string & dir() const { return path_dir; }

    // queue an entry of the prompt cache for writing, unless it is already on disk
    // returns the ticket of the write, see wait()
    uint64_t write(const server_prompt_cache_state & state);

    // the index, for matching requests
    std::vector<entry_ptr> entries() const;

    // read an entry in the background, then call on_loaded(id); false if the entry is unknown
    // true if the read is started or already in progress
    bool load_async(uint64_t id);

    // take the state read by load_async(); LOAD_FAILED if the read failed (a corrupt file is removed)
    load_result take_loaded(uint64_t id, server_prompt_cache_state & out, bool has_mtmd);

    // mark an entry as used now (LRU)
    void touch(uint64_t id);

    // remove an entry that cannot be restored
    void remove(uint64_t id);

    // the conversations of the slots: the entries they continue are not evicted
    void set_live(std::vector<llama_tokens> conversations);

    // the ticket of the last write queued so far
    uint64_t ticket() const;

    // wait until the writes up to `ticket` are done; false on timeout
    bool wait(uint64_t ticket, int64_t timeout_ms) const;

    // stop the threads: finish the queued writes within timeout_ms, then abandon the rest
    // never blocks much longer than timeout_ms, even if a write is stuck in the kernel
    void stop(int64_t timeout_ms);

    server_cache_disk_stats stats() const;

    struct state;

private:
    server_cache_disk() = default;

    std::string path_dir;

    std::shared_ptr<state> st; // shared with the threads, which may outlive this object if they are stuck
};
