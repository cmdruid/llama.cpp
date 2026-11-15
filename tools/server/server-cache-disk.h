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
// prompt cache entries are written to <dir>/<key>/<id>.bin by a background writer thread, and entries found there at
// start-up are indexed (tokens, checkpoint positions and sizes, not the states), so that a conversation survives a
// restart and conversations beyond --cache-ram stay resumable
//
// write-behind: --cache-dir-delay is the maximum age of a conversation's unwritten progress: an entry saved to the RAM
// cache is written once its oldest progress not on disk is --cache-dir-delay seconds old, when the RAM cache drops it
// to stay within its limits (the writer then holds its memory until it is written), when it is one of the next
// entries the RAM cache will drop (write-ahead, see write_ahead()), or when a flush asks for everything (persist,
// shutdown, sleep); an entry that leaves the RAM cache before then because a slot continues it is not written, the
// slot inherits the age of its progress (see drop()) and its next save supersedes it
//
// - <key> names everything that shapes a saved state (model and draft files, context and KV settings), so a state is
//   never loaded into an engine that would not have computed it, see server_context_impl::cache_disk_identity()
// - <id> is a hash of the entry's tokens: an entry present both in RAM and on disk is one entry
// - files are written to a temporary name, synced and renamed; a torn, truncated or corrupt file is detected by its
//   exact length and checksums, removed and never loaded
// - the directory root is bounded by size as a whole: the entries of other keys found at start-up are counted and
//   evicted with the others, least recently used first (file mtime); the entries of live slots are kept, and a write
//   that would exceed the bound because of them is skipped (the entry stays in RAM)
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
    size_t n_entries = 0;   // of this key
    size_t n_bytes   = 0;   // all files under the directory root, counted in the bound
    size_t n_bytes_max = 0;

    size_t n_entries_other = 0; // the entries of other keys under the directory root
    size_t n_bytes_other   = 0;

    size_t n_backlog       = 0; // entries not written yet: waiting for their delay, queued or in progress
    size_t n_backlog_bytes = 0;
    size_t n_loading       = 0; // reads in progress
    size_t n_loaded        = 0; // reads done, not taken yet

    size_t n_released       = 0; // entries of the backlog that the RAM cache dropped: memory beyond --cache-ram
    size_t n_released_bytes = 0;
    uint64_t n_backlog_waits   = 0; // times the server loop waited for the writer, see release()
    int64_t  t_backlog_wait_ms = 0;
    uint64_t n_backlog_timeouts = 0; // waits that ended after t_wait_max_ms: the writer is considered stalled

    int64_t t_delay_ms = 0;

    uint64_t n_writes        = 0;
    uint64_t n_write_skipped = 0; // left the RAM cache (a slot continued them) before they were written
    uint64_t n_write_present = 0; // already on disk
    uint64_t n_write_no_room = 0; // larger than the bound
    uint64_t n_write_errors  = 0;
    uint64_t n_write_dropped = 0; // dropped by the RAM cache while the writer did not keep up (all causes)
    uint64_t n_write_dropped_stalled = 0; // of which without waiting: the writer made no progress since a timeout
    uint64_t n_write_dropped_nowait  = 0; // of which without waiting: the server loop may not wait (going to sleep)
    uint64_t n_write_ahead   = 0; // writes started before the delay because the RAM cache was about to drop the entry
    uint64_t n_touches       = 0; // file mtime updates (LRU order across restarts), coalesced per entry
    uint64_t n_reads         = 0;
    uint64_t n_read_errors   = 0;
    uint64_t n_removed_corrupt    = 0;
    uint64_t n_removed_evicted    = 0;
    uint64_t n_removed_superseded = 0;
    uint64_t n_removed_missing    = 0; // indexed, but the file is gone
    uint64_t n_quarantined        = 0; // entries that failed to restore, not offered again until a restart

    uint64_t n_bytes_written = 0;
    uint64_t n_bytes_skipped = 0; // the size of the writes skipped because the entry left the RAM cache
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
            int64_t t_delay_ms,
            int64_t t_wait_max_ms,
            std::function<void(uint64_t)> on_loaded);

    ~server_cache_disk();

    const std::string & dir() const { return path_dir; }

    // an entry was saved to the RAM cache: write it once its oldest unwritten progress is t_delay_ms old
    // (state.prompt.t_unwritten_ms, now if 0), unless it is already on disk (then its file is touched)
    void stage(const server_prompt_cache_state & state);

    // the entry left the RAM cache because a slot continues it (or a newer entry contains it): do not write it
    // returns the time of its oldest unwritten progress (ggml_time_ms()), for the continuation to inherit, 0 if the
    // entry is not waiting to be written
    int64_t drop(uint64_t id);

    // the RAM cache is about to drop these entries (its least recently used ones, above its high-water mark): write
    // them now, regardless of their delay, so that release() finds them on disk
    void write_ahead(const std::vector<uint64_t> & ids);

    // false: release() never waits for the writer, it drops the entry instead (the server loop holds a lock that
    // must not wait for the disk, e.g. going to sleep)
    void set_wait_allowed(bool allowed);

    // the RAM cache drops the entry to stay within its limits: write it now, unless it is already on disk
    // the writer then holds its memory: if it holds another such entry already, this waits until that one is written
    // (at most t_wait_max_ms, then the entry is dropped), so that the memory beyond --cache-ram stays within one entry;
    // after such a timeout, entries are dropped without waiting until the writer finishes a write (a hung disk freezes
    // the server loop once), and never waits while set_wait_allowed(false)
    void release(const server_prompt_cache_state & state);

    // write every entry not written yet, the most recently saved first; returns the ticket for wait()
    uint64_t flush();

    // the index, for matching requests
    std::vector<entry_ptr> entries() const;

    // read an entry in the background, then call on_loaded(id); false if the entry is unknown
    // true if the read is started or already in progress
    bool load_async(uint64_t id);

    // nobody waits for the read of this entry any more: release its result now or when the read ends
    void discard(uint64_t id);

    // take the state read by load_async(); LOAD_FAILED if the read failed (a corrupt file is removed)
    load_result take_loaded(uint64_t id, server_prompt_cache_state & out, bool has_mtmd);

    // mark an entry as used now (LRU): its file's mtime, or the mtime it will be written with
    void touch(uint64_t id);

    // a restore of the entry failed: it is not offered again by this server, its file is kept
    // (a failed restore does not prove that the entry is bad, e.g. it may not fit next to the other slots)
    void quarantine(uint64_t id);

    // the conversations of the slots: the entries they continue are not evicted
    void set_live(std::vector<llama_tokens> conversations);

    // wait until the entries flushed by `ticket` are written (or dropped); false on timeout
    bool wait(uint64_t ticket, int64_t timeout_ms) const;

    // stop the threads: write the entries not written yet within timeout_ms, then abandon the rest
    // never blocks much longer than timeout_ms, even if a write is stuck in the kernel
    void stop(int64_t timeout_ms);

    server_cache_disk_stats stats() const;

    struct state;

private:
    server_cache_disk() = default;

    void stage_locked(const server_prompt_cache_state & state, bool released);

    std::string path_dir;

    std::shared_ptr<state> st; // shared with the threads, which may outlive this object if they are stuck
};
