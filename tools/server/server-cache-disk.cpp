#include "server-cache-disk.h"

#include "log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <filesystem>

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#define DSK_INF(fmt, ...) LOG_INF("disk %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define DSK_WRN(fmt, ...) LOG_WRN("disk %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define DSK_ERR(fmt, ...) LOG_ERR("disk %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define DSK_DBG(fmt, ...) LOG_DBG("disk %12.*s: " fmt, 12, __func__, __VA_ARGS__)

//
// hashing (XXH64)
//

namespace {

constexpr uint64_t XXH_P1 = 11400714785074694791ULL;
constexpr uint64_t XXH_P2 = 14029467366897019727ULL;
constexpr uint64_t XXH_P3 =  1609587929392839161ULL;
constexpr uint64_t XXH_P4 =  9650029242287828579ULL;
constexpr uint64_t XXH_P5 =  2870177450012600261ULL;

inline uint64_t xxh_rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
inline uint64_t xxh_rd64(const uint8_t * p) { uint64_t v; memcpy(&v, p, 8); return v; }
inline uint32_t xxh_rd32(const uint8_t * p) { uint32_t v; memcpy(&v, p, 4); return v; }

inline uint64_t xxh_round(uint64_t acc, uint64_t in) {
    acc += in * XXH_P2;
    acc  = xxh_rotl(acc, 31);
    return acc * XXH_P1;
}

inline uint64_t xxh_merge(uint64_t acc, uint64_t v) {
    acc ^= xxh_round(0, v);
    return acc * XXH_P1 + XXH_P4;
}

struct xxh64 {
    uint64_t v[4];
    uint64_t seed;
    uint64_t total = 0;
    uint8_t  buf[32];
    size_t   n_buf = 0;

    explicit xxh64(uint64_t seed = 0) : seed(seed) {
        v[0] = seed + XXH_P1 + XXH_P2;
        v[1] = seed + XXH_P2;
        v[2] = seed;
        v[3] = seed - XXH_P1;
    }

    void stripe(const uint8_t * p) {
        v[0] = xxh_round(v[0], xxh_rd64(p +  0));
        v[1] = xxh_round(v[1], xxh_rd64(p +  8));
        v[2] = xxh_round(v[2], xxh_rd64(p + 16));
        v[3] = xxh_round(v[3], xxh_rd64(p + 24));
    }

    void update(const void * data, size_t len) {
        const uint8_t * p = (const uint8_t *) data;
        total += len;
        if (n_buf + len < 32) {
            memcpy(buf + n_buf, p, len);
            n_buf += len;
            return;
        }
        if (n_buf > 0) {
            const size_t k = 32 - n_buf;
            memcpy(buf + n_buf, p, k);
            stripe(buf);
            p   += k;
            len -= k;
            n_buf = 0;
        }
        while (len >= 32) {
            stripe(p);
            p   += 32;
            len -= 32;
        }
        if (len > 0) {
            memcpy(buf, p, len);
            n_buf = len;
        }
    }

    uint64_t digest() const {
        uint64_t h;
        if (total >= 32) {
            h = xxh_rotl(v[0], 1) + xxh_rotl(v[1], 7) + xxh_rotl(v[2], 12) + xxh_rotl(v[3], 18);
            h = xxh_merge(h, v[0]);
            h = xxh_merge(h, v[1]);
            h = xxh_merge(h, v[2]);
            h = xxh_merge(h, v[3]);
        } else {
            h = seed + XXH_P5;
        }
        h += total;

        const uint8_t * p = buf;
        size_t len = n_buf;
        while (len >= 8) {
            h ^= xxh_round(0, xxh_rd64(p));
            h  = xxh_rotl(h, 27) * XXH_P1 + XXH_P4;
            p += 8; len -= 8;
        }
        if (len >= 4) {
            h ^= (uint64_t) xxh_rd32(p) * XXH_P1;
            h  = xxh_rotl(h, 23) * XXH_P2 + XXH_P3;
            p += 4; len -= 4;
        }
        while (len > 0) {
            h ^= (*p) * XXH_P5;
            h  = xxh_rotl(h, 11) * XXH_P1;
            p++; len--;
        }
        h ^= h >> 33; h *= XXH_P2;
        h ^= h >> 29; h *= XXH_P3;
        h ^= h >> 32;
        return h;
    }
};

std::string hex64(uint64_t v) {
    char buf[17];
    snprintf(buf, sizeof(buf), "%016" PRIx64, v);
    return buf;
}

int64_t time_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

} // namespace

uint64_t server_cache_disk_hash(const void * data, size_t size, uint64_t seed) {
    xxh64 h(seed);
    h.update(data, size);
    return h.digest();
}

uint64_t server_cache_disk_hash_tokens(const llama_tokens & tokens) {
    return server_cache_disk_hash(tokens.data(), tokens.size()*sizeof(llama_token), 0x4c535043);
}

bool server_cache_disk_text_tokens(const server_tokens & tokens, llama_tokens & out) {
    out = tokens.get_text_tokens();
    return out.size() == tokens.size();
}

//
// file format, version 1 (little-endian)
//
//   header (96 bytes):
//     u32 magic "LSPC", u32 version, u64 key hash, u64 id (hash of the tokens), u64 n_tokens, i64 n_protected,
//     i64 pos_min, u64 n_checkpoints, u64 size of the main state, u64 size of the draft state, u64 size of the meta
//     section, u64 size of the file, u64 hash of the header (without this field) and of the meta section
//   meta:
//     the tokens (i32 each), then per checkpoint: i64 n_tokens, i32 pos_min, i32 pos_max,
//     u64 size of the target, draft and speculative data
//   payload:
//     the main state, the draft state, then the target, draft and speculative data of each checkpoint
//   trailer (16 bytes):
//     u64 hash of the payload, u32 magic "CPSL", u32 0
//
// the payload is the output of llama_state_seq_get_data_ext() as kept by the RAM cache, so it is only
// valid for the engine that computed it - the key directory guarantees that
//

static constexpr uint32_t DISK_MAGIC     = 0x4350534c; // "LSPC"
static constexpr uint32_t DISK_MAGIC_END = 0x4c535043; // "CPSL"
static constexpr uint32_t DISK_VERSION   = 1;

static constexpr size_t DISK_HEADER_SIZE  = 96;
static constexpr size_t DISK_CKPT_SIZE    = 40;
static constexpr size_t DISK_TRAILER_SIZE = 16;

static constexpr size_t DISK_IO_CHUNK = 64ull*1024*1024;

namespace {

struct byte_writer {
    std::vector<uint8_t> buf;

    template <typename T>
    void put(T v) {
        const size_t n = buf.size();
        buf.resize(n + sizeof(T));
        memcpy(buf.data() + n, &v, sizeof(T));
    }

    void put_raw(const void * p, size_t n) {
        buf.insert(buf.end(), (const uint8_t *) p, (const uint8_t *) p + n);
    }
};

struct byte_reader {
    const uint8_t * p;
    size_t n;
    size_t off = 0;

    template <typename T>
    bool get(T & v) {
        if (off + sizeof(T) > n) {
            return false;
        }
        memcpy(&v, p + off, sizeof(T));
        off += sizeof(T);
        return true;
    }
};

struct file_header {
    uint32_t magic;
    uint32_t version;
    uint64_t key_hash;
    uint64_t id;
    uint64_t n_tokens;
    int64_t  n_protected;
    int64_t  pos_min;
    uint64_t n_ckpt;
    uint64_t size_main;
    uint64_t size_drft;
    uint64_t size_meta;
    uint64_t size_total;
    uint64_t meta_hash;
};

struct ckpt_meta {
    int64_t  n_tokens;
    int32_t  pos_min;
    int32_t  pos_max;
    uint64_t size_tgt;
    uint64_t size_dft;
    uint64_t size_spec;
};

// what the writer needs of a cache entry: copies of the small parts, shared references to the large ones
struct write_job {
    uint64_t id = 0;

    llama_tokens tokens;
    int32_t      n_protected = 0;
    llama_pos    pos_min     = 0;

    std::vector<std::shared_ptr<const common_prompt_checkpoint>> ckpts;
    std::shared_ptr<const server_prompt_data> data;

    size_t n_bytes = 0; // the size of the file

    // the last use of the entry (ns since epoch): its save to the RAM cache, not the time of the write, so that the
    // LRU order does not depend on the order of the writes; the file gets it as its mtime
    int64_t t_used = 0;
};

// a write that waits for its delay, see server_cache_disk::stage()
struct staged_write {
    write_job job;

    int64_t  t_staged_ms = 0;
    int64_t  t_due_ms    = 0;
    uint64_t flush       = 0;     // the flush that wants it written, 0 = none
    bool     released    = false; // the RAM cache dropped the entry: the writer holds its only copy
};

// a small file operation of the writer, done in order before the writes
struct file_op {
    enum type_t { TOUCH, REMOVE } type = TOUCH;

    uint64_t id = 0;
};

struct loaded_entry {
    bool ok = false;

    llama_tokens tokens;
    int32_t      n_protected = 0;
    llama_pos    pos_min     = 0;

    std::vector<std::shared_ptr<common_prompt_checkpoint>> ckpts;
    std::shared_ptr<server_prompt_data> data;
};

size_t file_size_of(size_t n_tokens, size_t n_ckpt, size_t n_payload) {
    return DISK_HEADER_SIZE + n_tokens*sizeof(llama_token) + n_ckpt*DISK_CKPT_SIZE + n_payload + DISK_TRAILER_SIZE;
}

// a temporary file older than this is the rest of an interrupted write, whatever its process id says
static constexpr int64_t DISK_TMP_MAX_AGE_S = 3600;

#ifndef _WIN32
// a temporary file of a write, ".<id>.<pid>.tmp": true if it is the rest of an interrupted write, i.e. its process is
// gone, it has our pid (a previous process with the same pid: we have not written anything yet) or it is older than
// DISK_TMP_MAX_AGE_S (the pid may have been reused)
bool is_stale_tmp(const std::string & name, const struct stat & sb) {
    const size_t p1 = name.find('.', 1);
    const size_t p2 = name.rfind('.');
    const long pid = p1 != std::string::npos && p2 > p1 ? atol(name.substr(p1 + 1, p2 - p1 - 1).c_str()) : 0;

    if (pid <= 0 || pid == (long) getpid() || (kill((pid_t) pid, 0) != 0 && errno == ESRCH)) {
        return true;
    }

    return time(nullptr) - sb.st_mtim.tv_sec > DISK_TMP_MAX_AGE_S;
}

bool is_tmp_name(const std::string & name) {
    return name.size() > 4 && name[0] == '.' && name.compare(name.size() - 4, 4, ".tmp") == 0;
}
#endif

} // namespace

struct server_cache_disk::state {
    std::string dir;
    std::string key;
    uint64_t    key_hash = 0;
    size_t      n_bytes_max = 0;

    int64_t     t_delay_ms  = 0;
    int64_t     t_wait_max_ms = 0; // how long release() waits for the writer

    std::function<void(uint64_t)> on_loaded;

    // test only: delay of the reader after a read and of the writer before a write
    // (env LLAMA_SERVER_TEST_CACHE_DISK_READ_DELAY_MS, LLAMA_SERVER_TEST_CACHE_DISK_WRITE_DELAY_MS)
    int64_t t_test_read_delay_ms  = 0;
    int64_t t_test_write_delay_ms = 0;

    // test only: every write fails with this errno once its header is written (env LLAMA_SERVER_TEST_CACHE_DISK_WRITE_ERRNO)
    int test_write_errno = 0;

    mutable std::mutex mtx;
    mutable std::condition_variable cv_jobs;   // writer and reader wait for work
    mutable std::condition_variable cv_done;   // wait() and stop() wait for progress

    // an indexed entry and its last use
    // the index is only changed by index_put(), index_touch() and index_erase(), so that it and the LRU order agree
    struct record {
        entry_ptr e;
        int64_t   t_used   = 0;     // ns since epoch
        bool      removing = false; // superseded or removed while it is loaded: removed when the load ends
    };

    // an entry of another key in the same directory root: counted in the bound and evicted like the entries of this
    // key, never loaded (a server with another key cannot use it)
    // a temporary file of a write that may still be in progress is counted too, but never removed
    struct other_file {
        size_t  n_bytes = 0;
        int64_t t_used  = 0; // ns since epoch, the mtime
        bool    pinned  = false;
    };

    // an entry to evict, of this key (id) or of another (path)
    struct victim {
        int64_t     t_used  = 0;
        uint64_t    id      = 0;
        std::string path;
        size_t      n_bytes = 0;
    };

    // guarded by mtx
    std::map<uint64_t, record>    index;
    std::map<std::string, other_file> others;  // path -> file
    size_t                        n_bytes = 0; // all files: the indexed entries, the others and the write in progress
    size_t                        n_bytes_others = 0;
    std::vector<llama_tokens>     live;        // the conversations of the slots
    std::set<uint64_t>            loading;     // reads requested or in progress
    std::set<uint64_t>            quarantined; // failed to restore, not offered again
    std::deque<uint64_t>          load_queue;
    std::map<uint64_t, std::shared_ptr<loaded_entry>> loaded;
    std::set<uint64_t>            discarded;   // reads in progress whose result nobody waits for any more
    std::map<uint64_t, staged_write> staged;   // the writes not started yet
    size_t                        staged_bytes = 0;
    std::deque<file_op>           ops;
    uint64_t                      cur_id    = 0; // the write in progress, 0 = none
    uint64_t                      cur_flush = 0;
    size_t                        cur_bytes = 0;
    size_t                        cur_reserved = 0; // counted in n_bytes while its temporary file exists
    bool                          cur_released = false;
    uint64_t                      flush_seq = 0;
    bool                          stopping   = false;
    bool                          writer_exited = false;
    bool                          reader_exited = false;
    server_cache_disk_stats       st;

    // set when the remaining writes must be abandoned, checked between I/O chunks
    std::atomic<bool> abort { false };

    // held while on_loaded() runs, so that stop() can wait for a callback in progress
    std::mutex mtx_callback;

    std::string dir_root;

    std::string path_of(uint64_t id) const {
        return dir + "/" + hex64(id) + ".bin";
    }

    // create the key directory (0700) and its IDENTITY, a description of the key for the operator
    bool make_dir(std::string & err) const {
        namespace fs = std::filesystem;

        std::error_code ec;
        fs::create_directories(dir, ec);
        if (ec) {
            err = "cannot create the cache directory '" + dir + "': " + ec.message();
            return false;
        }
        fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace, ec);

        const std::string path = dir + "/IDENTITY";
        const std::string tmp  = path + "." + std::to_string(getpid()) + ".tmp";

        FILE * f = fopen(tmp.c_str(), "wb");
        bool ok = f != nullptr && fwrite(key.data(), 1, key.size(), f) == key.size();
        ok = (f == nullptr || fclose(f) == 0) && ok;
        ok = ok && rename(tmp.c_str(), path.c_str()) == 0;
        if (!ok) {
            err = "cannot write '" + path + "': " + strerror(errno);
            unlink(tmp.c_str());
        }
        return ok;
    }

    // drop the index entries whose file is gone, e.g. after the directory was removed
    // called without the lock: the files are checked without it
    void prune_missing() {
        std::vector<std::pair<uint64_t, entry_ptr>> own;
        std::vector<std::string> other;
        {
            std::unique_lock<std::mutex> lock(mtx);
            for (const auto & [id, r] : index) {
                own.emplace_back(id, r.e);
            }
            for (const auto & [path, f] : others) {
                other.push_back(path);
            }
        }

        const auto missing = [](const std::string & path) {
            return access(path.c_str(), F_OK) != 0 && errno == ENOENT;
        };

        std::vector<std::pair<uint64_t, entry_ptr>> own_missing;
        std::vector<std::string> other_missing;
        for (const auto & [id, e] : own) {
            if (missing(path_of(id))) {
                own_missing.emplace_back(id, e);
            }
        }
        for (const auto & path : other) {
            if (missing(path)) {
                other_missing.push_back(path);
            }
        }

        std::unique_lock<std::mutex> lock(mtx);
        for (const auto & [id, e] : own_missing) {
            drop_missing_locked(id, e);
        }
        for (const auto & path : other_missing) {
            other_erase(path);
        }
    }

#ifndef _WIN32
    // a temporary file found at start-up: removed if it is the rest of an interrupted write, counted otherwise
    void scan_tmp(const std::string & path, std::vector<std::string> & files) {
        struct stat sb;
        if (stat(path.c_str(), &sb) != 0 || !S_ISREG(sb.st_mode)) {
            return;
        }
        const std::string name = std::filesystem::path(path).filename().string();
        if (is_stale_tmp(name, sb)) {
            DSK_WRN("removing interrupted write '%s'\n", path.c_str());
            files.push_back(path);
        } else {
            DSK_WRN("keeping '%s', the write of another process that runs (counted in the bound)\n", path.c_str());
            other_put(path, sb.st_size, (int64_t) sb.st_mtim.tv_sec*1000000000 + sb.st_mtim.tv_nsec, true);
        }
    }
#endif

    // the file of an indexed entry is gone: forget the entry, unless it was replaced meanwhile
    void drop_missing_locked(uint64_t id, const entry_ptr & e) {
        auto it = index.find(id);
        if (it == index.end() || it->second.e != e || loading.count(id)) {
            return;
        }
        DSK_WRN("entry %s is gone from the disk, dropping it from the index\n", hex64(id).c_str());
        index_erase(id);
        st.n_removed_missing++;
    }

    void index_put(entry_ptr e, int64_t t_used) {
        const uint64_t id = e->id;
        index_erase(id);
        n_bytes += e->n_bytes;
        index[id] = record { std::move(e), t_used, false };
    }

    void index_touch(uint64_t id, int64_t t_used) {
        if (auto it = index.find(id); it != index.end()) {
            it->second.t_used = t_used;
        }
    }

    // returns false if the entry is not indexed
    bool index_erase(uint64_t id) {
        auto it = index.find(id);
        if (it == index.end()) {
            return false;
        }
        n_bytes -= it->second.e->n_bytes;
        index.erase(it);
        return true;
    }

    // the entry is not offered to requests: being removed
    bool is_removing_locked(uint64_t id) const {
        auto it = index.find(id);
        return it != index.end() && it->second.removing;
    }

    // remove an entry and its file; an entry that is being loaded is removed when the load ends, the load completes
    // and its state is still valid for the request that waits for it
    void remove_locked(uint64_t id, std::vector<std::string> & files) {
        auto it = index.find(id);
        if (it == index.end()) {
            return;
        }
        if (loading.count(id)) {
            it->second.removing = true;
            return;
        }
        index_erase(id);
        files.push_back(path_of(id));
    }

    void other_put(const std::string & path, size_t n, int64_t t_used, bool pinned = false) {
        other_erase(path);
        others[path] = other_file { n, t_used, pinned };
        n_bytes        += n;
        n_bytes_others += n;
    }

    void other_erase(const std::string & path) {
        auto it = others.find(path);
        if (it == others.end()) {
            return;
        }
        n_bytes        -= it->second.n_bytes;
        n_bytes_others -= it->second.n_bytes;
        others.erase(it);
    }

    // the oldest entries of any key that are neither live, loading, being removed nor in `keep`, and were used before
    // t_before, until n_needed more bytes fit once n_freed bytes are released; false if they cannot be made to fit
    // (an entry to write evicts only entries used less recently than itself: otherwise it is the one to drop)
    // nothing is removed, see evict_locked()
    bool plan_evict_locked(size_t n_needed, size_t n_freed, const std::vector<uint64_t> & keep, int64_t t_before, std::vector<victim> & out) const {
        out.clear();

        size_t n_cur = n_bytes - std::min(n_bytes, n_freed);
        if (n_cur + n_needed <= n_bytes_max) {
            return true;
        }

        std::vector<victim> order;
        for (const auto & [id, r] : index) {
            if (r.removing || loading.count(id) || std::find(keep.begin(), keep.end(), id) != keep.end() || is_live_locked(*r.e)) {
                continue;
            }
            order.push_back({ r.t_used, id, std::string(), r.e->n_bytes });
        }
        for (const auto & [path, f] : others) {
            if (!f.pinned) {
                order.push_back({ f.t_used, 0, path, f.n_bytes });
            }
        }
        std::sort(order.begin(), order.end(), [](const victim & a, const victim & b) { return a.t_used < b.t_used; });

        for (auto & v : order) {
            if (n_cur + n_needed <= n_bytes_max || v.t_used >= t_before) {
                break;
            }
            n_cur -= std::min(n_cur, v.n_bytes);
            out.push_back(std::move(v));
        }

        return n_cur + n_needed <= n_bytes_max;
    }

    // caller holds mtx, the files are removed by the caller without the lock
    void evict_locked(const std::vector<victim> & victims, std::vector<std::string> & files) {
        for (const auto & v : victims) {
            if (v.path.empty()) {
                DSK_INF("evicting entry %s (%d tokens, %.3f MiB)\n", hex64(v.id).c_str(), index.at(v.id).e->prompt.n_tokens(), v.n_bytes / (1024.0 * 1024.0));
                remove_locked(v.id, files);
            } else {
                DSK_INF("evicting entry '%s' of another key (%.3f MiB)\n", v.path.c_str(), v.n_bytes / (1024.0 * 1024.0));
                other_erase(v.path);
                files.push_back(v.path);
            }
            st.n_removed_evicted++;
        }
    }

    // a slot continues the entry: it keeps the entry's protected prefix
    bool is_live_locked(const entry & e) const {
        const auto & tokens = e.prompt.tokens.get_tokens();
        const size_t n = e.prompt.n_tokens_protected();
        for (const auto & conv : live) {
            if (conv.size() >= n && std::equal(tokens.begin(), tokens.begin() + n, conv.begin())) {
                return true;
            }
        }
        return false;
    }

    void set_error_locked(const std::string & msg) {
        st.last_error = msg;
    }

    // remove the files, and the directories of other keys that they leave empty
    void unlink_files(const std::vector<std::string> & files) const {
        std::set<std::string> dirs;
        for (const auto & f : files) {
            std::error_code ec;
            std::filesystem::remove(f, ec);
            if (ec) {
                DSK_WRN("failed to remove '%s': %s\n", f.c_str(), ec.message().c_str());
            }
            const std::string parent = std::filesystem::path(f).parent_path().string();
            if (parent != dir) {
                dirs.insert(parent);
            }
        }
        for (const auto & d : dirs) {
            remove_stale_key_dir(d);
        }
    }

    // a key directory without entries: only its IDENTITY is left
    static bool remove_stale_key_dir(const std::string & path) {
        namespace fs = std::filesystem;

        std::error_code ec;
        for (const auto & de : fs::directory_iterator(path, ec)) {
            if (de.path().filename() != "IDENTITY") {
                return false;
            }
        }
        if (ec) {
            return false;
        }

        fs::remove(fs::path(path) / "IDENTITY", ec);
        if (!fs::remove(path, ec) || ec) {
            return false;
        }
        DSK_INF("removed the empty cache directory '%s'\n", path.c_str());
        return true;
    }

    // parse and validate the header and the meta section; meta receives the meta bytes
    bool parse_header(const uint8_t * hdr, size_t file_size, file_header & h) const {
        byte_reader r { hdr, DISK_HEADER_SIZE };
        r.get(h.magic); r.get(h.version); r.get(h.key_hash); r.get(h.id);
        r.get(h.n_tokens); r.get(h.n_protected); r.get(h.pos_min); r.get(h.n_ckpt);
        r.get(h.size_main); r.get(h.size_drft); r.get(h.size_meta); r.get(h.size_total); r.get(h.meta_hash);

        if (h.magic != DISK_MAGIC || h.version != DISK_VERSION || h.key_hash != key_hash) {
            return false;
        }
        // bound the counts by the file size before any allocation, the fields may be corrupt
        if (h.size_total != file_size || h.n_tokens == 0 || h.n_tokens > file_size || h.n_ckpt > file_size) {
            return false;
        }
        if (h.size_meta != h.n_tokens*sizeof(llama_token) + h.n_ckpt*DISK_CKPT_SIZE) {
            return false;
        }
        if (h.size_main > file_size || h.size_drft > file_size || DISK_HEADER_SIZE + h.size_meta + DISK_TRAILER_SIZE > file_size) {
            return false;
        }
        return true;
    }

#ifndef _WIN32
    static bool read_all(int fd, void * dst, size_t n, const std::atomic<bool> & abort, xxh64 * h) {
        uint8_t * p = (uint8_t *) dst;
        while (n > 0) {
            if (abort.load()) {
                return false;
            }
            const size_t k = std::min(n, DISK_IO_CHUNK);
            const ssize_t r = ::read(fd, p, k);
            if (r < 0 && errno == EINTR) {
                continue;
            }
            if (r <= 0) {
                return false;
            }
            if (h) {
                h->update(p, r);
            }
            p += r;
            n -= r;
        }
        return true;
    }

    static bool write_all(int fd, const void * src, size_t n, const std::atomic<bool> & abort, xxh64 * h) {
        const uint8_t * p = (const uint8_t *) src;
        if (h) {
            h->update(p, n);
        }
        while (n > 0) {
            if (abort.load()) {
                errno = ECANCELED;
                return false;
            }
            const size_t k = std::min(n, DISK_IO_CHUNK);
            const ssize_t r = ::write(fd, p, k);
            if (r < 0 && errno == EINTR) {
                continue;
            }
            if (r <= 0) {
                return false;
            }
            p += r;
            n -= r;
        }
        return true;
    }

    // read the header and the meta section of a file, validate them and build the index entry
    // returns nullptr if the file is not a valid entry of this key
    entry_ptr read_entry(const std::string & path, int64_t & mtime_ns) const {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            return nullptr;
        }

        struct stat sb;
        entry_ptr res;

        do {
            if (fstat(fd, &sb) != 0) {
                break;
            }
            mtime_ns = (int64_t) sb.st_mtim.tv_sec*1000000000 + sb.st_mtim.tv_nsec;

            uint8_t hdr[DISK_HEADER_SIZE];
            file_header h;
            if (!read_all(fd, hdr, sizeof(hdr), abort, nullptr) || !parse_header(hdr, sb.st_size, h)) {
                break;
            }

            std::vector<uint8_t> meta(h.size_meta);
            if (!read_all(fd, meta.data(), meta.size(), abort, nullptr)) {
                break;
            }

            xxh64 hm;
            hm.update(hdr, DISK_HEADER_SIZE - sizeof(uint64_t));
            hm.update(meta.data(), meta.size());
            if (hm.digest() != h.meta_hash) {
                break;
            }

            llama_tokens tokens(h.n_tokens);
            memcpy(tokens.data(), meta.data(), h.n_tokens*sizeof(llama_token));
            if (server_cache_disk_hash_tokens(tokens) != h.id) {
                break;
            }

            auto e = std::make_shared<entry>();
            e->id      = h.id;
            e->pos_min = h.pos_min;
            e->n_bytes = sb.st_size;

            size_t n_payload = h.size_main + h.size_drft;
            byte_reader r { meta.data() + h.n_tokens*sizeof(llama_token), h.n_ckpt*DISK_CKPT_SIZE };
            bool ok = true;
            for (uint64_t i = 0; i < h.n_ckpt && ok; ++i) {
                ckpt_meta c;
                ok = r.get(c.n_tokens) && r.get(c.pos_min) && r.get(c.pos_max) && r.get(c.size_tgt) && r.get(c.size_dft) && r.get(c.size_spec);
                ok = ok && c.size_tgt <= (uint64_t) sb.st_size && c.size_dft <= (uint64_t) sb.st_size && c.size_spec <= (uint64_t) sb.st_size;
                if (ok) {
                    // each term is at most the file size, the sum cannot overflow before it is checked
                    n_payload += c.size_tgt + c.size_dft + c.size_spec;
                    ok = n_payload <= (uint64_t) sb.st_size;
                }
                if (ok) {
                    auto ckpt = std::make_shared<common_prompt_checkpoint>();
                    ckpt->id_task  = -1;
                    ckpt->n_tokens = c.n_tokens;
                    ckpt->pos_min  = c.pos_min;
                    ckpt->pos_max  = c.pos_max;
                    e->prompt.checkpoints.push_back(std::move(ckpt));
                }
            }
            if (!ok || file_size_of(h.n_tokens, h.n_ckpt, n_payload) != (size_t) sb.st_size) {
                break;
            }

            e->prompt.tokens      = server_tokens(tokens, false);
            e->prompt.n_protected = h.n_protected;

            res = std::move(e);
        } while (false);

        ::close(fd);

        return res;
    }

    // read a whole entry, verifying the payload checksum
    bool read_full(uint64_t id, loaded_entry & out, bool & corrupt) const {
        corrupt = false;

        const std::string path = path_of(id);
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            return false;
        }

#if defined(POSIX_FADV_SEQUENTIAL)
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif

        bool ok = false;
        corrupt = true;

        do {
            struct stat sb;
            if (fstat(fd, &sb) != 0) {
                corrupt = false;
                break;
            }

            // the sizes in the header are checked against the file length before anything is allocated: a file larger
            // than the bound (e.g. a sparse file) is not read
            if ((uint64_t) sb.st_size > n_bytes_max) {
                DSK_WRN("entry %s is larger than the disk cache limit, not reading it\n", hex64(id).c_str());
                corrupt = false;
                break;
            }

            uint8_t hdr[DISK_HEADER_SIZE];
            file_header h;
            if (!read_all(fd, hdr, sizeof(hdr), abort, nullptr) || !parse_header(hdr, sb.st_size, h) || h.id != id) {
                break;
            }

            std::vector<uint8_t> meta(h.size_meta);
            if (!read_all(fd, meta.data(), meta.size(), abort, nullptr)) {
                break;
            }

            xxh64 hm;
            hm.update(hdr, DISK_HEADER_SIZE - sizeof(uint64_t));
            hm.update(meta.data(), meta.size());
            if (hm.digest() != h.meta_hash) {
                break;
            }

            out.tokens.resize(h.n_tokens);
            memcpy(out.tokens.data(), meta.data(), h.n_tokens*sizeof(llama_token));
            out.n_protected = h.n_protected;
            out.pos_min     = h.pos_min;

            std::vector<ckpt_meta> cms(h.n_ckpt);
            size_t n_payload = h.size_main + h.size_drft;
            {
                byte_reader r { meta.data() + h.n_tokens*sizeof(llama_token), h.n_ckpt*DISK_CKPT_SIZE };
                bool ok_meta = true;
                for (auto & c : cms) {
                    ok_meta = ok_meta && r.get(c.n_tokens) && r.get(c.pos_min) && r.get(c.pos_max) && r.get(c.size_tgt) && r.get(c.size_dft) && r.get(c.size_spec);
                    ok_meta = ok_meta && c.size_tgt <= (uint64_t) sb.st_size && c.size_dft <= (uint64_t) sb.st_size && c.size_spec <= (uint64_t) sb.st_size;
                    if (ok_meta) {
                        n_payload += c.size_tgt + c.size_dft + c.size_spec;
                        ok_meta = n_payload <= (uint64_t) sb.st_size;
                    }
                }
                if (!ok_meta || file_size_of(h.n_tokens, h.n_ckpt, n_payload) != (size_t) sb.st_size) {
                    break;
                }
            }

            xxh64 hp;

            out.data = std::make_shared<server_prompt_data>();
            out.data->main.resize(h.size_main);
            out.data->drft.resize(h.size_drft);
            if (!read_all(fd, out.data->main.data(), h.size_main, abort, &hp) ||
                !read_all(fd, out.data->drft.data(), h.size_drft, abort, &hp)) {
                corrupt = !abort.load();
                break;
            }

            bool ok_ckpt = true;
            for (const auto & c : cms) {
                auto ckpt = std::make_shared<common_prompt_checkpoint>();
                ckpt->id_task  = -1; // not created by a task - may fail to load, see update_slots()
                ckpt->n_tokens = c.n_tokens;
                ckpt->pos_min  = c.pos_min;
                ckpt->pos_max  = c.pos_max;
                ckpt->data_tgt.resize(c.size_tgt);
                ckpt->data_dft.resize(c.size_dft);
                ckpt->data_spec.resize(c.size_spec);
                if (!read_all(fd, ckpt->data_tgt.data(),  c.size_tgt,  abort, &hp) ||
                    !read_all(fd, ckpt->data_dft.data(),  c.size_dft,  abort, &hp) ||
                    !read_all(fd, ckpt->data_spec.data(), c.size_spec, abort, &hp)) {
                    ok_ckpt = false;
                    break;
                }
                out.ckpts.push_back(std::move(ckpt));
            }
            if (!ok_ckpt) {
                corrupt = !abort.load();
                break;
            }

            uint8_t trl[DISK_TRAILER_SIZE];
            if (!read_all(fd, trl, sizeof(trl), abort, nullptr)) {
                break;
            }
            uint64_t data_hash = 0;
            uint32_t magic_end = 0;
            memcpy(&data_hash, trl, sizeof(data_hash));
            memcpy(&magic_end, trl + sizeof(data_hash), sizeof(magic_end));
            if (magic_end != DISK_MAGIC_END || data_hash != hp.digest()) {
                break;
            }

            ok = true;
            corrupt = false;
        } while (false);

        ::close(fd);

        return ok;
    }

    // write an entry: temporary file, fsync, rename, fsync of the directory
    // dir_missing: the directory is gone, see make_dir()
    bool write_file(const write_job & job, std::string & err, bool & dir_missing) const {
        const std::string path = path_of(job.id);
        const std::string tmp  = dir + "/." + hex64(job.id) + "." + std::to_string(getpid()) + ".tmp";

        // everything that allocates comes first: an exception leaves no temporary file behind
        size_t n_payload = job.data->main.size() + job.data->drft.size();

        byte_writer meta;
        meta.put_raw(job.tokens.data(), job.tokens.size()*sizeof(llama_token));
        for (const auto & c : job.ckpts) {
            meta.put<int64_t> (c->n_tokens);
            meta.put<int32_t> (c->pos_min);
            meta.put<int32_t> (c->pos_max);
            meta.put<uint64_t>(c->data_tgt.size());
            meta.put<uint64_t>(c->data_dft.size());
            meta.put<uint64_t>(c->data_spec.size());
            n_payload += c->data_tgt.size() + c->data_dft.size() + c->data_spec.size();
        }

        byte_writer hdr;
        hdr.put<uint32_t>(DISK_MAGIC);
        hdr.put<uint32_t>(DISK_VERSION);
        hdr.put<uint64_t>(key_hash);
        hdr.put<uint64_t>(job.id);
        hdr.put<uint64_t>(job.tokens.size());
        hdr.put<int64_t> (job.n_protected);
        hdr.put<int64_t> (job.pos_min);
        hdr.put<uint64_t>(job.ckpts.size());
        hdr.put<uint64_t>(job.data->main.size());
        hdr.put<uint64_t>(job.data->drft.size());
        hdr.put<uint64_t>(meta.buf.size());
        hdr.put<uint64_t>(file_size_of(job.tokens.size(), job.ckpts.size(), n_payload));
        {
            xxh64 hm;
            hm.update(hdr.buf.data(), hdr.buf.size());
            hm.update(meta.buf.data(), meta.buf.size());
            hdr.put<uint64_t>(hm.digest());
        }
        GGML_ASSERT(hdr.buf.size() == DISK_HEADER_SIZE);

        const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (fd < 0) {
            dir_missing = errno == ENOENT;
            err = std::string("cannot create '") + tmp + "': " + strerror(errno);
            return false;
        }

        xxh64 hp;

        bool ok = write_all(fd, hdr.buf.data(),  hdr.buf.size(),  abort, nullptr);

        if (ok && test_write_errno != 0) {
            errno = test_write_errno;
            ok    = false;
        }

        ok = ok && write_all(fd, meta.buf.data(), meta.buf.size(), abort, nullptr)
                && write_all(fd, job.data->main.data(), job.data->main.size(), abort, &hp)
                && write_all(fd, job.data->drft.data(), job.data->drft.size(), abort, &hp);

        for (const auto & c : job.ckpts) {
            ok = ok && write_all(fd, c->data_tgt.data(),  c->data_tgt.size(),  abort, &hp)
                    && write_all(fd, c->data_dft.data(),  c->data_dft.size(),  abort, &hp)
                    && write_all(fd, c->data_spec.data(), c->data_spec.size(), abort, &hp);
        }

        if (ok) {
            byte_writer trl;
            trl.put<uint64_t>(hp.digest());
            trl.put<uint32_t>(DISK_MAGIC_END);
            trl.put<uint32_t>(0);
            ok = write_all(fd, trl.buf.data(), trl.buf.size(), abort, nullptr);
        }

        if (ok) {
            const struct timespec ts[2] = {
                { (time_t) (job.t_used / 1000000000), (long) (job.t_used % 1000000000) },
                { (time_t) (job.t_used / 1000000000), (long) (job.t_used % 1000000000) },
            };
            futimens(fd, ts);
        }

        ok = ok && fsync(fd) == 0;

        if (!ok) {
            err = std::string("cannot write '") + tmp + "': " + strerror(errno);
        }

        if (::close(fd) != 0 && ok) {
            ok  = false;
            err = std::string("cannot close '") + tmp + "': " + strerror(errno);
        }

        if (ok && rename(tmp.c_str(), path.c_str()) != 0) {
            ok  = false;
            err = std::string("cannot rename '") + tmp + "': " + strerror(errno);
        }

        if (!ok) {
            ::unlink(tmp.c_str());
            return false;
        }

        const int fd_dir = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
        if (fd_dir >= 0) {
            fsync(fd_dir);
            ::close(fd_dir);
        }

        return true;
    }
#endif

    void process_write(write_job & job) {
#ifndef _WIN32
        std::vector<std::string> files;

        {
            std::unique_lock<std::mutex> lock(mtx);

            if (index.count(job.id) && !index.at(job.id).removing) {
                index_touch(job.id, time_now_ns());
                ops.push_back({ file_op::TOUCH, job.id });
                st.n_write_present++;
                return;
            }

            if (job.n_bytes > n_bytes_max) {
                DSK_WRN("entry %s (%.3f MiB) is larger than the disk cache limit (%.3f MiB), skipping\n",
                        hex64(job.id).c_str(), job.n_bytes / (1024.0 * 1024.0), n_bytes_max / (1024.0 * 1024.0));
                st.n_write_no_room++;
                return;
            }

            // the entries that the new one continues are removed once it is written: they make room first
            // (they stay on disk during the write, in case it fails)
            const server_tokens tokens(job.tokens, false);
            std::vector<uint64_t> superseded;
            size_t n_freed = 0;
            for (const auto & [id, r] : index) {
                if (!r.removing && r.e->prompt.is_continued_by(tokens)) {
                    superseded.push_back(id);
                    n_freed += loading.count(id) ? 0 : r.e->n_bytes;
                }
            }

            // then the entries used less recently than this one; if the entries that are kept (live, being read, or
            // used more recently) leave no room, the write is skipped and the entry stays in RAM
            std::vector<victim> victims;
            if (!plan_evict_locked(job.n_bytes, n_freed, superseded, job.t_used, victims)) {
                DSK_WRN("no room for entry %s (%.3f MiB): live or more recently used entries fill the disk cache limit (%.3f of %.3f MiB), skipping\n",
                        hex64(job.id).c_str(), job.n_bytes / (1024.0 * 1024.0), n_bytes / (1024.0 * 1024.0), n_bytes_max / (1024.0 * 1024.0));
                st.n_write_no_room++;
                return;
            }
            evict_locked(victims, files);

            // the temporary file counts while it is written
            n_bytes     += job.n_bytes;
            cur_reserved = job.n_bytes;
        }

        unlink_files(files);
        files.clear();

        const int64_t t_start = ggml_time_us();

        if (t_test_write_delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(t_test_write_delay_ms));
        }

        std::string err;
        bool dir_missing = false;
        bool ok = write_file(job, err, dir_missing);

        // the directory was removed under us: create it again, forget the entries that went with it, retry
        if (!ok && dir_missing) {
            DSK_WRN("the cache directory '%s' is gone, creating it again\n", dir.c_str());
            dir_missing = false;
            ok = make_dir(err);
            prune_missing();
            ok = ok && write_file(job, err, dir_missing);
        }

        const double t_ms = (ggml_time_us() - t_start) / 1000.0;

        std::unique_lock<std::mutex> lock(mtx);

        n_bytes     -= cur_reserved;
        cur_reserved = 0;

        if (!ok) {
            if (!abort.load()) {
                DSK_ERR("failed to write entry %s: %s\n", hex64(job.id).c_str(), err.c_str());
                st.n_write_errors++;
                set_error_locked(err);
            }
            return;
        }

        auto e = std::make_shared<entry>();
        e->id      = job.id;
        e->pos_min = job.pos_min;
        e->n_bytes = job.n_bytes;
        e->prompt.tokens      = server_tokens(job.tokens, false);
        e->prompt.n_protected = job.n_protected;
        for (const auto & c : job.ckpts) {
            auto ckpt = std::make_shared<common_prompt_checkpoint>();
            ckpt->id_task  = -1;
            ckpt->n_tokens = c->n_tokens;
            ckpt->pos_min  = c->pos_min;
            ckpt->pos_max  = c->pos_max;
            e->prompt.checkpoints.push_back(std::move(ckpt));
        }

        // the entries that the new one continues are contained in it
        std::vector<uint64_t> superseded;
        for (const auto & [id, r] : index) {
            if (id != job.id && !r.removing && r.e->prompt.is_continued_by(e->prompt.tokens)) {
                superseded.push_back(id);
            }
        }
        for (const uint64_t id : superseded) {
            DSK_INF("removing entry %s (%d tokens), superseded by %s%s\n", hex64(id).c_str(), index.at(id).e->prompt.n_tokens(),
                    hex64(job.id).c_str(), loading.count(id) ? " once it is loaded" : "");
            remove_locked(id, files);
            st.n_removed_superseded++;
        }

        index_put(e, job.t_used);

        st.n_writes++;
        st.n_bytes_written += job.n_bytes;
        st.t_write_ms      += t_ms;

        DSK_INF("wrote entry %s: %zu tokens, %zu checkpoints, %.3f MiB in %.2f ms\n",
                hex64(job.id).c_str(), job.tokens.size(), job.ckpts.size(), job.n_bytes / (1024.0 * 1024.0), t_ms);

        lock.unlock();

        unlink_files(files);
#else
        GGML_UNUSED(job);
#endif
    }

    // the next write to do: one that the RAM cache released first (it holds memory the RAM cache no longer counts),
    // then the most recently saved entry, so that a deadline loses the least valuable ones
    std::map<uint64_t, staged_write>::iterator pick_locked(int64_t t_now) {
        auto res = staged.end();
        for (auto it = staged.begin(); it != staged.end(); ++it) {
            const auto & w = it->second;
            if (!(stopping || w.released || w.flush != 0 || w.t_due_ms <= t_now)) {
                continue;
            }
            if (res == staged.end()) {
                res = it;
                continue;
            }
            const auto & b = res->second;
            if (w.released != b.released ? w.released : (w.released ? w.t_staged_ms < b.t_staged_ms : w.t_staged_ms > b.t_staged_ms)) {
                res = it;
            }
        }
        return res;
    }

    // until when the writer may sleep
    int64_t t_next_due_locked() const {
        int64_t res = INT64_MAX;
        for (const auto & [id, w] : staged) {
            res = std::min(res, w.t_due_ms);
        }
        return res;
    }

    void process_op(const file_op & op) {
        switch (op.type) {
            case file_op::TOUCH:
                {
#ifndef _WIN32
                    // the mtime keeps the LRU order across restarts
                    if (utimensat(AT_FDCWD, path_of(op.id).c_str(), nullptr, 0) != 0 && errno == ENOENT) {
                        std::unique_lock<std::mutex> lock(mtx);
                        if (auto it = index.find(op.id); it != index.end()) {
                            drop_missing_locked(op.id, it->second.e);
                        }
                    }
#endif
                } break;
            case file_op::REMOVE:
                unlink_files({ path_of(op.id) });
                break;
        }
    }

    void writer_loop() {
        while (true) {
            std::unique_lock<std::mutex> lock(mtx);

            file_op op;
            bool    has_op = false;
            write_job job;

            while (true) {
                if (abort.load()) {
                    break;
                }
                if (!ops.empty()) {
                    op = ops.front();
                    ops.pop_front();
                    has_op = true;
                    break;
                }
                const int64_t t_now = ggml_time_ms();
                if (auto it = pick_locked(t_now); it != staged.end()) {
                    job = std::move(it->second.job);
                    cur_id      = it->first;
                    cur_flush   = it->second.flush;
                    cur_bytes   = job.n_bytes;
                    cur_released = it->second.released;
                    staged_bytes -= job.n_bytes;
                    staged.erase(it);
                    break;
                }
                if (stopping && staged.empty()) {
                    break;
                }
                const int64_t t_due = t_next_due_locked();
                if (t_due == INT64_MAX) {
                    cv_jobs.wait(lock);
                } else {
                    cv_jobs.wait_for(lock, std::chrono::milliseconds(std::max<int64_t>(1, t_due - t_now)));
                }
            }

            if (!has_op && cur_id == 0) {
                break;
            }

            lock.unlock();

            // an exception (e.g. std::bad_alloc) fails the job, not the server
            try {
                if (has_op) {
                    process_op(op);
                } else {
                    process_write(job);
                }
            } catch (const std::exception & e) {
                const uint64_t id = has_op ? op.id : job.id;
                DSK_ERR("failed to %s entry %s: %s\n", has_op ? "update" : "write", hex64(id).c_str(), e.what());

                std::unique_lock<std::mutex> lock_st(mtx);
                n_bytes     -= cur_reserved;
                cur_reserved = 0;
                st.n_write_errors++;
                set_error_locked(std::string("cannot write entry ") + hex64(id) + ": " + e.what());
            }

            if (has_op) {
                continue;
            }

            // release the data before waking up the waiters
            job = write_job();

            lock.lock();
            cur_id       = 0;
            cur_flush    = 0;
            cur_bytes    = 0;
            cur_released = false;
            lock.unlock();

            cv_done.notify_all();
        }

        {
            std::unique_lock<std::mutex> lock(mtx);
            writer_exited = true;
        }
        cv_done.notify_all();
    }

    void reader_loop() {
        while (true) {
            uint64_t  id = 0;
            entry_ptr e0; // the entry as indexed when the read starts: it may be replaced meanwhile
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv_jobs.wait(lock, [&] { return stopping || !load_queue.empty(); });
                if (stopping) {
                    break;
                }
                id = load_queue.front();
                load_queue.pop_front();
                if (auto it = index.find(id); it != index.end()) {
                    e0 = it->second.e;
                }
            }

            auto res = std::make_shared<loaded_entry>();

            const int64_t t_start = ggml_time_us();

            bool corrupt   = false;
            bool exception = false; // e.g. std::bad_alloc: the entry may be fine, it stays indexed
#ifndef _WIN32
            try {
                res->ok = read_full(id, *res, corrupt);
            } catch (const std::exception & e) {
                DSK_ERR("failed to read entry %s: %s\n", hex64(id).c_str(), e.what());
                res = std::make_shared<loaded_entry>();
                corrupt   = false;
                exception = true;
            }
#endif

            if (t_test_read_delay_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(t_test_read_delay_ms));
            }

            const double t_ms = (ggml_time_us() - t_start) / 1000.0;

            {
                std::unique_lock<std::mutex> lock(mtx);

                loading.erase(id);

                // the record that was read, unless a write replaced it meanwhile
                auto it = index.find(id);
                const bool same = e0 && it != index.end() && it->second.e == e0;

                const size_t n_bytes_entry = e0 ? e0->n_bytes : 0;

                // the files are removed and touched by the writer, in order with the writes of the same entry
                std::vector<std::string> files;
                bool remove = false;

                if (res->ok) {
                    st.n_reads++;
                    st.n_bytes_read += n_bytes_entry;
                    st.t_read_ms    += t_ms;
                    DSK_INF("read entry %s: %zu tokens, %.3f MiB in %.2f ms\n", hex64(id).c_str(), res->tokens.size(), n_bytes_entry / (1024.0 * 1024.0), t_ms);

                    // the load is over: an entry superseded or removed meanwhile goes now
                    remove = same && it->second.removing;
                    if (same && !remove) {
                        index_touch(id, time_now_ns());
                        ops.push_back({ file_op::TOUCH, id });
                    }
                } else {
                    st.n_read_errors++;
                    if (corrupt) {
                        DSK_ERR("entry %s is corrupt, removing it\n", hex64(id).c_str());
                        set_error_locked("corrupt entry " + hex64(id));
                        st.n_removed_corrupt++;
                        remove = same;
                    } else if (exception) {
                        set_error_locked("cannot read entry " + hex64(id) + ": out of memory");
                    } else if (!abort.load()) {
                        // the file is missing or unreadable: forget the entry (a restart indexes it again if it is there),
                        // so that later requests do not wait for it again
                        DSK_ERR("failed to read entry %s, dropping it from the index\n", hex64(id).c_str());
                        set_error_locked("cannot read entry " + hex64(id));
                        if (same) {
                            remove = it->second.removing;
                            if (!remove) {
                                index_erase(id);
                                st.n_removed_missing++;
                            }
                        }
                    }
                    res = std::make_shared<loaded_entry>();
                }

                if (remove) {
                    remove_locked(id, files);
                    if (!files.empty()) {
                        ops.push_back({ file_op::REMOVE, id });
                    }
                }

                // the task that wanted it was cancelled: release the memory now
                if (discarded.erase(id) == 0) {
                    loaded[id] = res;
                }
            }

            cv_jobs.notify_all();

            {
                std::unique_lock<std::mutex> lock(mtx_callback);
                bool cancelled;
                {
                    std::unique_lock<std::mutex> lock_st(mtx);
                    cancelled = stopping;
                }
                if (!cancelled && on_loaded) {
                    on_loaded(id);
                }
            }
        }

        {
            std::unique_lock<std::mutex> lock(mtx);
            reader_exited = true;
        }
        cv_done.notify_all();
    }

    std::thread writer;
    std::thread reader;
};

std::string server_cache_disk_file_identity(const std::string & path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        return "missing";
    }

    constexpr size_t n_part = 1024*1024;

    std::string res = "size=" + std::to_string(size);

    FILE * f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return res + ",unreadable";
    }

    std::vector<uint8_t> buf(n_part);

    const size_t n_head = fread(buf.data(), 1, std::min<size_t>(n_part, size), f);
    res += ",head=" + hex64(server_cache_disk_hash(buf.data(), n_head));

    if (size > n_part && fseek(f, -(long) n_part, SEEK_END) == 0) {
        const size_t n_tail = fread(buf.data(), 1, n_part, f);
        res += ",tail=" + hex64(server_cache_disk_hash(buf.data(), n_tail));
    }

    fclose(f);

    return res;
}

//
// server_cache_disk
//

std::unique_ptr<server_cache_disk> server_cache_disk::open(
        const std::string & dir_root,
        const std::string & key_name,
        const std::string & key,
        size_t n_bytes_max,
        int64_t t_delay_ms,
        int64_t t_wait_max_ms,
        std::function<void(uint64_t)> on_loaded) {
#ifdef _WIN32
    GGML_UNUSED(dir_root);
    GGML_UNUSED(key_name);
    GGML_UNUSED(key);
    GGML_UNUSED(n_bytes_max);
    GGML_UNUSED(t_delay_ms);
    GGML_UNUSED(t_wait_max_ms);
    GGML_UNUSED(on_loaded);
    DSK_ERR("%s", "the disk prompt cache is not supported on Windows\n");
    return nullptr;
#else
    namespace fs = std::filesystem;

    const uint64_t key_hash = server_cache_disk_hash(key.data(), key.size());
    const std::string dir = (fs::path(dir_root) / (key_name + "-" + hex64(key_hash))).string();

    std::unique_ptr<server_cache_disk> res(new server_cache_disk());
    res->path_dir = dir;
    res->st = std::make_shared<state>();

    auto & s = *res->st;
    s.dir         = dir;
    s.key         = key;

    std::error_code ec;
    if (std::string err; !s.make_dir(err)) {
        DSK_ERR("%s\n", err.c_str());
        return nullptr;
    }

    if (access(dir.c_str(), R_OK | W_OK | X_OK) != 0) {
        DSK_ERR("the cache directory '%s' is not writable: %s\n", dir.c_str(), strerror(errno));
        return nullptr;
    }

    s.key_hash    = key_hash;
    s.n_bytes_max = n_bytes_max;
    s.t_delay_ms  = t_delay_ms;
    s.t_wait_max_ms = t_wait_max_ms;
    s.on_loaded   = std::move(on_loaded);

    if (const char * v = getenv("LLAMA_SERVER_TEST_CACHE_DISK_READ_DELAY_MS")) {
        s.t_test_read_delay_ms = atoll(v);
        DSK_WRN("test: the reader waits %" PRId64 " ms after each read\n", s.t_test_read_delay_ms);
    }
    if (const char * v = getenv("LLAMA_SERVER_TEST_CACHE_DISK_WRITE_ERRNO")) {
        s.test_write_errno = atoi(v);
        DSK_WRN("test: every write fails with errno %d\n", s.test_write_errno);
    }
    if (const char * v = getenv("LLAMA_SERVER_TEST_CACHE_DISK_WRITE_DELAY_MS")) {
        s.t_test_write_delay_ms = atoll(v);
        DSK_WRN("test: the writer waits %" PRId64 " ms before each write\n", s.t_test_write_delay_ms);
    }
    s.st.n_bytes_max = n_bytes_max;

    // index the entries: headers and tokens only
    const int64_t t_start = ggml_time_us();

    std::vector<std::string> files;
    for (const auto & de : fs::directory_iterator(dir, ec)) {
        const std::string name = de.path().filename().string();

        if (is_tmp_name(name)) {
            s.scan_tmp(de.path().string(), files);
            continue;
        }

        if (name.size() != 16 + 4 || name.compare(16, 4, ".bin") != 0) {
            continue;
        }

        int64_t mtime_ns = 0;
        entry_ptr e;
        try {
            e = s.read_entry(de.path().string(), mtime_ns);
        } catch (const std::exception & ex) {
            DSK_ERR("cannot index '%s': %s, skipping it\n", name.c_str(), ex.what());
            continue;
        }
        if (!e || hex64(e->id) + ".bin" != name) {
            DSK_WRN("removing invalid or truncated entry '%s'\n", name.c_str());
            s.st.n_removed_corrupt++;
            files.push_back(de.path().string());
            continue;
        }

        s.index_put(std::move(e), mtime_ns);
    }

    // the entries of the other keys in the directory root count in the same bound
    s.dir_root = dir_root;
    for (const auto & de : fs::directory_iterator(dir_root, ec)) {
        if (!de.is_directory() || de.path().filename() == fs::path(dir).filename() || !fs::exists(de.path() / "IDENTITY")) {
            continue;
        }

        size_t n_files = 0;
        for (const auto & df : fs::directory_iterator(de.path(), ec)) {
            const std::string name = df.path().filename().string();
            if (name == "IDENTITY") {
                continue;
            }
            n_files++;

            if (is_tmp_name(name)) {
                s.scan_tmp(df.path().string(), files);
                continue;
            }

            struct stat sb;
            if (name.size() == 16 + 4 && name.compare(16, 4, ".bin") == 0 && stat(df.path().c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) {
                s.other_put(df.path().string(), sb.st_size, (int64_t) sb.st_mtim.tv_sec*1000000000 + sb.st_mtim.tv_nsec);
            }
        }

        if (n_files == 0) {
            state::remove_stale_key_dir(de.path().string());
        }
    }

    // the bound may be lower than in the previous run
    {
        std::vector<state::victim> victims;
        s.plan_evict_locked(0, 0, {}, INT64_MAX, victims);
        s.evict_locked(victims, files);
    }

    s.unlink_files(files);

    DSK_INF("disk prompt cache in '%s': %zu entries, %.3f MiB, other keys %zu entries, %.3f MiB (limit %.3f MiB), write delay %.1f s, indexed in %.2f ms\n",
            dir.c_str(), s.index.size(), (s.n_bytes - s.n_bytes_others) / (1024.0 * 1024.0), s.others.size(), s.n_bytes_others / (1024.0 * 1024.0),
            n_bytes_max / (1024.0 * 1024.0), t_delay_ms / 1000.0, (ggml_time_us() - t_start) / 1000.0);

    auto sp = res->st;
    s.writer = std::thread([sp]() { sp->writer_loop(); });
    s.reader = std::thread([sp]() { sp->reader_loop(); });

    return res;
#endif
}

server_cache_disk::~server_cache_disk() {
    stop(0);
}

// the id of an entry that can be written: text only, with a state
static bool disk_entry_id(const server_prompt_cache_state & state, llama_tokens & tokens, uint64_t & id) {
    // media chunks are not written, their tokens are placeholders
    if (!state.data || !server_cache_disk_text_tokens(state.prompt.tokens, tokens) || tokens.empty()) {
        return false;
    }
    id = server_cache_disk_hash_tokens(tokens);
    return true;
}

void server_cache_disk::stage_locked(const server_prompt_cache_state & state, bool released) {
    llama_tokens tokens;
    uint64_t id = 0;
    if (st->stopping || !disk_entry_id(state, tokens, id)) {
        return;
    }

    if (auto it = st->staged.find(id); it != st->staged.end()) {
        it->second.released = it->second.released || released;
        return;
    }

    if (id == st->cur_id) {
        st->cur_released = st->cur_released || released;
        return;
    }

    if (st->index.count(id) && !st->is_removing_locked(id)) {
        st->index_touch(id, time_now_ns());
        st->st.n_write_present++;
        return;
    }

    staged_write w;
    w.job.id          = id;
    w.job.n_protected = state.prompt.n_protected;
    w.job.pos_min     = state.pos_min;
    w.job.data        = state.data;

    size_t n_payload = state.data->main.size() + state.data->drft.size();
    for (const auto & c : state.prompt.checkpoints) {
        w.job.ckpts.push_back(c);
        n_payload += c->data_tgt.size() + c->data_dft.size() + c->data_spec.size();
    }
    w.job.n_bytes = file_size_of(tokens.size(), w.job.ckpts.size(), n_payload);
    w.job.tokens  = std::move(tokens);
    w.job.t_used  = time_now_ns();

    w.t_staged_ms = ggml_time_ms();
    w.t_due_ms    = released ? w.t_staged_ms : w.t_staged_ms + st->t_delay_ms;
    w.released    = released;

    st->staged_bytes += w.job.n_bytes;
    st->staged.emplace(id, std::move(w));
}

void server_cache_disk::stage(const server_prompt_cache_state & state) {
    {
        std::unique_lock<std::mutex> lock(st->mtx);
        stage_locked(state, false);
    }
    st->cv_jobs.notify_all();
}

void server_cache_disk::drop(uint64_t id) {
    std::unique_lock<std::mutex> lock(st->mtx);

    auto it = st->staged.find(id);
    if (it == st->staged.end() || it->second.released) {
        return;
    }

    DSK_DBG("entry %s left the RAM cache before it was written, not writing it\n", hex64(id).c_str());

    st->st.n_write_skipped++;
    st->st.n_bytes_skipped += it->second.job.n_bytes;
    st->staged_bytes       -= it->second.job.n_bytes;
    st->staged.erase(it);

    lock.unlock();
    st->cv_done.notify_all();
}

void server_cache_disk::release(const server_prompt_cache_state & state) {
    llama_tokens tokens;
    uint64_t id = 0;
    if (!disk_entry_id(state, tokens, id)) {
        return;
    }

    std::unique_lock<std::mutex> lock(st->mtx);

    if (id == st->cur_id) {
        st->cur_released = true;
        return;
    }

    const bool needed = st->staged.count(id) || !st->index.count(id) || st->is_removing_locked(id);

    // the memory of the released entries is not counted by the RAM cache: it is bounded to one entry beyond --cache-ram
    // by waiting for the writer before releasing another one; this blocks the server loop, but only when the RAM cache
    // drops entries faster than the disk takes them
    const auto n_released = [&] {
        size_t n = st->cur_released ? 1 : 0;
        for (const auto & [i, w] : st->staged) {
            n += w.released ? 1 : 0;
        }
        return n;
    };

    if (needed && n_released() > 0 && !st->stopping) {
        const int64_t t_start = ggml_time_ms();

        st->st.n_backlog_waits++;
        st->cv_jobs.notify_all();

        const bool ok = st->cv_done.wait_for(lock, std::chrono::milliseconds(st->t_wait_max_ms), [&] {
            return n_released() == 0 || st->writer_exited || st->stopping;
        });

        st->st.t_backlog_wait_ms += ggml_time_ms() - t_start;

        if (!ok) {
            DSK_ERR("the disk writer did not keep up for %.1f s, dropping entry %s instead of holding its memory\n",
                    st->t_wait_max_ms / 1000.0, hex64(id).c_str());
            if (auto it = st->staged.find(id); it != st->staged.end()) {
                st->staged_bytes -= it->second.job.n_bytes;
                st->staged.erase(it);
            }
            st->st.n_write_dropped++;
            return;
        }
    }

    stage_locked(state, true);

    lock.unlock();
    st->cv_jobs.notify_all();
}

void server_cache_disk::discard(uint64_t id) {
    std::unique_lock<std::mutex> lock(st->mtx);

    st->loaded.erase(id);
    if (st->loading.count(id)) {
        st->discarded.insert(id);
    }
}

uint64_t server_cache_disk::flush() {
    uint64_t ticket;
    {
        std::unique_lock<std::mutex> lock(st->mtx);

        ticket = ++st->flush_seq;
        for (auto & [id, w] : st->staged) {
            if (w.flush == 0) {
                w.flush = ticket;
            }
        }
        if (st->cur_id != 0 && st->cur_flush == 0) {
            st->cur_flush = ticket;
        }
    }
    st->cv_jobs.notify_all();

    return ticket;
}

std::vector<server_cache_disk::entry_ptr> server_cache_disk::entries() const {
    std::unique_lock<std::mutex> lock(st->mtx);

    std::vector<entry_ptr> res;
    res.reserve(st->index.size());
    for (const auto & [id, r] : st->index) {
        if (!r.removing && !st->quarantined.count(id)) {
            res.push_back(r.e);
        }
    }
    return res;
}

bool server_cache_disk::load_async(uint64_t id) {
    std::unique_lock<std::mutex> lock(st->mtx);

    if (st->loading.count(id)) {
        return true;
    }

    if (st->stopping || !st->index.count(id) || st->is_removing_locked(id) || st->quarantined.count(id)) {
        return false;
    }

    st->loading.insert(id);
    st->load_queue.push_back(id);
    st->loaded.erase(id);
    st->discarded.erase(id);

    lock.unlock();
    st->cv_jobs.notify_all();

    return true;
}

server_cache_disk::load_result server_cache_disk::take_loaded(uint64_t id, server_prompt_cache_state & out, bool has_mtmd) {
    std::shared_ptr<loaded_entry> res;
    {
        std::unique_lock<std::mutex> lock(st->mtx);
        auto it = st->loaded.find(id);
        if (it == st->loaded.end()) {
            return st->loading.count(id) ? LOAD_PENDING : LOAD_FAILED;
        }
        res = std::move(it->second);
        st->loaded.erase(it);
    }

    if (!res || !res->ok) {
        return LOAD_FAILED;
    }

    out.prompt.tokens      = server_tokens(res->tokens, has_mtmd);
    out.prompt.n_protected = res->n_protected;
    out.prompt.checkpoints.clear();
    for (auto & c : res->ckpts) {
        out.prompt.checkpoints.push_back(std::move(c));
    }
    out.pos_min = res->pos_min;
    out.data    = std::move(res->data);
    out.id      = id;

    return LOAD_OK;
}

void server_cache_disk::touch(uint64_t id) {
    std::unique_lock<std::mutex> lock(st->mtx);

    if (st->stopping || !st->index.count(id)) {
        return;
    }

    st->index_touch(id, time_now_ns());
    st->ops.push_back({ file_op::TOUCH, id });

    lock.unlock();
    st->cv_jobs.notify_all();
}

void server_cache_disk::quarantine(uint64_t id) {
    std::unique_lock<std::mutex> lock(st->mtx);

    if (st->quarantined.insert(id).second) {
        DSK_WRN("entry %s failed to restore, not offering it again (its file is kept)\n", hex64(id).c_str());
        st->st.n_quarantined++;
    }
}

void server_cache_disk::set_live(std::vector<llama_tokens> conversations) {
    std::unique_lock<std::mutex> lock(st->mtx);
    st->live = std::move(conversations);
}

bool server_cache_disk::wait(uint64_t ticket, int64_t timeout_ms) const {
    const auto flushed = [&] {
        const auto wanted = [&](uint64_t flush) { return flush != 0 && flush <= ticket; };
        if (st->cur_id != 0 && wanted(st->cur_flush)) {
            return false;
        }
        return std::none_of(st->staged.begin(), st->staged.end(), [&](const auto & w) { return wanted(w.second.flush); });
    };

    std::unique_lock<std::mutex> lock(st->mtx);
    st->cv_done.wait_for(lock, std::chrono::milliseconds(std::max<int64_t>(0, timeout_ms)), [&] {
        return flushed() || st->writer_exited;
    });
    return flushed();
}

void server_cache_disk::stop(int64_t timeout_ms) {
    if (!st) {
        return;
    }

    {
        std::unique_lock<std::mutex> lock(st->mtx);
        if (!st->writer.joinable() && !st->reader.joinable()) {
            return;
        }
        st->stopping = true;
    }
    st->cv_jobs.notify_all();

    // a load callback in progress finishes before the server goes away
    {
        std::unique_lock<std::mutex> lock(st->mtx_callback);
    }

    const auto exited = [&] { return st->writer_exited && st->reader_exited; };

    bool ok;
    {
        std::unique_lock<std::mutex> lock(st->mtx);
        ok = st->cv_done.wait_for(lock, std::chrono::milliseconds(std::max<int64_t>(0, timeout_ms)), exited);
        if (!ok && !st->staged.empty()) {
            DSK_WRN("abandoning %zu queued writes (%.3f MiB)\n", st->staged.size(), st->staged_bytes / (1024.0 * 1024.0));
        }
    }

    if (!ok) {
        // abandon the remaining writes, a write in progress stops at the next chunk and removes its temporary file
        st->abort.store(true);
        st->cv_jobs.notify_all();

        std::unique_lock<std::mutex> lock(st->mtx);
        ok = st->cv_done.wait_for(lock, std::chrono::seconds(5), exited);
    }

    if (ok) {
        st->writer.join();
        st->reader.join();
    } else {
        // stuck in the kernel: let the process exit without it, the threads keep the state alive
        DSK_ERR("%s", "the disk cache threads did not stop, detaching them\n");
        st->writer.detach();
        st->reader.detach();
    }
}

server_cache_disk_stats server_cache_disk::stats() const {
    std::unique_lock<std::mutex> lock(st->mtx);

    server_cache_disk_stats res = st->st;
    res.n_entries       = st->index.size();
    res.n_entries_other = st->others.size();
    res.n_bytes_other   = st->n_bytes_others;
    res.n_bytes         = st->n_bytes;
    res.n_bytes_max     = st->n_bytes_max;
    res.n_backlog       = st->staged.size() + (st->cur_id != 0 ? 1 : 0);
    res.n_backlog_bytes = st->staged_bytes + st->cur_bytes;
    res.n_released       = st->cur_released ? 1 : 0;
    res.n_released_bytes = st->cur_released ? st->cur_bytes : 0;
    for (const auto & [id, w] : st->staged) {
        if (w.released) {
            res.n_released++;
            res.n_released_bytes += w.job.n_bytes;
        }
    }
    res.n_loaded = st->loaded.size();
    res.t_delay_ms      = st->t_delay_ms;
    res.n_loading       = st->loading.size();
    return res;
}
