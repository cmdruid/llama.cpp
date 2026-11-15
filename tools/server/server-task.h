#pragma once

#include "common.h"
#include "llama.h"

#include <cstdlib>
#include <string>
#include <unordered_set>
#include <list>
#include <memory>
#include <map>

// TODO: prevent including the whole server-common.h as we only use server_tokens
#include "server-common.h"
#include "mtmd-helper.h"


enum server_task_type {
    SERVER_TASK_TYPE_COMPLETION,
    SERVER_TASK_TYPE_EMBEDDING,
    SERVER_TASK_TYPE_RERANK,
    SERVER_TASK_TYPE_DECISION,
    SERVER_TASK_TYPE_INFILL,
    SERVER_TASK_TYPE_TRANSCRIBE_RT,
    SERVER_TASK_TYPE_REALTIME, // a conversation over /v1/realtime, see server-realtime.cpp
    SERVER_TASK_TYPE_CANCEL,
    SERVER_TASK_TYPE_CONTROL,
    SERVER_TASK_TYPE_NEXT_RESPONSE,
    SERVER_TASK_TYPE_METRICS,
    SERVER_TASK_TYPE_SLOT_GET,
    SERVER_TASK_TYPE_SLOT_SAVE,
    SERVER_TASK_TYPE_SLOT_RESTORE,
    SERVER_TASK_TYPE_SLOT_ERASE,
    SERVER_TASK_TYPE_GET_LORA,
    SERVER_TASK_TYPE_SET_LORA,
    SERVER_TASK_TYPE_TTS,
    SERVER_TASK_TYPE_CACHE_LOADED, // a disk cache entry was read, see server-cache-disk.h
    SERVER_TASK_TYPE_CACHE_STATUS,
};

// TODO: change this to more generic "response_format" to replace the "format_response_*" in server-common
enum task_response_type {
    TASK_RESPONSE_TYPE_NONE, // llama.cpp native format
    TASK_RESPONSE_TYPE_OAI_CHAT,
    TASK_RESPONSE_TYPE_OAI_CMPL,
    TASK_RESPONSE_TYPE_OAI_RESP,
    TASK_RESPONSE_TYPE_OAI_ASR, // transcriptions API
    TASK_RESPONSE_TYPE_OAI_EMBD,
    TASK_RESPONSE_TYPE_ANTHROPIC,
};

enum stop_type {
    STOP_TYPE_NONE,
    STOP_TYPE_EOS,
    STOP_TYPE_WORD,
    STOP_TYPE_LIMIT,
};

enum server_control_action {
    SERVER_CONTROL_ACTION_NONE,
    SERVER_CONTROL_ACTION_REASONING_END,
    SERVER_CONTROL_ACTION_AUDIO_APPEND, // push control_audio into the slot's audio stream
    SERVER_CONTROL_ACTION_AUDIO_COMMIT, // flush the audio stream, more audio can follow
    SERVER_CONTROL_ACTION_AUDIO_END,    // flush the audio stream and end it
    // realtime conversations (SERVER_TASK_TYPE_REALTIME), applied in order with the audio actions
    SERVER_CONTROL_ACTION_TEXT_APPEND,     // append control_text to the conversation
    SERVER_CONTROL_ACTION_RESPONSE_CREATE, // generate after control_text, see control_response
    SERVER_CONTROL_ACTION_RESPONSE_CANCEL, // stop the response being generated
    SERVER_CONTROL_ACTION_CLEAR,           // drop the conversation
};

// a response of a realtime conversation
struct server_control_response {
    mtmd_helper_gen_audio_out_mode out_mode  = MTMD_HELPER_GEN_AUDIO_OUT_AUTO;
    common_params_sampling         sampling;       // of the text tokens: tool call grammar and preserved tokens
    int32_t                        n_predict = -1; // text tokens and audio frames, -1: until the context is full
    std::string                    lang;           // empty: model default
    std::string                    voice;          // empty: model default
};

struct task_params {
    bool stream          = false;
    bool include_usage   = false;
    bool cache_prompt    = true; // remember the prompt to avoid reprocessing all prompt
    bool return_tokens   = false;
    bool return_progress = false;

    int32_t sse_ping_interval = 30; // seconds between SSE comment pings while the stream stays silent, -1 disables

    int32_t n_keep    =  0; // number of tokens to keep from initial prompt
    int32_t n_discard =  0; // number of tokens after n_keep that may be discarded when shifting context, 0 defaults to half
    int32_t n_predict = -1; // new tokens to predict
    int32_t n_indent  =  0; // minimum line indentation for the generated text in number of whitespace characters
    int32_t n_cmpl    =  1; // number of completions to generate from this prompt

    int32_t n_cache_reuse = 0; // min chunk size to attempt reusing from the cache via KV shifting (0 = disabled)

    int64_t t_max_prompt_ms  = -1; // TODO: implement
    int64_t t_max_predict_ms = -1; // if positive, limit the generation phase to this time limit

    std::map<int, float> lora; // mapping adapter ID -> scale

    std::vector<std::string> antiprompt;
    std::vector<std::string> response_fields;

    bool timings_per_token   = false;
    bool post_sampling_probs = false;

    struct common_params_sampling sampling;
    struct common_params_speculative speculative;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;

    // realtime control (SERVER_TASK_TYPE_CONTROL)
    server_control_action control_action = SERVER_CONTROL_ACTION_NONE;
    std::string           control_cmpl_id;
    std::vector<float>    control_audio; // PCM F32 samples for SERVER_CONTROL_ACTION_AUDIO_APPEND
    std::string           control_text;  // raw text, special tokens are parsed
    server_control_response control_response;

    // reported in generation_settings, parsing itself is owned by the chat session
    common_chat_format      chat_format      = COMMON_CHAT_FORMAT_CONTENT_ONLY;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_NONE;

    // message spans for checkpointing
    common_chat_msg_spans message_spans;

    // Embeddings
    int32_t embd_normalize = 2; // (-1=none, 0=max absolute int16, 1=taxicab, 2=Euclidean/L2, >2=p-norm)

    json format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const;
    json to_json(bool only_metrics = false) const;
};

// struct for tracking the state of a task (e.g., for streaming)
struct task_result_state {
    // tracking diffs for partial tool calls
    std::vector<common_chat_msg_diff> diffs;
    common_chat_session chat_session; // owns all parsing for this generation
    common_chat_msg chat_msg;
    std::vector<std::string> generated_tool_call_ids;
    std::unordered_set<size_t> sent_tool_call_names;

    // for OpenAI Responses and Anthropic streaming API:
    // track output item / content block state across chunks
    bool thinking_block_started = false;
    bool text_block_started = false;

    // for OpenAI Responses streaming API
    bool oai_resp_created = false;
    const std::string oai_resp_id;
    const std::string oai_resp_reasoning_id;
    const std::string oai_resp_message_id;
    std::string oai_resp_fc_id; // function call ID for current args delta

    task_result_state(common_chat_session session = {});

    // parse partial tool calls and update the internal state
    common_chat_msg update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls = false);
};

struct server_task {
    int id = -1; // to be filled by server_queue

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // used when there are multiple prompts (batch request)

    // used by SERVER_TASK_TYPE_CANCEL
    int id_target = -1;
    int id_slot   = -1;

    // used by parallel sampling (multiple completions from same prompt)
    int id_parent  = -1;
    // temporary store of child tasks for scheduling
    // note: accessing to elements is invalid after the task is moved to server_slot
    std::vector<server_task> child_tasks;
    // if set on a parent, the children have their own prompt and only share its first n_tokens_shared tokens
    int32_t n_tokens_shared = 0;

    // used by SERVER_TASK_TYPE_INFERENCE
    task_params   params;
    server_tokens tokens;

    // only used by CLI, this allow tokenizing CLI inputs on server side
    // we need this because mtmd_context and vocab are not accessible outside of server_context
    bool                    cli = false;
    std::string             cli_prompt;
    std::vector<raw_buffer> cli_files;

    server_task_type type;

    // used by SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE
    struct slot_action {
        int id_slot;
        std::string filename;
        std::string filepath;
    };
    slot_action slot_action;

    // used by SERVER_TASK_TYPE_METRICS
    bool metrics_reset_bucket = false;

    // used by SERVER_TASK_TYPE_DECISION
    // where to read the model output of each option, exactly one of the two lists is used
    struct decision {
        std::vector<llama_token> labels;       // logits of these tokens, at the last prompt token
        std::vector<int32_t>     label_groups; // if set, number of labels per output, the output is their max
        std::vector<int32_t>     markers;      // embeddings[column] at these prompt positions
        int32_t                  column = 0;
        // if set, embeddings is [q | k], and the output is instead the scaled dot product of q[pointer] and k[marker]
        int32_t                  pointer = -1;
        // if set, the prompt must be the only sequence of its batch
        bool                     no_batching = false;

        // if set, the outputs are the pooled embeddings of the sequence, one score per class
        bool                     pooled = false;

        // first prompt position that is read, -1 if none
        int32_t pos_first() const {
            int32_t pos = pointer;
            for (const int32_t marker : markers) {
                pos = pos < 0 ? marker : std::min(pos, marker);
            }
            for (size_t i = 0; i < order.size(); i++) {
                if (order[i] != 0) {
                    return pos < 0 ? (int32_t) i : std::min(pos, (int32_t) i);
                }
            }
            return pos;
        }

        // for a joint head: one value per prompt token, see llama_batch_ext_set_decision_order()
        // the scores are the first n_scores rows of the embeddings
        std::vector<int32_t> order;
        int32_t              n_scores = 0;
    };
    decision decision;

    // used by SERVER_TASK_TYPE_SET_LORA
    std::map<int, float> set_lora; // mapping adapter ID -> scale

    // used by SERVER_TASK_TYPE_TTS
    mtmd_helper::gen_audio::inp tts_inp;

    server_task() = default;

    server_task(server_task_type type) : type(type) {}

    int32_t n_tokens() const {
        return tokens.size();
    }

    bool need_embd() const {
        switch (type) {
            case SERVER_TASK_TYPE_EMBEDDING:
            case SERVER_TASK_TYPE_RERANK:
                return true;
            case SERVER_TASK_TYPE_DECISION:
                return !decision.markers.empty() || !decision.order.empty() || decision.pooled;
            default:
                return false;
        }
    }

    bool need_logits() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_TRANSCRIBE_RT:
                return true;
            case SERVER_TASK_TYPE_DECISION:
                return !decision.labels.empty();
            default:
                return false;
        }
    }

    bool need_sampling() const {
        switch (type) {
            case SERVER_TASK_TYPE_COMPLETION:
            case SERVER_TASK_TYPE_INFILL:
            case SERVER_TASK_TYPE_TTS:
            case SERVER_TASK_TYPE_TRANSCRIBE_RT:
                return true;
            default:
                return false;
        }
    }

    // utility function
    static std::unordered_set<int> get_list_id(const std::vector<server_task> & tasks) {
        std::unordered_set<int> ids(tasks.size());
        for (size_t i = 0; i < tasks.size(); i++) {
            ids.insert(tasks[i].id);
            for (auto & child : tasks[i].child_tasks) {
                ids.insert(child.id);
            }
        }
        return ids;
    }

    void apply_chat_session(const common_chat_session & session) {
        if (!session.has_template()) {
            return;
        }

        session.apply_sampling(params.sampling);
        params.chat_format = session.format();
        params.antiprompt.insert(params.antiprompt.end(), session.additional_stops().begin(), session.additional_stops().end());
        params.message_spans = tokens.find_message_spans(session.message_delimiters());
    }

    void add_child(int id_parent, int id_child) {
        server_task copy;

        copy.id        = id_child;
        copy.id_parent = id_parent;
        copy.params    = params;
        copy.type      = type;
        copy.tokens    = tokens.clone();
        copy.id_slot   = -1; // child tasks cannot specify slot

        // use different sampling seed for each child
        // note: https://github.com/ggml-org/llama.cpp/pull/18700#discussion_r2675115723
        if (copy.params.sampling.seed != LLAMA_DEFAULT_SEED) {
            copy.params.sampling.seed += (uint32_t)child_tasks.size() + 1;
        }

        child_tasks.push_back(std::move(copy));
    }

    bool is_parent() const {
        return child_tasks.size() > 0;
    }

    bool is_child() const {
        return id_parent != -1;
    }

    // embedding and rerank tasks always process their whole prompt, their state is never reused
    bool is_stateless() const {
        return type == SERVER_TASK_TYPE_EMBEDDING || type == SERVER_TASK_TYPE_RERANK;
    }

    // the part of the prompt that a later request must share to continue this conversation:
    // up to the end of the last message that is not the assistant's (user, tool or system),
    // or the whole prompt when it has no message spans
    // the assistant turns after it may be rendered differently when the client sends them back
    int32_t n_tokens_protected() const {
        const auto & spans = params.message_spans.spans;
        for (auto it = spans.rbegin(); it != spans.rend(); ++it) {
            if (it->valid() && it->role != COMMON_CHAT_ROLE_ASSISTANT) {
                return std::min<int32_t>(it->pos + it->len, tokens.size());
            }
        }
        return tokens.size();
    }
};

struct result_prompt_progress {
    int32_t total = 0;
    int32_t cache = 0;
    int32_t processed = 0;
    int64_t time_ms = 0;

    json to_json() const;
};

struct server_task_result {
    int id           = -1;
    int id_slot      = -1;

    // TODO @ngxson : remove this field and implement a mapping task_id -> idx in the response_reader
    size_t index = 0; // to be used for batched tasks

    virtual bool is_error() {
        // only used by server_task_result_error
        return false;
    }
    virtual bool is_stop() {
        // only used by server_task_result_cmpl_*
        return true;
    }
    virtual void update(task_result_state &) {
        // only used by server_task_result_cmpl_*
    }
    virtual json to_json() = 0;
    virtual ~server_task_result() = default;
    virtual server_task_result * clone() const {
        GGML_ABORT("not implemented for this task type");
    }
};

// using shared_ptr for polymorphism of server_task_result
using server_task_result_ptr = std::unique_ptr<server_task_result>;

struct completion_token_output {
    llama_token tok;
    float prob;
    std::string text_to_send;
    struct prob_info {
        llama_token tok;
        std::string txt;
        float prob;
    };
    std::vector<prob_info> probs;

    json to_json(bool post_sampling_probs) const;

    static json probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs);

    static float logarithm(float x);

    static std::vector<unsigned char> str_to_bytes(const std::string & str);

};

struct server_task_result_cmpl_final : server_task_result {
    json reasoning_transition;
    common_chat_input content;
    llama_tokens tokens;

    bool stream;
    bool include_usage;
    server_slot_stats stats;
    std::string prompt;

    bool truncated;
    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;
    int32_t n_tokens_cached;
    bool has_new_line;
    std::string stopping_word;
    stop_type stop = STOP_TYPE_NONE;

    bool post_sampling_probs;
    std::vector<completion_token_output> probs_output;
    std::vector<std::string>  response_fields;

    task_params generation_params;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    common_chat_msg    oaicompat_msg; // to be populated by update()

    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // for OpenAI Responses API
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;

    virtual bool is_stop() override {
        return true; // in stream mode, final responses are considered stop
    }

    virtual json to_json() override;

    virtual void update(task_result_state & state) override {
        is_updated = true;
        oaicompat_msg = state.update_chat_msg(content, false, oaicompat_msg_diffs);

        oai_resp_id = state.oai_resp_id;
        oai_resp_reasoning_id = state.oai_resp_reasoning_id;
        oai_resp_message_id = state.oai_resp_message_id;
    }

    json to_json_non_oaicompat();

    json usage_json_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_chat_stream();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_resp_stream();

    json to_json_oaicompat_asr();

    json to_json_anthropic();

    json to_json_anthropic_stream();
};

struct server_task_result_cmpl_partial : server_task_result {
    common_chat_input content;
    llama_tokens      tokens;

    int32_t n_decoded;
    int32_t n_prompt_tokens;
    int32_t n_prompt_tokens_cache;

    bool post_sampling_probs;
    bool is_progress = false;
    bool is_begin = false; // whether to send 200 status to HTTP client (begin of SSE stream)
                           // ref: https://github.com/ggml-org/llama.cpp/pull/23884
    bool audio_commit_drained = false; // realtime: non-final commit has fully decoded
    completion_token_output prob_output;
    server_slot_stats stats;
    result_prompt_progress progress;

    // response formatting
    bool               verbose  = false;
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;
    std::string        oaicompat_model;
    std::string        oaicompat_cmpl_id;
    std::vector<common_chat_msg_diff> oaicompat_msg_diffs; // to be populated by update()
    bool is_updated = false;

    // Streaming state copied from task_result_state for this chunk
    bool thinking_block_started = false;
    bool text_block_started     = false;

    // for OpenAI Responses API
    bool oai_resp_created = false;
    std::string oai_resp_id;
    std::string oai_resp_reasoning_id;
    std::string oai_resp_message_id;
    std::string oai_resp_fc_id;

    // for Anthropic API: track if any reasoning content has been generated
    bool anthropic_has_reasoning = false;

    virtual bool is_stop() override {
        return false; // in stream mode, partial responses are not considered stop
    }

    virtual void update(task_result_state & state) override;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();

    json to_json_oaicompat_chat();

    json to_json_oaicompat_resp();

    json to_json_oaicompat_asr();

    json to_json_anthropic();
};

struct server_task_result_embd : server_task_result {
    std::vector<std::vector<float>> embedding;

    int32_t n_tokens;

    // response formatting
    task_response_type res_type = TASK_RESPONSE_TYPE_NONE;

    virtual json to_json() override;

    json to_json_non_oaicompat();

    json to_json_oaicompat();
};

// one event of a realtime conversation: text and audio of the response being generated, or its end
struct server_task_result_realtime : server_task_result {
    common_chat_input  content;             // the text with its token
    std::vector<float> audio;               // PCM F32 at the sample rate of mtmd_gen_audio_get_info()
    bool               response_done = false;
    bool               cancelled     = false;
    int32_t            n_decoded     = 0;

    virtual bool is_stop() override {
        return false; // the session ends when the client leaves, the task is cancelled then
    }

    virtual json to_json() override;
};

struct server_task_result_tts : server_task_result {
    std::string audio; // raw bytes for this chunk (WAV or PCM, per request's out_type)
    int32_t     sample_rate = 0;
    bool        final = false; // true for the last chunk of a request

    virtual bool is_stop() override { return final; }

    virtual json to_json() override;
};

struct server_task_result_rerank : server_task_result {
    float score = -1e6;

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_decision : server_task_result {
    std::vector<float> scores; // one raw model output per option

    int32_t n_tokens;

    virtual json to_json() override;
};

struct server_task_result_error : server_task_result {
    error_type err_type = ERROR_TYPE_SERVER;
    std::string err_msg;

    // for ERROR_TYPE_EXCEED_CONTEXT_SIZE
    int32_t n_prompt_tokens = 0;
    int32_t n_ctx           = 0;

    virtual bool is_error() override {
        return true;
    }

    virtual json to_json() override;
};

// used by /metrics API
struct server_task_result_metrics : server_task_result {
    // these are immediate stats, not accumulated (server_metrics is cumulative)
    int n_processing_slots = 0;
    int n_tasks_deferred = 0;

    server_metrics metrics;

    virtual json to_json() override;

    struct metric_item {
        std::string name;
        std::string description;
        double value; // prometheus values are always float64
    };
    std::string to_metrics();
};

// used by /slots API
struct server_task_result_slots : server_task_result {
    int n_idle_slots = 0;

    // while we can also use std::vector<server_slot> this requires copying the slot object which can be quite messy
    // therefore, we use json to temporarily store the slot.to_json() result
    json slots_data = json::array();

    virtual json to_json() override;
};

struct server_task_result_slot_save_load : server_task_result {
    std::string filename;
    bool is_save; // true = save, false = load

    size_t n_tokens;
    size_t n_bytes;
    double t_ms;

    virtual json to_json() override;
};

struct server_task_result_cache : server_task_result {
    json data;

    virtual json to_json() override {
        return data;
    }
};

struct server_task_result_slot_erase : server_task_result {
    size_t n_erased;

    virtual json to_json() override;
};

struct server_task_result_control : server_task_result {
    bool        success = false;
    std::string message; // optional detail when success is false

    virtual json to_json() override {
        json out = json { { "success", success } };
        if (!message.empty()) {
            out["message"] = message;
        }
        return out;
    }
};

struct server_task_result_get_lora : server_task_result {
    struct lora {
        common_adapter_lora_info info;
        std::string  alora_invocation_string;
        llama_tokens alora_invocation_tokens;
    };
    std::vector<lora> loras;

    virtual json to_json() override;
};

struct server_task_result_apply_lora : server_task_result {
    virtual json to_json() override;
};

struct server_prompt {
    server_tokens tokens;

    // a checkpoint does not change after it is created, so the prompt cache shares it with the slot instead of a copy
    std::list<std::shared_ptr<const common_prompt_checkpoint>> checkpoints;

    // the prefix that a request must share to continue this conversation, 0 = all tokens
    // see server_task::n_tokens_protected()
    int32_t n_protected = 0;

    // the time (ggml_time_ms()) of the oldest progress of this conversation that is not on disk, 0 = none or unknown
    // set when a slot starts a task, inherited from the cache entries it continues, see server_cache_disk::drop();
    // the disk tier writes an entry once this is --cache-dir-delay old
    int64_t t_unwritten_ms = 0;

    void clear() {
        tokens.clear();
        checkpoints.clear();
        n_protected = 0;
        // t_unwritten_ms is kept: a slot that clears its prompt to process a task keeps the age of its progress
    }

    int n_tokens() const {
        return tokens.size();
    }

    int32_t n_tokens_protected() const {
        return n_protected > 0 ? std::min(n_protected, n_tokens()) : n_tokens();
    }

    // true if a request with these tokens continues this conversation: it keeps the whole protected prefix
    // a request that does not continue it would discard part of the conversation
    bool is_continued_by(const server_tokens & tokens_new) const {
        return (int32_t) tokens.get_common_prefix(tokens_new) >= n_tokens_protected();
    }

    server_prompt clone() const {
        // a copy is a branch: its progress starts when it is used
        return server_prompt {
            tokens.clone(),
            checkpoints,
            n_protected,
            0,
        };
    }
};

struct server_prompt_data {
    common_state_data main;
    common_state_data drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_cache_disk;
struct server_cache_disk_entry;

struct server_prompt_cache_state {
    server_prompt prompt;

    // not modified once the entry is complete, so the disk writer can share it
    std::shared_ptr<server_prompt_data> data;

    // the lowest position held by the saved state of the sequence, see llama_memory_seq_pos_min()
    llama_pos pos_min = 0;

    // hash of the tokens, the name of the entry in the disk cache
    uint64_t id = 0;

    size_t size() const {
        size_t res = data ? data->size() : 0;

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt->size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens, int32_t n_swa) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
        this->n_swa        = n_swa;

#ifdef LLAMA_SERVER_TEST_HOOKS
        // test builds only: every restore fails, as for a state that does not fit
        this->test_restore_fail = getenv("LLAMA_SERVER_TEST_CACHE_RESTORE_FAIL") != nullptr;
#endif
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    // the SWA window of the context, 0 = none (see server_context_impl::n_swa)
    int32_t n_swa = 0;

#ifdef LLAMA_SERVER_TEST_HOOKS
    bool test_restore_fail = false;
#endif

    size_t size() const;

    size_t n_tokens() const;

    // the number of tokens of `prompt` that a request with `tokens_new` keeps once the state of `prompt` is restored,
    // pos_min is the lowest position held by that state:
    // the common prefix if the state can be truncated there, otherwise (SWA, recurrent or hybrid memory) the last
    // context checkpoint before the point where they diverge, as decided when the prompt is processed
    int32_t n_usable(const server_prompt & prompt, llama_pos pos_min, const server_tokens & tokens_new) const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    // n_usable() if the entry is a candidate for the request, 0 otherwise:
    // restoring a large state to reuse a small part of it costs more than it saves, unless the request continues it
    int32_t n_candidate(const server_prompt & prompt, llama_pos pos_min, const server_tokens & tokens_new) const;

    // restore the entry that the request reuses most, if it reuses more than `prompt` (the slot) already holds
    // returns false if the state could not be restored, id_failed is then the entry that failed
    bool load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    void update();

    //
    // disk tier, see server-cache-disk.h
    //

    server_cache_disk * disk = nullptr;

    uint64_t id_failed = 0;

    struct stats_t {
        uint64_t n_saves      = 0; // entries added by saving a slot
        uint64_t n_hits_ram   = 0; // entries restored into a slot
        uint64_t n_hits_disk  = 0; // entries read from disk into RAM for a request
        uint64_t n_misses     = 0; // lookups where neither the slot nor an entry had anything for the request
        uint64_t n_evictions  = 0; // entries dropped from RAM to stay within the limits (they remain on disk)
        uint64_t n_failed     = 0; // restores that failed
        double   t_save_ms    = 0.0;
        double   t_load_ms    = 0.0;
    } stats;

    // the entry was completed by saving a slot: the disk tier writes it after its delay
    void on_saved(const server_prompt_cache_state & state);

    // drop the oldest entry to stay within the limits, the disk tier writes it first
    void evict_oldest();

    // above this share of limit_size, the least recently used entries are written to disk ahead of their delay, so
    // that an eviction finds them there instead of waiting for the writer, see write_ahead()
    static constexpr double WRITE_AHEAD_MARK = 0.8;

    // above the mark: ask the disk tier to write the next eviction victims now (the two least recently used entries,
    // and more if evicting them would not bring the cache back to the mark)
    void write_ahead() const;

    // the disk entry the request reuses most, if it reuses more than n_floor tokens and is not in RAM already
    std::shared_ptr<const server_cache_disk_entry> find_disk(const server_tokens & tokens_new, int32_t n_floor) const;

    // the most tokens a RAM entry offers to the request
    int32_t n_best(const server_tokens & tokens_new) const;

    // add an entry read from disk, making room for it; false if it does not fit
    bool insert(server_prompt_cache_state && state);

    // drop an entry, e.g. one that failed to restore; the disk tier writes it first
    void erase(uint64_t id);

    bool contains(uint64_t id) const;
};

// used exclusively by router mode
struct server_task_result_router : server_task_result {
    json data;
    virtual json to_json() override { return data; }
    virtual server_task_result * clone() const override {
        return new server_task_result_router(*this);
    }
};
