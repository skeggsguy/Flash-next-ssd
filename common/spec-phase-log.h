#pragma once

// The apprentice's round, phase by phase (the study's C1-PROFILE-PLAN.md): `<GGML_METAL_CBLOG>.phases`, written
// beside the Metal timeline so a joiner can say what the engine was doing while the GPU sat idle.
//
// Off unless GGML_METAL_CBLOG=<path> is set; GGML_METAL_CBLOG_SIDECARS=0 keeps the Metal timeline and turns this
// log (and the book manager's `.stops`) off, for an overhead A/B. When off every hook is one load of a pointer:
// no file, no buffer, no clock call.
//
//   A <uptime_raw_s> <unix_s>                   clock anchor, once (the first line)
//   P <round> <phase> <t_start> <t_end> <arg>   one line per phase, in the order they ended
//
// Times are seconds of CLOCK_UPTIME_RAW, the clock mach_absolute_time, the Metal timeline and Metal's GPU
// start/end use; never CLOCK_MONOTONIC or _RAW, which count sleep on macOS. Records are kept in memory and written
// at exit (or when a test's log object is destroyed).
//
// A round is one turn of writing: the server begins one each time a slot writes (common_spec_phase_round_begin),
// whatever its kind (apprentice, reused draft, or plain), so the round id rises by one per turn; the apprentice
// trace (LLAMA_SPEC_RATE_TRACE) carries the same id on each cycle while this log is on. Phases outside a round
// carry round -1 (the openings); check, process and accept are logged only inside a round, because reading in
// runs the same decode.
//
// arg: bytes for a copy (save_dft, load_dft, save_tgt, restore_tgt), the step index for draft_step, tokens for
// check and process, guesses kept for accept, 0 for the openings.

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

enum common_spec_phase : int32_t {
    COMMON_SPEC_PHASE_OPEN_TGT = 0,
    COMMON_SPEC_PHASE_OPEN_DFT,
    COMMON_SPEC_PHASE_SAVE_DFT,
    COMMON_SPEC_PHASE_DRAFT_STEP,
    COMMON_SPEC_PHASE_LOAD_DFT,
    COMMON_SPEC_PHASE_SAVE_TGT,
    COMMON_SPEC_PHASE_CHECK,
    COMMON_SPEC_PHASE_ACCEPT,
    COMMON_SPEC_PHASE_PROCESS,
    COMMON_SPEC_PHASE_RESTORE_TGT,
    COMMON_SPEC_PHASE_COUNT,
};

const char * common_spec_phase_name(common_spec_phase ph);

// seconds of the timeline's clock as integer nanoseconds (CLOCK_UPTIME_RAW on Apple)
int64_t common_spec_phase_now_ns();
// the wall clock in microseconds, for the anchor
int64_t common_spec_phase_unix_us();

class common_spec_phase_log {
public:
    struct rec {
        int64_t round;
        int64_t t0_ns;
        int64_t t1_ns;
        int64_t arg;
        int32_t phase;
    };

    static constexpr size_t CHUNK = 1 << 14; // records per block: growth never copies what is already logged

    // the process's log: nullptr when off. Resolved once; the first call opens it and arranges the write at exit
    static common_spec_phase_log * get();

    // a log for `<cblog>.phases` if `env_cblog` names a path and `env_sidecars` is not "0"; nullptr otherwise.
    // Nothing is created on disk until write()
    static std::unique_ptr<common_spec_phase_log> open_from_env(const char * env_cblog, const char * env_sidecars);

    explicit common_spec_phase_log(std::string path);
    ~common_spec_phase_log(); // writes, if not written yet

    common_spec_phase_log(const common_spec_phase_log &) = delete;
    common_spec_phase_log & operator=(const common_spec_phase_log &) = delete;

    void put(common_spec_phase ph, int64_t t0_ns, int64_t t1_ns, int64_t arg);

    void    round_begin();     // the next round: the id moves on by one and the round is open
    void    round_end();       // between rounds: round-only phases are not logged
    bool    in_round() const;
    int64_t round() const;     // the round in force, -1 before the first

    size_t n_records() const;
    size_t n_chunks()  const;  // blocks allocated (0 until the first record)
    const std::string & path() const { return out_path; }

    // the anchor and every record, once; true when the file was written
    bool write();

    static std::string format(const rec & r);

private:
    mutable std::mutex mtx;
    std::string        out_path;
    int64_t            anchor_ns;
    int64_t            anchor_unix_us;
    std::vector<std::unique_ptr<rec[]>> chunks;
    size_t             n = 0;
    int64_t            cur_round = -1;
    bool               open_round = false;
    bool               written    = false;
};

// hooks for upstream-owned files: one pointer check each when the log is off
void    common_spec_phase_round_begin();
void    common_spec_phase_round_end();
int64_t common_spec_phase_round(); // -1 when the log is off or before the first round

// times one phase from construction to end() or destruction. round_only: log it only inside a round
struct common_spec_phase_scope {
    common_spec_phase_log * log;
    common_spec_phase       ph;
    int64_t                 arg;
    int64_t                 t0_ns = 0;

    explicit common_spec_phase_scope(common_spec_phase ph, int64_t arg = 0, bool round_only = false);
    ~common_spec_phase_scope() { end(); }

    common_spec_phase_scope(const common_spec_phase_scope &) = delete;
    common_spec_phase_scope & operator=(const common_spec_phase_scope &) = delete;

    void end(); // records now; later calls and the destructor do nothing
};
