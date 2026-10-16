// Desk books first (study #39, step 1): GGML_OP_MOE_GATE on MTL0 with no model. One floor in small: a CPU op
// splits the slip into two id planes (the desk's books, and the books still being fetched: the arrivals), the
// desk link and a shared expert run, then the gate, then the arrivals link written into the desk link's tensor
// (ggml_mul_mat_id_into) and the shared expert added. Before each run the arriving slots hold a stale book; the
// gate fn waits a little (a trip landing) and only then writes the right book into those slots. It checks:
//   - the output uses the books the fn wrote: everything after the gate was sent after the fn returned;
//   - the desk link's rows were already on the GPU's output while the fn ran: the main buffer was sent first
//     (the overlap; not with one command buffer, argv "ncb0", where the fn runs before the only buffer is sent);
//   - the split run's bytes equal a run with every book on the desk before it starts (exact however it falls);
//   - a node after the gate that does not read it still waits for the fn (the reorder does not pull it across);
//   - the gate fn is called once a graph, through the encode-ahead commit when that is on and the plain path when
//     not, and a backend with no fn set computes the gate as a no-op (as does the CPU backend);
// ctest runs it with encode ahead on (one and two extra command buffers), off, and with one command buffer.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <csignal>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    printf("  %-76s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

constexpr int K = 256, N = 64, SLOTS = 16, TOPK = 4, NTOK = 4;

typedef void (*set_gate_t)(ggml_backend_t, void (*)(void *, void *), void *);
typedef void (*counts_t)(int64_t *, int64_t *);

std::vector<float> seeded(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(n);
    for (float & v : f) v = u(rng);
    return f;
}

// one floor stop's slip: each token's TOPK slots, and which slots are still being fetched
struct slip {
    std::vector<int32_t> ids = std::vector<int32_t>(TOPK*NTOK);
    std::vector<char>    arriving = std::vector<char>(SLOTS, 0);
};

// the CPU op (the floor stop): both id planes at once, [TOPK, NTOK, 2]: plane 0 the desk's pairs, plane 1 the
// arrivals'; -1 where the other plane holds the pair. One op, so the floor has one CPU stop (two ops would put
// the second after the gate in the graph: a second stop)
void planes_fn(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith != 0) return;
    const slip * s = (const slip *) userdata;
    const int32_t * src = (const int32_t *) dst->src[0]->data;
    int32_t * out = (int32_t *) dst->data;
    for (int i = 0; i < TOPK*NTOK; i++) {
        const bool arr = s->arriving[src[i]] != 0;
        out[i]               = arr ? -1 : src[i];
        out[TOPK*NTOK + i]   = arr ? src[i] : -1;
    }
}

// the gate fn's state: the books to land, and the desk link's rows to look for while it waits
struct gate_state {
    ggml_tensor *        W = nullptr;
    std::vector<float>   W_true;
    const slip *         s = nullptr;
    ggml_tensor *        desk = nullptr;
    std::vector<float>   desk_want; // [N, TOPK, NTOK]; rows of arriving pairs unused
    bool                 poll = false;
    int                  land_ms = 0;
    int                  calls = 0;
    bool                 desk_seen = false;
    ggml_tensor *        late = nullptr;  // a node after the gate that does not need it: it must still wait
    std::vector<float>   late_want;       // [N, NTOK]
    bool                 poll_late = false; // only on a run with new inputs: its buffer may hold this x's values
    bool                 late_done = false;
    void *               want_ud = nullptr;
    bool                 ud_ok = true;
};

bool close(float got, float want) {
    return std::fabs(got - want) <= 1e-3f*(1.0f + std::fabs(want));
}

bool desk_rows_done(const gate_state * g) {
    const volatile float * d = (const volatile float *) g->desk->data;
    for (int t = 0; t < NTOK; t++) for (int j = 0; j < TOPK; j++) {
        if (g->s->arriving[g->s->ids[t*TOPK + j]]) continue;
        for (int n = 0; n < N; n++) {
            const size_t i = ((size_t) t*TOPK + j)*N + n;
            if (!close(d[i], g->desk_want[i])) return false;
        }
    }
    return true;
}

void gate_fn(void * user_data, void * gate_userdata) {
    gate_state * g = (gate_state *) user_data;
    g->calls++;
    g->ud_ok = g->ud_ok && gate_userdata == g->want_ud;
    if (g->poll) {
        const auto t0 = std::chrono::steady_clock::now();
        while (!(g->desk_seen = desk_rows_done(g)) && std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(1000)) {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(g->land_ms));
    const volatile float * l = (const volatile float *) g->late->data;
    bool done = true;
    for (size_t i = 0; i < g->late_want.size(); i++) {
        done = done && close(l[i], g->late_want[i]);
    }
    g->late_done = g->late_done || (g->poll_late && done);
    const size_t row = (size_t) K*N;
    for (int sl = 0; sl < SLOTS; sl++) {
        if (g->s->arriving[sl]) memcpy((float *) g->W->data + sl*row, g->W_true.data() + sl*row, row*sizeof(float));
    }
}

} // namespace

int main(int argc, char ** argv) {
    const bool ncb0     = argc > 1 && strcmp(argv[1], "ncb0") == 0;
    const bool cpu_view = argc > 1 && strcmp(argv[1], "cpu-view") == 0; // the child must abort, with the message
    if (cpu_view) {
        // before anything touches Metal: the child runs the floor with the gate over the CPU stop's own plane
        int fds[2];
        if (pipe(fds) != 0) return 1;
        const pid_t pid = fork();
        if (pid == 0) {
            close(fds[0]);
            dup2(fds[1], STDERR_FILENO);
            close(fds[1]);
        } else {
            close(fds[1]);
            std::string err;
            char buf[4096];
            for (ssize_t n; (n = read(fds[0], buf, sizeof(buf))) > 0; ) {
                err.append(buf, (size_t) n);
            }
            close(fds[0]);
            int status = 0;
            waitpid(pid, &status, 0);
            check(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT, "a gate over a CPU op's output aborts the run");
            check(err.find("its view is not in a Metal buffer") != std::string::npos, "... saying why, before a kernel reads it");
            printf("%s\n", g_failures == 0 ? "OK" : "FAILED");
            return g_failures == 0 ? 0 : 1;
        }
    }
    const char * ea = getenv("GGML_METAL_ENCODE_AHEAD");
    const bool ahead = ea == nullptr || strcmp(ea, "0") != 0;
    printf("encode ahead %s, %s\n", ahead ? "on" : "off", ncb0 ? "one command buffer" : "GGML_METAL_NCB extra buffers");

    // the CPU backend computes a gate as a no-op, and the op carries its userdata and deps
    {
        ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 64);
        ggml_tensor * d = ggml_scale(ctx, x, 3.0f);
        int tag = 0;
        ggml_tensor * gate = ggml_moe_gate(ctx, x, &d, 1, &tag);
        ggml_tensor * y = ggml_add(ctx, ggml_scale(ctx, gate, 2.0f), d);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        const std::vector<float> xv = seeded(64, 7);
        ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));
        ggml_backend_graph_compute(cpu, gf);
        std::vector<float> yv(64);
        ggml_backend_tensor_get(y, yv.data(), 0, ggml_nbytes(y));
        bool ok = true;
        for (int i = 0; i < 64; i++) ok = ok && yv[i] == xv[i]*2.0f + xv[i]*3.0f;
        check(ok, "CPU: the gate is a no-op view of its input");
        check(ggml_moe_gate_get_userdata(gate) == &tag && gate->view_src == x && gate->src[1] == d && gate->src[2] == nullptr,
                "the gate carries its userdata, its input and its deps");
        ggml_backend_buffer_free(buf);
        ggml_backend_free(cpu);
        ggml_free(ctx);
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (dev == nullptr) {
        printf("no MTL0 device: skipped\n");
        return g_failures == 0 ? 0 : 1;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    auto set_gate     = (set_gate_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_set_moe_gate");
    auto gate_counts  = (counts_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_moe_gate_counts");
    auto ahead_counts = (counts_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_encode_ahead_counts");
    check(set_gate && gate_counts && ahead_counts, "the backend exports the gate fn's setter and the counts");
    if (!set_gate || !gate_counts || !ahead_counts) return 1;
    if (ncb0) {
        // fusion stats encode with one command buffer (ggml_metal_set_n_cb), set before the backend exists
        auto fget  = (void * (*)(ggml_backend_dev_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_get");
        auto finit = (void (*)(void *))               ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_init");
        finit(fget(dev));
    }
    ggml_backend_t mtl = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();

    // the books (the desk's slots) and the shared expert, in a Metal buffer the CPU writes as a runner does
    ggml_init_params wp = { ggml_tensor_overhead()*4, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * W = ggml_new_tensor_3d(wctx, GGML_TYPE_F32, K, N, SLOTS);
    ggml_tensor * S = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, K, N);
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, mtl);
    ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // Metal's shared buffers say is_host false, but their tensors' data is the CPU's address of the memory (the desk
    // is written so); the landed-books check below fails if the fn's writes do not reach the GPU
    const std::vector<float> W_true = seeded((size_t) K*N*SLOTS, 11), W_stale = seeded((size_t) K*N*SLOTS, 12);
    const std::vector<float> Sv = seeded((size_t) K*N, 13);
    ggml_backend_tensor_set(S, Sv.data(), 0, ggml_nbytes(S));

    // the floor: [Metal] cur, [CPU] the two planes, [Metal] desk link, shared expert, gate, arrivals link, sum
    slip s;
    gate_state g;
    // a fresh graph each run, as llama builds one (the scheduler rewires a graph's inputs when it splits it)
    struct floor_graph {
        ggml_context * ctx;
        ggml_tensor * x, * ids, * desk, * gate, * late, * out;
        ggml_cgraph * gf;
    };
    auto build = [&]() {
        floor_graph f;
        ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx = f.ctx = ggml_init(ip);
        f.x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, NTOK);
        ggml_set_input(f.x);
        f.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, TOPK, NTOK);
        ggml_set_input(f.ids);
        ggml_tensor * cur = ggml_scale(ctx, f.x, 1.0f);
        ggml_tensor * args[2] = { f.ids, ggml_view_1d(ctx, cur, 1, 0) }; // the second orders the stop after cur
        ggml_tensor * planes = ggml_custom_4d(ctx, GGML_TYPE_I32, TOPK, NTOK, 2, 1, args, 2, planes_fn, 1, &s);
        ggml_tensor * ids_desk = ggml_view_2d(ctx, planes, TOPK, NTOK, planes->nb[1], 0);
        ggml_tensor * ids_arr  = ggml_view_2d(ctx, planes, TOPK, NTOK, planes->nb[1], planes->nb[2]);
        f.desk = ggml_mul_mat_id_into(ctx, W, ggml_reshape_3d(ctx, cur, K, 1, NTOK), ids_desk, nullptr);
        ggml_tensor * sh = ggml_mul_mat(ctx, S, cur);
        ggml_tensor * deps[2] = { f.desk, sh };
        ggml_tensor * arr;
        if (!cpu_view) {
            f.gate = ggml_moe_gate(ctx, cur, deps, 2, &g);
            arr = ggml_mul_mat_id_into(ctx, W, ggml_reshape_3d(ctx, f.gate, K, 1, NTOK), ids_arr, f.desk);
        } else {
            // gating the CPU stop's own plane: Metal would read the view from CPU memory, so it must refuse
            f.gate = ggml_moe_gate(ctx, ids_arr, deps, 2, &g);
            arr = ggml_mul_mat_id_into(ctx, W, ggml_reshape_3d(ctx, cur, K, 1, NTOK), f.gate, f.desk);
        }
        // built after the gate and reading neither it nor cur (the gate's own range): the reorder would pull it
        // ahead of the gate, into the main buffer, if the gate did not stop it (cur == x: the scale is by 1)
        f.late = ggml_mul_mat(ctx, S, f.x);
        f.out = ggml_add(ctx, arr, ggml_reshape_3d(ctx, ggml_add(ctx, sh, f.late), N, 1, NTOK));
        ggml_set_output(f.out);
        f.gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(f.gf, f.out);
        return f;
    };

    ggml_backend_t backends[2] = { mtl, cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 256, false, true);

    g.W = W; g.W_true = W_true; g.s = &s; g.want_ud = &g;
    int n_splits = 0; // of the last run
    bool on_metal = true;

    std::mt19937 rng(5);
    std::vector<float> xv, want((size_t) N*TOPK*NTOK);
    auto new_slip = [&](int round, bool any_arriving) {
        xv = seeded((size_t) K*NTOK, 100 + round);
        std::fill(s.arriving.begin(), s.arriving.end(), 0);
        for (int t = 0; t < NTOK; t++) {
            std::vector<int32_t> pool(SLOTS);
            for (int i = 0; i < SLOTS; i++) pool[i] = i;
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int j = 0; j < TOPK; j++) s.ids[t*TOPK + j] = pool[j];
        }
        // a third of the slip's books arrive; the first pair is always on the desk, so the desk has rows to show
        for (int i = 1; any_arriving && i < TOPK*NTOK; i += 3) s.arriving[s.ids[i]] = 1;
        s.arriving[s.ids[0]] = 0;
        g.desk_want.assign((size_t) N*TOPK*NTOK, 0.0f);
        g.late_want.assign((size_t) N*NTOK, 0.0f);
        for (int t = 0; t < NTOK; t++) for (int j = 0; j < TOPK; j++) for (int n = 0; n < N; n++) {
            double acc = 0.0, accs = 0.0;
            const float * w = W_true.data() + ((size_t) s.ids[t*TOPK + j]*N + n)*K;
            for (int k = 0; k < K; k++) {
                acc  += (double) w[k]*xv[(size_t) t*K + k];
                accs += (double) Sv[(size_t) n*K + k]*xv[(size_t) t*K + k];
            }
            g.desk_want[((size_t) t*TOPK + j)*N + n] = (float) acc;
            want[((size_t) t*TOPK + j)*N + n] = (float) (acc + 2.0*accs);
            g.late_want[(size_t) t*N + n] = (float) accs;
        }
    };
    // stale: the arriving slots hold another book until the gate fn lands the right one
    auto run = [&](bool stale) {
        std::vector<float> wv = W_true;
        for (int sl = 0; stale && sl < SLOTS; sl++) {
            if (s.arriving[sl]) memcpy(wv.data() + (size_t) sl*K*N, W_stale.data() + (size_t) sl*K*N, (size_t) K*N*sizeof(float));
        }
        ggml_backend_tensor_set(W, wv.data(), 0, ggml_nbytes(W));
        floor_graph f = build();
        ggml_backend_sched_reset(sched);
        ggml_backend_sched_alloc_graph(sched, f.gf);
        ggml_backend_tensor_set(f.x, xv.data(), 0, ggml_nbytes(f.x));
        ggml_backend_tensor_set(f.ids, s.ids.data(), 0, ggml_nbytes(f.ids));
        g.desk = f.desk; g.late = f.late; g.calls = 0; g.desk_seen = false;
        ggml_backend_sched_graph_compute(sched, f.gf);
        std::vector<float> o(ggml_nelements(f.out));
        ggml_backend_tensor_get(f.out, o.data(), 0, ggml_nbytes(f.out));
        n_splits = ggml_backend_sched_get_n_splits(sched);
        on_metal = on_metal && ggml_backend_sched_get_tensor_backend(sched, f.desk) == mtl &&
                ggml_backend_sched_get_tensor_backend(sched, f.gate) == mtl;
        ggml_free(f.ctx);
        return o;
    };
    auto matches = [&](const std::vector<float> & o) {
        for (size_t i = 0; i < o.size(); i++) if (!close(o[i], want[i])) return false;
        return true;
    };
    int64_t c0, k0, e0, u0, c1, k1, e1, u1;

    // no gate fn set: the gate is a no-op and nothing is counted (books all on the desk first)
    gate_counts(&c0, &k0);
    new_slip(0, false);
    const std::vector<float> plain = run(false);
    gate_counts(&c1, &k1);
    check(matches(plain) && g.calls == 0 && c1 == c0, "no gate fn: the gate is a no-op, the right output");

    set_gate(mtl, gate_fn, &g);
    check(on_metal && n_splits == 3, "the floor: Metal, one CPU stop, Metal; the desk link and the gate on Metal");

    bool all_landed = true, all_exact = true, all_once = true, all_seen = true, all_new = true;
    std::vector<float> want_prev = want;
    gate_counts(&c0, &k0);
    ahead_counts(&e0, &u0);
    const int rounds = 12;
    for (int r = 1; r <= rounds; r++) {
        new_slip(r, true);
        all_new = all_new && memcmp(want.data(), want_prev.data(), want.size()*sizeof(float)) != 0;
        want_prev = want;
        g.poll = !ncb0; g.poll_late = true; g.land_ms = 20;
        const std::vector<float> split = run(true);
        all_landed = all_landed && matches(split);
        all_once   = all_once && g.calls == 1;
        all_seen   = all_seen && (ncb0 || g.desk_seen);

        // the same slip with every book on the desk before the run: nothing arrives, byte-equal output
        std::fill(s.arriving.begin(), s.arriving.end(), 0);
        g.poll = false; g.poll_late = false; g.land_ms = 0;
        const std::vector<float> whole = run(false);
        all_exact = all_exact && memcmp(split.data(), whole.data(), split.size()*sizeof(float)) == 0;
        all_once  = all_once && g.calls == 1;
    }
    gate_counts(&c1, &k1);
    ahead_counts(&e1, &u1);
    check(all_landed, "12 floors: the arrivals used the books the gate fn landed");
    check(all_seen, ncb0 ? "one command buffer: no overlap asked for" : "12 floors: the desk link's rows were done while the fn waited");
    check(all_exact, "12 floors: split output byte-equal to every book on the desk");
    check(!g.late_done, "a node after the gate that does not need it still waits for the fn");
    check(all_once && g.ud_ok, "the gate fn was called once a graph, with the gate's userdata");
    check(c1 - c0 == 2*rounds && k1 - k0 == (ncb0 ? 0 : 2*rounds),
            ncb0 ? "one command buffer: every call counted, none as cut" : "every call made with the main buffer already sent");
    check(ahead ? u1 - u0 == 2*rounds : u1 == u0,
            ahead ? "every floor's graph went through the encode-ahead commit" : "the plain path: nothing encoded ahead");
    check(all_new, "each floor's output differs from the one before (a graph that did not run would show)");

    ggml_backend_sched_free(sched);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    ggml_backend_free(cpu);
    ggml_backend_free(mtl);
    printf("%s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
