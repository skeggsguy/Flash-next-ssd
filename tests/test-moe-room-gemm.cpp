// The reading room's expert maths (ggml_mul_mat_id_into): a floor whose books are split between the
// desk and the belt must give the same bytes as one ggml_mul_mat_id over a tensor holding every book.
//
// The reading room (expert streaming for long read-ins) keeps some of a floor's books on the desk (the
// expert cache tensor, one slot per book) and brings the rest in on the belt: one buffer of per-book
// records, gate_up then down, so each weight is a tensor whose nb[2] is the record stride, larger than
// that weight's own bytes, cut into parts. The floor runs one GEMM per source, chained through
// ggml_mul_mat_id_into: a pair whose book is in another source has id -1, so every output row is
// written exactly once, by the kernel an unsplit run would use. Phase B's promise - the words do not
// depend on which books happen to be on the desk - rests on that being exact, so this test compares
// with memcmp, not a tolerance, on every CPU and GPU backend it finds.
//
// Magnitudes are O(1) on purpose. With the generated models' N(0, 0.01) weights the down GEMM's input
// rounds to zero in F16 on Metal and a wrong-book bug compares equal (the e64 fixtures' --unit-scales
// exist for this), so every compared tensor is also checked to be almost all nonzero.

#include "testing.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t N_EMBD   = 256; // gate_up's k and down's rows
constexpr int64_t N_FF     = 256; // one book's hidden width: gate_up has 2*N_FF rows, down has k = N_FF
constexpr int     N_EXPERT = 64;
// The desk is the smallest source here on purpose: the chain's first link reserves the Metal scratch
// every link uses, so a part with more books than the desk is what an undersized reserve would break.
constexpr int     N_DESK   = 12;                        // books on the desk, in shuffled slots
const std::vector<int> BELT_PARTS = { 28, 16, 8 };     // the other 52 books, in belt order
constexpr size_t  ALIGN    = 256;                       // belt placement (Phase B's slabs)

size_t pad(size_t n) {
    return (n + ALIGN - 1) / ALIGN * ALIGN;
}

// one weight of every book, quantized; book e is bytes [e*book_bytes, (e + 1)*book_bytes)
struct book_weight {
    ggml_type            type;
    int64_t              ne0;
    int64_t              ne1;
    size_t               book_bytes;
    std::vector<uint8_t> data;

    book_weight(ggml_type type, int64_t ne0, int64_t ne1, std::mt19937 & rng)
        : type(type), ne0(ne0), ne1(ne1), book_bytes(ggml_row_size(type, ne0) * ne1) {
        std::uniform_real_distribution<float> u(-0.1f, 0.1f);
        std::vector<float> f(ne0 * ne1 * N_EXPERT);
        for (float & x : f) {
            x = u(rng);
        }
        data.resize(book_bytes * N_EXPERT);
        ggml_quantize_chunk(type, f.data(), data.data(), 0, ne1 * N_EXPERT, ne0, nullptr);
    }

    const uint8_t * book(int e) const { return data.data() + (size_t) e * book_bytes; }
};

// where each book lives once the floor is split: source 0 is the desk, source 1 + p is belt part p
struct room_layout {
    std::vector<int> desk_book;  // desk slot -> book
    std::vector<int> belt_book;  // belt record -> book (part p's records are contiguous)
    std::vector<int> source_of;  // book -> source
    std::vector<int> local_of;   // book -> its slot, or its record within the part

    explicit room_layout(std::mt19937 & rng) : source_of(N_EXPERT), local_of(N_EXPERT) {
        std::vector<int> books(N_EXPERT);
        std::iota(books.begin(), books.end(), 0);
        std::shuffle(books.begin(), books.end(), rng);
        desk_book.assign(books.begin(), books.begin() + N_DESK);
        belt_book.assign(books.begin() + N_DESK, books.end());
        GGML_ASSERT((int) belt_book.size() == std::accumulate(BELT_PARTS.begin(), BELT_PARTS.end(), 0));

        for (int s = 0; s < N_DESK; s++) {
            source_of[desk_book[s]] = 0;
            local_of[desk_book[s]]  = s;
        }
        int r = 0;
        for (size_t p = 0; p < BELT_PARTS.size(); p++) {
            for (int i = 0; i < BELT_PARTS[p]; i++, r++) {
                source_of[belt_book[r]] = 1 + (int) p;
                local_of[belt_book[r]]  = i;
            }
        }
    }

    int n_sources() const { return 1 + (int) BELT_PARTS.size(); }

    // the router's ids for one source: the book's slot or record there, -1 where another source has it
    std::vector<int32_t> remap(const std::vector<int32_t> & ids, int source) const {
        std::vector<int32_t> res(ids.size());
        for (size_t i = 0; i < ids.size(); i++) {
            res[i] = source_of[ids[i]] == source ? local_of[ids[i]] : -1;
        }
        return res;
    }
};

struct room_case {
    ggml_type type_gu;
    ggml_type type_down;
    int       n_used;
    int       n_tokens;

    std::string name() const {
        return std::string(ggml_type_name(type_gu)) + "_" + ggml_type_name(type_down) +
               "_used" + std::to_string(n_used) + "_tokens" + std::to_string(n_tokens);
    }
};

// every context and buffer a case makes, freed when the case ends
struct owned {
    std::vector<ggml_context *>        ctxs;
    std::vector<ggml_backend_buffer_t> bufs;

    ggml_context * ctx(int n_tensors) {
        ggml_init_params params = { ggml_tensor_overhead() * n_tensors + ggml_graph_overhead(), nullptr, true };
        ctxs.push_back(ggml_init(params));
        return ctxs.back();
    }

    ggml_backend_buffer_t alloc(ggml_context * c, ggml_backend_t backend) {
        bufs.push_back(ggml_backend_alloc_ctx_tensors(c, backend));
        GGML_ASSERT(bufs.back() != nullptr);
        return bufs.back();
    }

    ~owned() {
        for (auto * b : bufs) { ggml_backend_buffer_free(b); }
        for (auto * c : ctxs) { ggml_free(c); }
    }
};

// n books of one weight, one per ne[2] index
ggml_tensor * new_experts(ggml_context * ctx, const book_weight & w, int n) {
    return ggml_new_tensor_3d(ctx, w.type, w.ne0, w.ne1, n);
}

void set_books(ggml_tensor * t, const book_weight & w, const std::vector<int> & order) {
    for (size_t i = 0; i < order.size(); i++) {
        ggml_backend_tensor_set(t, w.book(order[i]), i * t->nb[2], w.book_bytes);
    }
}

// the belt's record tensor for one weight: n records of `stride` bytes, this weight at `offs` in each
ggml_tensor * belt_weight(ggml_context * ctx, ggml_backend_buffer_t buf, const book_weight & w,
                          int n, size_t stride, size_t offs) {
    ggml_tensor * t = new_experts(ctx, w, n);
    t->nb[2] = stride;
    t->nb[3] = stride * n;
    GGML_ASSERT(ggml_backend_tensor_alloc(buf, t, (char *) ggml_backend_buffer_get_base(buf) + offs) == GGML_STATUS_SUCCESS);
    return t;
}

// the chain: the desk's books, then each belt part's; the first link's tensor holds every link's rows
ggml_tensor * chain(ggml_context * ctx, const std::vector<ggml_tensor *> & as, ggml_tensor * b,
                    const std::vector<ggml_tensor *> & ids) {
    ggml_tensor * out = nullptr;
    for (size_t i = 0; i < as.size(); i++) {
        out = ggml_mul_mat_id_into(ctx, as[i], b, ids[i], out);
        if (i == 0) {
            // later links are views of this one, and an output flag on a view does not keep its
            // view_src from being freed and reused once the graph moves on
            ggml_set_output(out);
        }
    }
    ggml_set_output(out);
    return out;
}

std::vector<uint8_t> bytes_of(const ggml_tensor * t) {
    std::vector<uint8_t> res(ggml_nbytes(t));
    ggml_backend_tensor_get(t, res.data(), 0, res.size());
    return res;
}

void expect_same(testing & t, const std::string & what, const ggml_tensor * ref, const ggml_tensor * got) {
    const std::vector<uint8_t> a = bytes_of(ref);
    const std::vector<uint8_t> b = bytes_of(got);
    t.assert_equal(what + ": same size", a.size(), b.size());
    t.assert_true(what + ": byte-identical to one mul_mat_id", a.size() == b.size() && memcmp(a.data(), b.data(), a.size()) == 0);

    const float * f = (const float *) a.data();
    const size_t  n = a.size() / sizeof(float);
    size_t nonzero = 0;
    for (size_t i = 0; i < n; i++) {
        nonzero += f[i] != 0.0f;
    }
    t.assert_true(what + ": at least 99% of the compared values are nonzero (" + std::to_string(nonzero) + " of " +
                  std::to_string(n) + ")", nonzero * 100 >= n * 99);
}

void run_case(testing & t, ggml_backend_t backend, ggml_backend_sched_t sched, const room_case & rc) {
    std::mt19937 rng(1234 + rc.n_tokens * 16 + rc.n_used + (int) rc.type_gu * 64 + (int) rc.type_down * 4096);

    const book_weight gu  (rc.type_gu,   N_EMBD, 2 * N_FF, rng);
    const book_weight down(rc.type_down, N_FF,   N_EMBD,   rng);
    const room_layout lay(rng);

    owned own;

    // every book in one tensor per weight: what a run without streaming multiplies by
    ggml_context * ctx_w = own.ctx(8);
    ggml_tensor * gu_all    = new_experts(ctx_w, gu,   N_EXPERT);
    ggml_tensor * down_all  = new_experts(ctx_w, down, N_EXPERT);
    ggml_tensor * gu_desk   = new_experts(ctx_w, gu,   N_DESK);
    ggml_tensor * down_desk = new_experts(ctx_w, down, N_DESK);
    ggml_backend_buffer_set_usage(own.alloc(ctx_w, backend), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    std::vector<int> all(N_EXPERT);
    std::iota(all.begin(), all.end(), 0);
    set_books(gu_all,    gu,   all);
    set_books(down_all,  down, all);
    set_books(gu_desk,   gu,   lay.desk_book);
    set_books(down_desk, down, lay.desk_book);

    // the belt: records of [gate_up | down], each start 256-aligned, one tensor per weight over them
    const int    n_rec  = (int) lay.belt_book.size();
    const size_t stride = pad(gu.book_bytes) + pad(down.book_bytes);
    own.bufs.push_back(ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), n_rec * stride));
    ggml_backend_buffer_t belt = own.bufs.back();
    ggml_backend_buffer_set_usage(belt, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    ggml_context * ctx_b = own.ctx(4);
    ggml_tensor * gu_rec   = belt_weight(ctx_b, belt, gu,   n_rec, stride, 0);
    ggml_tensor * down_rec = belt_weight(ctx_b, belt, down, n_rec, stride, pad(gu.book_bytes));
    set_books(gu_rec,   gu,   lay.belt_book);
    set_books(down_rec, down, lay.belt_book);

    // the router's choice for each token, distinct books, and its remap onto each source
    std::vector<int32_t> ids(rc.n_used * rc.n_tokens);
    for (int tok = 0; tok < rc.n_tokens; tok++) {
        std::shuffle(all.begin(), all.end(), rng);
        std::copy(all.begin(), all.begin() + rc.n_used, ids.begin() + tok * rc.n_used);
    }

    ggml_context * ctx_in = own.ctx(8);
    ggml_tensor * cur     = ggml_new_tensor_3d(ctx_in, GGML_TYPE_F32, N_EMBD, 1, rc.n_tokens);
    ggml_tensor * ids_all = ggml_new_tensor_2d(ctx_in, GGML_TYPE_I32, rc.n_used, rc.n_tokens);
    std::vector<ggml_tensor *> ids_src;
    for (int s = 0; s < lay.n_sources(); s++) {
        ids_src.push_back(ggml_new_tensor_2d(ctx_in, GGML_TYPE_I32, rc.n_used, rc.n_tokens));
    }
    own.alloc(ctx_in, backend);

    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x(N_EMBD * rc.n_tokens);
    for (float & v : x) {
        v = nd(rng);
    }
    ggml_backend_tensor_set(cur, x.data(), 0, ggml_nbytes(cur));
    ggml_backend_tensor_set(ids_all, ids.data(), 0, ggml_nbytes(ids_all));

    std::vector<int> owners(ids.size(), 0);
    for (int s = 0; s < lay.n_sources(); s++) {
        const std::vector<int32_t> r = lay.remap(ids, s);
        for (size_t i = 0; i < r.size(); i++) {
            owners[i] += r[i] >= 0;
        }
        ggml_backend_tensor_set(ids_src[s], r.data(), 0, ggml_nbytes(ids_src[s]));
    }
    t.assert_true("every pair has exactly one source", std::all_of(owners.begin(), owners.end(), [](int n) { return n == 1; }));

    // desk, then the parts: views of the record tensors, a part's first record at its offset
    std::vector<ggml_tensor *> gu_src   = { gu_desk };
    std::vector<ggml_tensor *> down_src = { down_desk };
    ggml_context * ctx_g = own.ctx(64);
    for (int p = 0, first = 0; p < (int) BELT_PARTS.size(); first += BELT_PARTS[p], p++) {
        gu_src.push_back(ggml_view_3d(ctx_g, gu_rec, gu.ne0, gu.ne1, BELT_PARTS[p],
                                      gu_rec->nb[1], stride, first * stride));
        down_src.push_back(ggml_view_3d(ctx_g, down_rec, down.ne0, down.ne1, BELT_PARTS[p],
                                        down_rec->nb[1], stride, first * stride));
    }

    // build_moe_ffn's shape: gate_up, SwiGLU, down; the chains then read back the same ids
    ggml_tensor * up_ref   = ggml_mul_mat_id(ctx_g, gu_all, cur, ids_all);
    ggml_tensor * down_ref = ggml_mul_mat_id(ctx_g, down_all, ggml_swiglu(ctx_g, up_ref), ids_all);
    ggml_set_output(up_ref);
    ggml_set_output(down_ref);

    ggml_tensor * up_room   = chain(ctx_g, gu_src, cur, ids_src);
    ggml_tensor * down_room = chain(ctx_g, down_src, ggml_swiglu(ctx_g, up_room), ids_src);

    // the reference's gate_up output is computed, and so placed, right after the chain's first link, so
    // a later link whose scratch overran that link's reservation would write into a kept output
    ggml_cgraph * gf = ggml_new_graph(ctx_g);
    ggml_build_forward_expand(gf, up_room->view_src);
    ggml_build_forward_expand(gf, up_ref);
    ggml_build_forward_expand(gf, down_room);
    ggml_build_forward_expand(gf, down_ref);

    ggml_backend_sched_reset(sched);
    t.assert_true("graph computed", ggml_backend_sched_graph_compute(sched, gf) == GGML_STATUS_SUCCESS);
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        const ggml_tensor * node = ggml_graph_node(gf, i);
        if (node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_MUL_MAT_ID_INTO) {
            t.assert_true(std::string("expert GEMM ran on ") + ggml_backend_name(backend),
                          ggml_backend_sched_get_tensor_backend(sched, const_cast<ggml_tensor *>(node)) == backend);
        }
    }

    expect_same(t, "gate_up", up_ref, up_room);
    expect_same(t, "down", down_ref, down_room);
}

} // namespace

int main(int argc, char ** argv) {
    testing t;

    const char * verbose = getenv("LLAMA_TEST_VERBOSE");
    t.verbose = verbose && std::string(verbose) == "1";

    if (argc > 1) {
        t.set_filter(argv[1]);
    }

    ggml_backend_load_all();

    // the paperback's books are Q4_K gate_up with Q5_K or Q8_0 down; F16 runs the float kernels.
    // Fewer than 32 tokens take Metal's vector kernel (mv), 32 and more the matrix kernel (mm)
    std::vector<room_case> cases;
    for (auto types : { std::make_pair(GGML_TYPE_F16, GGML_TYPE_F16), std::make_pair(GGML_TYPE_Q4_K, GGML_TYPE_Q8_0),
                        std::make_pair(GGML_TYPE_Q4_K, GGML_TYPE_Q5_K) }) {
        for (int n_used : { 8, 10 }) {
            for (int n_tokens : { 1, 7, 33, 257 }) {
                cases.push_back({ types.first, types.second, n_used, n_tokens });
            }
        }
    }

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(cpu != nullptr);

    for (size_t d = 0; d < ggml_backend_dev_count(); d++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(d);
        const auto type = ggml_backend_dev_type(dev);
        if (type != GGML_BACKEND_DEVICE_TYPE_CPU && type != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }

        ggml_backend_t backend = type == GGML_BACKEND_DEVICE_TYPE_CPU ? cpu : ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend != nullptr);

        // the scheduler llama.cpp computes with: the device, then the CPU as its fallback
        std::vector<ggml_backend_t> backends = { backend };
        if (backend != cpu) {
            backends.push_back(cpu);
        }
        ggml_backend_sched_t sched = ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(),
                                                            GGML_DEFAULT_GRAPH_SIZE, false, true);

        t.test(ggml_backend_dev_name(dev), [&](testing & t) {
            for (const room_case & rc : cases) {
                t.test(rc.name(), [&](testing & t) { run_case(t, backend, sched, rc); });
            }
        });

        ggml_backend_sched_free(sched);
        if (backend != cpu) {
            ggml_backend_free(backend);
        }
    }

    ggml_backend_free(cpu);

    return t.summary();
}
