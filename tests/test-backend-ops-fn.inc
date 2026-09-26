// Fix 2 (fewer GPU jobs per written token): the op chains GGML_METAL_FUSION_FN merges, at
// Qwen3.8-Flash-Next's own shapes, included by test-backend-ops.cpp.
//
// Each chain is built exactly as the model builds it, so on a backend with GGML_METAL_FUSION_FN=1 the
// whole-graph comparison runs the merged kernel against the CPU. The "blocked" variant marks one
// intermediate as a graph output: a merged kernel never writes its intermediates, so the chain must
// then run unmerged, and the comparison checks the intermediate as well as the result. The same
// cases, timed with `perf`, give the before/after of a merge (C0 records the before).
//
// No input is a param (ggml_set_param): a param becomes an OP_NONE node and would sit between the
// chain's nodes, which is not what the model's graph looks like.

#include <numeric>

// P4: hc mix's SCALE(1/hc) + SILU, [hc_low_rank, n_tokens] (build_hc_mix)
struct test_fn_scale_silu : public test_case {
    const std::array<int64_t, 2> ne;
    const bool blocked;

    std::string op_desc(ggml_tensor * t) override { GGML_UNUSED(t); return "FN_SCALE_SILU"; }
    std::string vars() override { return VARS_TO_STR2(ne, blocked); }
    bool run_whole_graph() override { return true; }
    std::vector<ggml_tensor *> fusion_test_nodes() override { return nodes; }

    test_fn_scale_silu(std::array<int64_t, 2> ne = {320, 1}, bool blocked = false) : ne(ne), blocked(blocked) {}

    std::vector<ggml_tensor *> nodes;

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * lo = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne[0], ne[1]);
        ggml_set_name(lo, "lo");

        ggml_tensor * s = ggml_scale(ctx, lo, 1.0f/4.0f);
        ggml_set_name(s, "scaled");
        ggml_tensor * out = ggml_silu(ctx, s);
        ggml_set_name(out, "out");

        nodes = { out };
        if (blocked) {
            ggml_set_output(s);
            nodes.push_back(s);
        }
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -80.0f, 80.0f);
        }
    }
};

// P1: hc combine's scatter weights 2*sigmoid(inject/hc) folded into DSV4_HC_POST (build_hc_combine)
struct test_fn_hc_post : public test_case {
    const int64_t n_embd;
    const int64_t n_tokens;
    const int blocked; // 0: none, 1: the first SCALE is an output, 2: the SIGMOID is

    std::string op_desc(ggml_tensor * t) override { GGML_UNUSED(t); return "FN_HC_POST"; }
    std::string vars() override { return VARS_TO_STR3(n_embd, n_tokens, blocked); }
    bool run_whole_graph() override { return true; }
    std::vector<ggml_tensor *> fusion_test_nodes() override { return nodes; }

    test_fn_hc_post(int64_t n_embd = 2560, int64_t n_tokens = 1, int blocked = 0)
        : n_embd(n_embd), n_tokens(n_tokens), blocked(blocked) {}

    std::vector<ggml_tensor *> nodes;

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t hc = 4;

        ggml_tensor * block_out = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_tensor * residual  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_tensor * inject    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, n_tokens);
        ggml_set_name(block_out, "block_out");
        ggml_set_name(residual,  "residual");
        ggml_set_name(inject,    "inject");

        ggml_tensor * s = ggml_scale(ctx, inject, 1.0f/(float) hc);
        ggml_tensor * g = ggml_sigmoid(ctx, s);
        ggml_tensor * w = ggml_scale(ctx, g, 2.0f);
        ggml_tensor * out = ggml_dsv4_hc_post(ctx, block_out, residual, w, nullptr);
        ggml_set_name(out, "out");

        nodes = { out };
        if (blocked == 1) {
            ggml_set_output(s);
            nodes.push_back(s);
        } else if (blocked == 2) {
            ggml_set_output(g);
            nodes.push_back(g);
        }
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            init_tensor_uniform(t, -20.0f, 20.0f);
        }
    }
};

// P3: the router's weights, GET_ROWS + SUM_ROWS + CLAMP + DIV (build_moe_ffn with norm_w)
struct test_fn_router_w : public test_case {
    const int64_t n_expert;
    const int64_t n_used;
    const int64_t n_tokens;
    const int blocked; // 0: none, 1: the picked weights (GET_ROWS) are an output, 2: the sum is

    std::string op_desc(ggml_tensor * t) override { GGML_UNUSED(t); return "FN_ROUTER_W"; }
    std::string vars() override { return VARS_TO_STR4(n_expert, n_used, n_tokens, blocked); }
    bool run_whole_graph() override { return true; }
    std::vector<ggml_tensor *> fusion_test_nodes() override { return nodes; }

    test_fn_router_w(int64_t n_expert = 512, int64_t n_used = 10, int64_t n_tokens = 1, int blocked = 0)
        : n_expert(n_expert), n_used(n_used), n_tokens(n_tokens), blocked(blocked) {}

    std::vector<ggml_tensor *> nodes;

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * probs = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_expert, n_tokens);
        ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used,   n_tokens);
        ggml_set_name(probs, "probs");
        ggml_set_name(ids,   "ids");

        ggml_tensor * w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_expert, n_tokens), ids);
        ggml_tensor * w2 = ggml_reshape_2d(ctx, w, n_used, n_tokens);
        ggml_tensor * sum = ggml_sum_rows(ctx, w2);
        ggml_tensor * sum_c = ggml_clamp(ctx, sum, 6.103515625e-5, INFINITY);
        ggml_tensor * out = ggml_div(ctx, w2, sum_c);
        ggml_set_name(out, "out");

        nodes = { out };
        if (blocked == 1) {
            ggml_set_output(w);
            nodes.push_back(w);
        } else if (blocked == 2) {
            ggml_set_output(sum);
            nodes.push_back(sum);
        }
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        std::random_device rd;
        std::default_random_engine rng(rd());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type == GGML_TYPE_I32) {
                // n_used distinct books per token, as the top-k picks them
                std::vector<int32_t> all(n_expert);
                for (int64_t r = 0; r < ggml_nrows(t); r++) {
                    std::iota(all.begin(), all.end(), 0);
                    std::shuffle(all.begin(), all.end(), rng);
                    ggml_backend_tensor_set(t, all.data(), r*t->nb[1], n_used*sizeof(int32_t));
                }
            } else if (!ggml_is_view_op(t->op)) {
                init_tensor_uniform(t, 0.0f, 1.0f);
            }
        }
    }
};

// eval: the model's shapes at 1, 6 and 512 tokens, plus each chain blocked by an intermediate output
static void test_fn_add_eval_cases(std::vector<std::unique_ptr<test_case>> & test_cases) {
    for (int64_t nt : {1, 4, 6, 512}) {
        test_cases.emplace_back(new test_fn_scale_silu({320, nt}, false));
        test_cases.emplace_back(new test_fn_hc_post(2560, nt, 0));
        test_cases.emplace_back(new test_fn_router_w(512, 10, nt, 0));
    }
    test_cases.emplace_back(new test_fn_scale_silu({322, 3}, false)); // not a multiple of 4: the scalar kernels
    test_cases.emplace_back(new test_fn_router_w(64, 8, 5, 0));       // the 64-book test models
    for (int64_t nt : {1, 6}) {
        test_cases.emplace_back(new test_fn_scale_silu({320, nt}, true));
        test_cases.emplace_back(new test_fn_hc_post(2560, nt, 1));
        test_cases.emplace_back(new test_fn_hc_post(2560, nt, 2));
        test_cases.emplace_back(new test_fn_router_w(512, 10, nt, 1));
        test_cases.emplace_back(new test_fn_router_w(512, 10, nt, 2));
    }
}

// perf: writing (1 token) at the model's shapes
static void test_fn_add_perf_cases(std::vector<std::unique_ptr<test_case>> & test_cases) {
    test_cases.emplace_back(new test_fn_scale_silu({320, 1}, false));
    test_cases.emplace_back(new test_fn_hc_post(2560, 1, 0));
    test_cases.emplace_back(new test_fn_router_w(512, 10, 1, 0));
}
