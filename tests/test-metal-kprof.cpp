// GGML_METAL_KPROF's node map (study patch, #10b step 1.2: the check's GPU time by kind of work), on MTL0 with no
// model. The profiler times each node as its own pass and names a pass by its graph's node map (KPROFN lines:
// op, name, shape). A matrix product's name alone cannot say whether it multiplies staff (a weight) or the
// context (attention, the picker), and many are unnamed ("node_N"), so each map line also carries the name of the
// node's first source ("src0"): a weight's GGUF name for staff, an activation's for the rest.
//
// On (GGML_METAL_KPROF=1): the product's line names its weight, the next node's names the product, a graph's map
// is written once however often it runs, and the timed passes name the product's node. Off (=0): no KPROF line
// at all, and the result is the same either way.
//
// ctest runs it twice: GGML_METAL_KPROF=1 and =0.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string & what) {
    printf("  %-72s %s\n", what.c_str(), ok ? "ok" : "FAILED");
    g_failures += !ok;
}

std::vector<std::string> lines_of(FILE * f) {
    std::vector<std::string> out;
    rewind(f);
    char buf[4096];
    while (fgets(buf, sizeof(buf), f)) {
        out.emplace_back(buf);
    }
    return out;
}

bool has(const std::string & s, const char * part) {
    return s.find(part) != std::string::npos;
}

// the "node" number of a KPROFN line
int node_of(const std::string & line) {
    const size_t at = line.find("\"node\":");
    return at == std::string::npos ? -2 : atoi(line.c_str() + at + 7);
}

} // namespace

int main() {
    const char * env = getenv("GGML_METAL_KPROF");
    const bool on = env != nullptr && atoi(env) > 0;
    printf("GGML_METAL_KPROF=%s: %s\n", env ? env : "(unset)", on ? "on" : "off");

    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (dev == nullptr) {
        printf("no MTL0 device: skipped\n");
        return 0;
    }

    // everything the backend writes to stderr from here to its free goes to a file
    fflush(stderr);
    FILE * cap = tmpfile();
    const int saved = dup(2);
    dup2(fileno(cap), 2);

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);

    const int64_t k = 64, m = 32, n = 2;
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, m);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_set_name(w, "blk.0.probe.weight");
    ggml_set_name(x, "probe_x");
    ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    ggml_set_name(y, "probe_y");
    ggml_tensor * z = ggml_scale(ctx, y, 2.0f);
    ggml_set_name(z, "probe_z");
    ggml_cgraph * g = ggml_new_graph(ctx);
    ggml_build_forward_expand(g, z);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> wv(k*m), xv(k*n);
    for (size_t i = 0; i < wv.size(); i++) {
        wv[i] = (float) ((int) (i % 7) - 3) / 8.0f;
    }
    for (size_t i = 0; i < xv.size(); i++) {
        xv[i] = (float) ((int) (i % 5) - 2) / 4.0f;
    }
    ggml_backend_tensor_set(w, wv.data(), 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, xv.data(), 0, ggml_nbytes(x));

    ggml_backend_graph_compute(backend, g);
    ggml_backend_graph_compute(backend, g);
    std::vector<float> got(m*n);
    ggml_backend_tensor_get(z, got.data(), 0, ggml_nbytes(z));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend); // flushes the profiler's last passes

    fflush(stderr);
    dup2(saved, 2);
    close(saved);
    const std::vector<std::string> lines = lines_of(cap);
    fclose(cap);

    double err = 0.0;
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < m; i++) {
            double want = 0.0;
            for (int64_t c = 0; c < k; c++) {
                want += (double) wv[i*k + c]*xv[j*k + c];
            }
            err = fmax(err, fabs(2.0*want - got[j*m + i]));
        }
    }
    check(err < 1e-4, "the result is 2 w.x (profiled or not)");

    int n_kprof = 0, n_graphs = 0, n_y = 0, n_z = 0, y_node = -1;
    bool y_src = false, z_src = false, y_shape = false;
    std::vector<int> seg_nodes;
    for (const std::string & l : lines) {
        if (l.rfind("KPROF", 0) != 0) {
            continue;
        }
        n_kprof++;
        if (l.rfind("KPROFS ", 0) == 0) {
            n_graphs++;
        } else if (l.rfind("KPROFN ", 0) == 0 && has(l, "\"name\":\"probe_y\"")) {
            n_y++;
            y_node = node_of(l);
            y_src = has(l, "\"op\":\"MUL_MAT\"") && has(l, "\"src0\":\"blk.0.probe.weight\"");
            y_shape = has(l, "\"ne\":[32,2,1,1]}");
        } else if (l.rfind("KPROFN ", 0) == 0 && has(l, "\"name\":\"probe_z\"")) {
            n_z++;
            z_src = has(l, "\"src0\":\"probe_y\"");
        } else if (l.rfind("KPROF ", 0) == 0 && !has(l, "\"error\"")) {
            seg_nodes.push_back(node_of(l));
        }
    }

    if (!on) {
        check(n_kprof == 0, "off: no KPROF line");
    } else {
        check(n_graphs == 2, "on: one KPROFS line a compute");
        check(n_y == 1 && n_z == 1, "on: the graph's map is written once for two computes");
        check(y_src, "on: the product's map line names its weight as src0");
        check(z_src, "on: the next node's map line names the product as src0");
        check(y_shape, "on: the map line still ends with the node's shape");
        int y_segs = 0;
        for (int s : seg_nodes) {
            y_segs += s == y_node;
        }
        check(y_node >= 0 && y_segs == 2, "on: each compute has a timed pass named by the product's node");
    }

    printf("%s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
