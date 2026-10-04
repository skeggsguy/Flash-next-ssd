// The chat window's three options (test-server-lane's second file, CHAT-WINDOW-PLAN.md step 1, Tom's "easy for
// other users to configure, off by default"): off by default; set on the command line, by environment variable, or
// in a --models-preset INI file by name; a negative window or an empty header or value is refused.

#include "arg.h"
#include "common.h"
#include "preset.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

extern int n_fail;

void test_config();

#define CHECK(cond) do { if (!(cond)) { ++n_fail; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

static bool parse(std::vector<const char *> argv, common_params & params) {
    argv.insert(argv.begin(), "llama-server");
    return common_params_parse((int) argv.size(), const_cast<char **>(argv.data()), params, LLAMA_EXAMPLE_SERVER);
}

void test_config() {
    {
        common_params p;
        CHECK(p.chat_window == 0); // off
        CHECK(p.chat_window_header == "X-Lane");
        CHECK(p.chat_window_background == "code");
        CHECK(parse({}, p));
        CHECK(p.chat_window == 0);
    }
    {
        common_params p;
        CHECK(parse({"--chat-window", "1200", "--chat-window-header", "X-Client-Role", "--chat-window-background", "agent"}, p));
        CHECK(p.chat_window == 1200);
        CHECK(p.chat_window_header == "X-Client-Role");
        CHECK(p.chat_window_background == "agent");
    }
    {
        common_params p;
        CHECK(!parse({"--chat-window", "-5"}, p));
        common_params q;
        CHECK(!parse({"--chat-window-header", ""}, q));
        common_params r;
        CHECK(!parse({"--chat-window-background", ""}, r));
    }
    {
        setenv("LLAMA_ARG_CHAT_WINDOW", "90", 1);
        setenv("LLAMA_ARG_CHAT_WINDOW_HEADER", "X-Job", 1);
        setenv("LLAMA_ARG_CHAT_WINDOW_BACKGROUND", "batch", 1);
        common_params p;
        CHECK(parse({}, p));
        CHECK(p.chat_window == 90);
        CHECK(p.chat_window_header == "X-Job");
        CHECK(p.chat_window_background == "batch");
        unsetenv("LLAMA_ARG_CHAT_WINDOW");
        unsetenv("LLAMA_ARG_CHAT_WINDOW_HEADER");
        unsetenv("LLAMA_ARG_CHAT_WINDOW_BACKGROUND");
    }
    {
        // the README's preset example: the options by name in a model's section, the window also in [*]
        const std::string path = "test-server-lane-preset.ini";
        {
            std::ofstream f(path);
            f << "version = 1\n[*]\nchat-window = 60\n\n[my-model]\nmodel = /path/to/model.gguf\nparallel = 1\n"
                 "chat-window = 1200\nchat-window-header = X-Lane\nchat-window-background = code\n\n"
                 "[other-model]\nmodel = /path/to/other.gguf\nLLAMA_ARG_CHAT_WINDOW_BACKGROUND = agent\n";
        }
        common_preset_context ctx(LLAMA_EXAMPLE_SERVER);
        common_preset global;
        common_presets presets;
        try {
            presets = ctx.load_from_ini(path, global);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "preset load failed: %s\n", e.what());
        }
        std::remove(path.c_str());
        CHECK(presets.count("my-model") == 1);
        CHECK(presets.count("other-model") == 1);
        if (presets.count("my-model") == 1 && presets.count("other-model") == 1) {
            common_params p;
            presets.at("my-model").apply_to_params(p);
            CHECK(p.chat_window == 1200);
            CHECK(p.chat_window_header == "X-Lane");
            CHECK(p.chat_window_background == "code");

            common_params g;
            global.apply_to_params(g);
            CHECK(g.chat_window == 60);

            common_params o;
            presets.at("other-model").apply_to_params(o);
            CHECK(o.chat_window_background == "agent");
            // and the router hands the options to the model's server as arguments
            const auto args = presets.at("my-model").to_args();
            bool has_window = false;
            for (size_t i = 0; i + 1 < args.size(); i++) {
                has_window |= args[i] == "--chat-window" && args[i + 1] == "1200";
            }
            CHECK(has_window);
        }
    }
}
