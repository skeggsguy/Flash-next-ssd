#pragma once

// helpers shared by test-server-shelf.cpp and test-server-shelf-file.cpp (one test, two files to stay short)

#include "server-shelf.h"

#include <cstdio>
#include <vector>

extern int n_fail;

#define CHECK(cond) do { if (!(cond)) { ++n_fail; std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); } } while (0)

// n tokens counting up from first, then n2 counting up from second (two conversations share the first part)
inline llama_tokens toks(llama_token first, int n, llama_token second, int n2) {
    llama_tokens t;
    for (int i = 0; i < n; i++) {
        t.push_back(first + i);
    }
    for (int i = 0; i < n2; i++) {
        t.push_back(second + i);
    }
    return t;
}

// v followed by n tokens counting up from first (a follow-up in the same conversation)
inline llama_tokens extend(llama_tokens v, llama_token first, int n) {
    for (int i = 0; i < n; i++) {
        v.push_back(first + i);
    }
    return v;
}

inline server_tokens st(const llama_tokens & t) {
    return server_tokens(t, false);
}

// a bookmark the rules see, without a file
inline void add_entry(server_shelf & shelf, uint64_t id, const llama_tokens & t, uint64_t size) {
    server_shelf_entry e;
    e.id     = id;
    e.tokens = st(t);
    e.size   = size;
    shelf.entries.push_back(std::move(e));
}

// in test-server-shelf-file.cpp
void test_files();
void test_record();
void test_folder();
