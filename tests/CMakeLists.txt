llama_add_compile_flags()

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common) # TODO: [TAG_TESTS_LLAMA_LINK]
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

llama_build(test-recurrent-state-rollback.cpp)
llama_build(test-save-load-state.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-batch-alloc.cpp)
    llama_build_and_test(test-moe-stream-trace.cpp)
    llama_build_and_test(test-moe-stream-split.cpp)
    llama_build_and_test(test-moe-room-gemm.cpp)
    llama_build_and_test(test-moe-room-size.cpp)
    llama_build_and_test(test-moe-room-belt.cpp)
    llama_build_and_test(test-moe-room-lend.cpp)
    llama_build_and_test(test-moe-room-plan.cpp)
    llama_build_and_test(test-top-k-determinism.cpp)
    llama_build_and_test(test-qsa-picks.cpp)
    llama_build_and_test(test-qsa-slice.cpp)

    if (GGML_METAL)
        # fix 2: merged op chains write the same bytes as the unmerged ops, GGML_METAL_FUSION_FN on and off
        llama_build(test-metal-fusion-fn.cpp)
        target_link_libraries(test-metal-fusion-fn PRIVATE ggml-metal)
        foreach (FN 1 0)
            llama_test(test-metal-fusion-fn NAME test-metal-fusion-fn-${FN})
            set_tests_properties(test-metal-fusion-fn-${FN} PROPERTIES ENVIRONMENT "GGML_METAL_FUSION_FN=${FN}")
        endforeach()
        # fix 2a: the trimmed Q8_0 mat-vec's bytes (the old-vs-new diff is in the file's header)
        llama_build_and_test(test-mv-q8-bytes.cpp)
    endif()
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-json-schema-to-grammar.cpp)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build(test-llama-archs.cpp)
    llama_test(test-llama-archs NAME test-layer-input-order ARGS --layer-input-order)

    set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    llama_test(
        test-llama-archs
        NAME test-generate-models
        LABEL main
        ARGS -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models PROPERTIES
        FIXTURES_SETUP generate-models
    )

    # Expert streaming needs more books per floor than a desk of 3*n_expert_used slots holds, so it
    # gets fixtures of its own with 64 experts, 8 used: qwen4exp has a fused gate_up weight, qwen3moe
    # separate gate/up/down. The seed is fixed so an old and a new build read the same weights, and the
    # weight scales sit around 1 so qwen3moe's three scaled GEMMs do not read in as zero on Metal.
    foreach (E64_ARCH qwen4exp qwen3moe)
        llama_test(
            test-llama-archs
            NAME test-generate-models-${E64_ARCH}-e64
            LABEL main
            ARGS -a ${E64_ARCH} --n-expert 64 --n-expert-used 8 --suffix -e64 --unit-scales -s 42 -o "${MODEL_DIR}"
        )
        set_tests_properties(test-generate-models-${E64_ARCH}-e64 PROPERTIES
            FIXTURES_SETUP generate-models-e64
        )
    endforeach()
    # four floors, so a lookahead two floors ahead (LLAMA_MOE_STREAM_LOOKAHEAD_DEPTH2) has somewhere to go
    llama_test(
        test-llama-archs
        NAME test-generate-models-qwen4exp-e64l4
        LABEL main
        ARGS -a qwen4exp --n-expert 64 --n-expert-used 8 --n-layer 4 --suffix -e64l4 --unit-scales -s 42 -o "${MODEL_DIR}"
    )
    set_tests_properties(test-generate-models-qwen4exp-e64l4 PROPERTIES
        FIXTURES_SETUP generate-models-e64l4
    )

    # A desk of 40 slots of 64, i.e. waves of 16 books: -ub 256 reads in through 4 masked waves (2048
    # pairs, too few for 600 a wave), -ub 1024 through 5 partitioned ones; writing takes one wave.
    # Not 24 slots (3*n_expert_used): the partition path's spare-wave sizing then makes 10 waves of
    # 6-7 books, fewer than n_expert_used, and stage_wave_locked has no parking slots to borrow on
    # wave 0 - an abort a real desk (~100 books a wave) cannot reach. The reading room is on by default
    # and would take these read-ins, so the wave paths ask for it off.
    llama_build(test-moe-stream-waves.cpp)
    foreach (E64_ARCH qwen4exp qwen3moe)
        foreach (E64_PATH masked partition)
            if (E64_PATH STREQUAL "masked")
                set(E64_UB 256)
            else()
                set(E64_UB 1024)
            endif()
            llama_test(
                test-moe-stream-waves
                NAME test-moe-stream-waves-${E64_ARCH}-${E64_PATH}
                LABEL main
                ARGS -m "${MODEL_DIR}/${E64_ARCH}-moe-e64.gguf" --moe-stream-cache 40s --moe-stream-room 0
                     -b 1024 -ub ${E64_UB} -n 8 --expect ${E64_PATH}
            )
            set_tests_properties(test-moe-stream-waves-${E64_ARCH}-${E64_PATH} PROPERTIES
                FIXTURES_REQUIRED generate-models-e64
            )
        endforeach()
    endforeach()

    # The reading room against no streaming, byte for byte, on the GPU and on the CPU (-ngl 0); -paths
    # is the study's own I/O path (direct reads, two runners) and the apprentice's graph builds.
    llama_build(test-moe-room-model.cpp)
    llama_build(test-moe-room-model-paths.cpp)
    llama_build(test-moe-room-lend-model.cpp)
    llama_build(test-moe-stream-lookahead-all.cpp)
    llama_build(test-moe-stream-lookahead-depth2.cpp)
    foreach (E64_NGL 99 0)
        llama_test(
            test-moe-stream-lookahead-depth2
            NAME test-moe-stream-lookahead-depth2-qwen4exp-ngl${E64_NGL}
            LABEL main
            ARGS -m "${MODEL_DIR}/qwen4exp-moe-e64l4.gguf" -ngl ${E64_NGL}
        )
        set_tests_properties(test-moe-stream-lookahead-depth2-qwen4exp-ngl${E64_NGL} PROPERTIES
            FIXTURES_REQUIRED generate-models-e64l4
        )
    endforeach()
    foreach (E64_TEST test-moe-room-model test-moe-room-model-paths test-moe-room-lend-model test-moe-stream-lookahead-all)
        foreach (E64_ARCH qwen4exp qwen3moe)
            foreach (E64_NGL 99 0)
                llama_test(
                    ${E64_TEST}
                    NAME ${E64_TEST}-${E64_ARCH}-ngl${E64_NGL}
                    LABEL main
                    ARGS -m "${MODEL_DIR}/${E64_ARCH}-moe-e64.gguf" -ngl ${E64_NGL}
                )
                set_tests_properties(${E64_TEST}-${E64_ARCH}-ngl${E64_NGL} PROPERTIES
                    FIXTURES_REQUIRED generate-models-e64
                )
            endforeach()
        endforeach()
    endforeach()

    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-nemotron-h
        LABEL main
        ARGS -m "${MODEL_DIR}/nemotron_h-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-nemotron-h PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-dsv4
        LABEL main
        ARGS -m "${MODEL_DIR}/deepseek4-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-dsv4 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-kimi-k3
        LABEL main
        ARGS -m "${MODEL_DIR}/kimi-k3-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-kimi-k3 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    # Test state save/load functionality across all architectures, using the generated dummy models
    llama_test(
        test-save-load-state
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    set_tests_properties(test-save-load-state PROPERTIES FIXTURES_REQUIRED generate-models)

    # qwen4exp's indexer cache, with the QSA block summaries rebuilt every ubatch (switch off) and kept (patch 4i)
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-qwen4exp
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf"
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-qwen4exp-qsa-keep
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-qwen4exp test-recurrent-state-rollback-qwen4exp-qsa-keep PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    set_tests_properties(test-recurrent-state-rollback-qwen4exp PROPERTIES
        ENVIRONMENT "LLAMA_QSA_KEEP=0"
    )
    set_tests_properties(test-recurrent-state-rollback-qwen4exp-qsa-keep PROPERTIES
        ENVIRONMENT "LLAMA_QSA_KEEP=1"
    )

    llama_test(
        test-save-load-state
        NAME test-save-load-state-qsa-keep
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    set_tests_properties(test-save-load-state-qsa-keep PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "LLAMA_QSA_KEEP=1"
    )
    # both write dump_state.bin into the same directory
    set_tests_properties(test-save-load-state test-save-load-state-qsa-keep PROPERTIES RESOURCE_LOCK dump_state_bin)

    # patch 4i: keeping the QSA block summaries is bit-identical to rebuilding them, on the GPU and
    # the CPU, and with each switch that changes how the summaries are pooled or read
    llama_build(test-qsa-keep.cpp)

    foreach (QSA_KEEP_ENV "" "QWEN4EXP_QSA_GATHER=0" "LLAMA_QSA_POOL_VIEWS=0" "LLAMA_QWEN4EXP_BLOCK_TOPK=0" "LLAMA_QSA_GATHER=0")
        set(QSA_KEEP_NAME test-qsa-keep)
        if (QSA_KEEP_ENV)
            string(REGEX REPLACE "=.*" "" QSA_KEEP_SUFFIX "${QSA_KEEP_ENV}")
            string(TOLOWER "${QSA_KEEP_SUFFIX}" QSA_KEEP_SUFFIX)
            string(REPLACE "_" "-" QSA_KEEP_SUFFIX "${QSA_KEEP_SUFFIX}")
            set(QSA_KEEP_NAME "${QSA_KEEP_NAME}-${QSA_KEEP_SUFFIX}")
        endif()

        llama_test(test-qsa-keep NAME ${QSA_KEEP_NAME}     LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf")
        llama_test(test-qsa-keep NAME ${QSA_KEEP_NAME}-cpu LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" -ngl 0)

        set_tests_properties(${QSA_KEEP_NAME} ${QSA_KEEP_NAME}-cpu PROPERTIES FIXTURES_REQUIRED generate-models)
        if (QSA_KEEP_ENV)
            set_tests_properties(${QSA_KEEP_NAME} ${QSA_KEEP_NAME}-cpu PROPERTIES ENVIRONMENT "${QSA_KEEP_ENV}")
        endif()
    endforeach()

    # SHARE-PARTS-PLAN phase 3: the picker and attention sliced 128 rows at a time (batches of 512) write every
    # byte the whole batch does, on the GPU and the CPU; the run fails unless slicing really ran. The unit-scale
    # fixture is the one whose attention output depends on the queries (N(0, 0.01) weights leave it flat)
    llama_test(test-qsa-keep NAME test-qsa-keep-slice         LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf")
    llama_test(test-qsa-keep NAME test-qsa-keep-slice-cpu     LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" -ngl 0)
    llama_test(test-qsa-keep NAME test-qsa-keep-slice-e64     LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe-e64.gguf")
    llama_test(test-qsa-keep NAME test-qsa-keep-slice-e64-cpu LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe-e64.gguf" -ngl 0)
    # and with a quantized cache, whose rotation is applied once to the whole batch's q and undone after the join
    llama_test(test-qsa-keep NAME test-qsa-keep-slice-rot     LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe-e64.gguf" -ctk q8_0 -ctv q8_0 -fa on)
    set_tests_properties(test-qsa-keep-slice test-qsa-keep-slice-cpu test-qsa-keep-slice-e64 test-qsa-keep-slice-e64-cpu
        test-qsa-keep-slice-rot PROPERTIES ENVIRONMENT "QSA_KEEP_SLICE=128")
    set_tests_properties(test-qsa-keep-slice test-qsa-keep-slice-cpu PROPERTIES FIXTURES_REQUIRED generate-models)
    set_tests_properties(test-qsa-keep-slice-e64 test-qsa-keep-slice-e64-cpu test-qsa-keep-slice-rot PROPERTIES
        FIXTURES_REQUIRED generate-models-e64)

    # SHARE-PARTS-PLAN phase 1: the block top-k picker never picks a block after the token. The block
    # path on the GPU and the CPU (the CPU run also holds the picks to one token at a time); the per-cell
    # path (LLAMA_QWEN4EXP_BLOCK_TOPK=0, read once per process) must not move a byte either way
    llama_build(test-qsa-causal.cpp)
    llama_test(test-qsa-causal NAME test-qsa-causal              LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf")
    llama_test(test-qsa-causal NAME test-qsa-causal-cpu          LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" -ngl 0)
    llama_test(test-qsa-causal NAME test-qsa-causal-per-cell     LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf")
    llama_test(test-qsa-causal NAME test-qsa-causal-per-cell-cpu LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" -ngl 0)
    set_tests_properties(test-qsa-causal test-qsa-causal-cpu test-qsa-causal-per-cell test-qsa-causal-per-cell-cpu
        PROPERTIES FIXTURES_REQUIRED generate-models)
    set_tests_properties(test-qsa-causal-per-cell test-qsa-causal-per-cell-cpu
        PROPERTIES ENVIRONMENT "LLAMA_QWEN4EXP_BLOCK_TOPK=0")

    # SHARE-PARTS-PLAN phase 2: the phrasebook trace (LLAMA_PLE_TRACE). The writer's bytes by hand, then the tiny
    # qwen4exp fixture read the default way and with direct reads: the rows are the n-gram hash's, the logits do not move
    llama_build(test-ple-trace.cpp)
    llama_test(test-ple-trace NAME test-ple-trace     LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf")
    llama_test(test-ple-trace NAME test-ple-trace-cpu LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" -ngl 0)
    set_tests_properties(test-ple-trace test-ple-trace-cpu PROPERTIES FIXTURES_REQUIRED generate-models)

    llama_build(test-fusion.cpp)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
llama_build_and_test(test-json-schema.cpp)
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp)
llama_build_and_test(test-speculative-adaptive.cpp)
llama_build_and_test(test-speculative-rate.cpp)
llama_build_and_test(test-model-resolution.cpp)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()

# TODO: make this test (and others) not link `libllama` as it is not needed [TAG_TESTS_LLAMA_LINK]
llama_build(test-backend-ops.cpp)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

if (APPLE)
    llama_build(test-rset-release.cpp)
endif()

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build_and_test(test-rope.cpp)
    llama_build_and_test(test-col2im-1d.cpp)
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

llama_build_and_test(test-mtmd-impl.cpp)
target_link_libraries(test-mtmd-impl PRIVATE mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()
