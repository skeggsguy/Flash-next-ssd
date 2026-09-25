#pragma once

// The reading room, manager side: the book manager's (llama_moe_stream, llama-moe-stream.h) belt of
// books for long read-ins. The sizing arithmetic is llama-moe-room-size.h.

#include "llama.h"
#include "llama-arch.h"
#include "llama-moe-room-size.h"

#include <cstdint>

struct llama_hparams;
struct llama_model_loader;

// Sizes the room for a model about to load, from the desk's slots with the room off: the layout's
// desk_slots are what the desk keeps. Logs the startup line (and any warning) in plain words; throws
// std::runtime_error with the plain-words reason when the room cannot be made - an arch whose expert
// maths a split floor would change, or a size the arithmetic refuses.
llama_moe_room_layout llama_moe_room_size_model(const llama_model_params & params, llm_arch arch,
        const llama_hparams & hparams, const llama_model_loader & ml, uint32_t n_slots);
