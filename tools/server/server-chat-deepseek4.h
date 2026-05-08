#pragma once

#include "common.h"
#include "chat.h"

#include <map>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;

json dsv4_build_prompt_tokens(
    const json & messages,
    const json & tools,
    const std::map<std::string, std::string> & chat_template_kwargs,
    bool enable_thinking,
    bool tool_choice_none,
    const llama_vocab * vocab);

void dsv4_apply_sampling_defaults(
    json & llama_params,
    const json & body,
    bool enable_thinking);
