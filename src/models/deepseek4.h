#pragma once

#include <string>

struct llama_model;

bool llama_deepseek4_validate_mtp_draft_gguf(const std::string & path, const llama_model * model, std::string & err);
