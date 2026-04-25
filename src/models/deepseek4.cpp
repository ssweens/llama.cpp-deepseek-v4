#include "models.h"

#include <stdexcept>

llm_build_deepseek4::llm_build_deepseek4(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    GGML_UNUSED(model);
    throw std::runtime_error("DeepSeek-V4 graph execution is not implemented yet");
}
