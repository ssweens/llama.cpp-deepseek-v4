#include "server-chat-deepseek4.h"
#include "server-common.h"

#include <array>
#include <unordered_map>
#include <unordered_set>

namespace {

static bool dsv4_role_is_system(const std::string & role) {
    return role == "system" || role == "developer";
}

static bool dsv4_role_is_user_like(const std::string & role) {
    return role == "user" || role == "tool" || role == "function";
}

static void dsv4_append_attr_escaped(std::string & out, const std::string & s) {
    for (char c : s) {
        if (c == '&') out += "&amp;";
        else if (c == '<') out += "&lt;";
        else if (c == '>') out += "&gt;";
        else if (c == '"') out += "&quot;";
        else out.push_back(c);
    }
}

static void dsv4_append_text_escaped(std::string & out, const std::string & s) {
    for (char c : s) {
        if (c == '&') out += "&amp;";
        else if (c == '<') out += "&lt;";
        else if (c == '>') out += "&gt;";
        else out.push_back(c);
    }
}

static void dsv4_append_json_literal_escaped(std::string & out, const std::string & s) {
    for (char c : s) {
        if (c == '<') out += "\\u003c";
        else if (c == '>') out += "\\u003e";
        else if (c == '&') out += "\\u0026";
        else out.push_back(c);
    }
}

static std::string dsv4_message_content_text(const json & msg) {
    if (!msg.contains("content") || msg["content"].is_null()) {
        return "";
    }
    const auto & content = msg["content"];
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (!content.is_array()) {
        return "";
    }

    std::string out;
    for (const auto & part : content) {
        if (!part.is_object()) {
            continue;
        }
        const std::string type = json_value(part, "type", std::string());
        if ((type == "text" || type == "media_marker") && part.contains("text") && part["text"].is_string()) {
            out += part["text"].get<std::string>();
        }
    }
    return out;
}

static std::unordered_map<std::string, std::vector<std::string>> dsv4_tool_prop_orders(const json & tools) {
    std::unordered_map<std::string, std::vector<std::string>> orders;
    if (!tools.is_array()) {
        return orders;
    }

    for (const auto & tool : tools) {
        const json * schema = &tool;
        if (tool.is_object() && json_value(tool, "type", std::string()) == "function" && tool.contains("function") && tool["function"].is_object()) {
            schema = &tool["function"];
        }
        if (!schema->is_object()) {
            continue;
        }

        const std::string name = json_value(*schema, "name", std::string());
        if (name.empty()) {
            continue;
        }

        const json * props = nullptr;
        if (schema->contains("input_schema") && (*schema)["input_schema"].is_object()) {
            const auto & input_schema = (*schema)["input_schema"];
            if (input_schema.contains("properties") && input_schema["properties"].is_object()) {
                props = &input_schema["properties"];
            }
        }
        if (props == nullptr && schema->contains("parameters") && (*schema)["parameters"].is_object()) {
            const auto & parameters = (*schema)["parameters"];
            if (parameters.contains("properties") && parameters["properties"].is_object()) {
                props = &parameters["properties"];
            }
        }

        if (props != nullptr) {
            auto & dst = orders[name];
            for (const auto & item : props->items()) {
                dst.push_back(item.key());
            }
        }
    }

    return orders;
}

static std::string dsv4_tools_schema_lines(const json & tools) {
    if (!tools.is_array() || tools.empty()) {
        return "";
    }

    std::string out;
    for (const auto & tool : tools) {
        const json * schema = &tool;
        if (tool.is_object() && json_value(tool, "type", std::string()) == "function" && tool.contains("function") && tool["function"].is_object()) {
            schema = &tool["function"];
        }
        if (!schema->is_object()) {
            continue;
        }
        out += schema->dump();
        out.push_back('\n');
    }
    if (!out.empty()) {
        out.pop_back();
    }
    return out;
}

static void dsv4_append_parameter(std::string & out, const std::string & key, const json & value) {
    out += "<｜DSML｜parameter name=\"";
    dsv4_append_attr_escaped(out, key);
    out += "\" string=\"";
    out += value.is_string() ? "true" : "false";
    out += "\">";
    if (value.is_string()) {
        dsv4_append_text_escaped(out, value.get<std::string>());
    } else {
        dsv4_append_json_literal_escaped(out, value.dump());
    }
    out += "</｜DSML｜parameter>\n";
}

static void dsv4_append_tool_calls_text(
        std::string & out,
        const json & tool_calls,
        const std::unordered_map<std::string, std::vector<std::string>> & orders) {
    if (!tool_calls.is_array() || tool_calls.empty()) {
        return;
    }

    out += "\n\n<｜DSML｜tool_calls>\n";
    for (const auto & call : tool_calls) {
        const auto & fn = call.contains("function") && call["function"].is_object() ? call["function"] : call;
        const std::string name = json_value(fn, "name", std::string());
        const std::string args = json_value(fn, "arguments", std::string("{}"));

        out += "<｜DSML｜invoke name=\"";
        dsv4_append_attr_escaped(out, name);
        out += "\">\n";

        bool appended = false;
        try {
            json args_json = json::parse(args);
            if (args_json.is_object()) {
                std::unordered_set<std::string> used;
                auto it = orders.find(name);
                if (it != orders.end()) {
                    for (const auto & key : it->second) {
                        if (args_json.contains(key)) {
                            dsv4_append_parameter(out, key, args_json[key]);
                            used.insert(key);
                        }
                    }
                }
                for (const auto & item : args_json.items()) {
                    if (used.find(item.key()) == used.end()) {
                        dsv4_append_parameter(out, item.key(), item.value());
                    }
                }
                appended = true;
            }
        } catch (...) {
        }

        if (!appended) {
            out += "<｜DSML｜parameter name=\"arguments\" string=\"true\">";
            dsv4_append_text_escaped(out, args);
            out += "</｜DSML｜parameter>\n";
        }

        out += "</｜DSML｜invoke>\n";
    }
    out += "</｜DSML｜tool_calls>";
}

static std::string dsv4_tools_prompt_block(const std::string & tool_schemas) {
    if (tool_schemas.empty()) {
        return "";
    }

    return std::string(
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user question. "
        "You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n"
        "<｜DSML｜tool_calls>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
        "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>\n\n"
        "String parameters should be specified as is and set `string=\"true\"`. "
        "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
        "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n"
        "Otherwise, output directly after </think> with tool calls or final response.\n\n"
        "### Available Tool Schemas\n\n") +
        tool_schemas +
        "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. "
        "Emit parameters in the same order as each tool's input_schema.properties or parameters.properties object.";
}

static std::string dsv4_think_max_prefix() {
    return
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";
}

static std::string dsv4_render_chat_prompt(
        const json & messages,
        const json & tools,
        bool think_enabled,
        bool think_max,
        bool tool_choice_none) {
    const json active_tools = tool_choice_none ? json::array() : tools;
    const std::string tool_schemas = dsv4_tools_schema_lines(active_tools);
    const auto orders = dsv4_tool_prop_orders(active_tools);

    bool tool_context = !tool_schemas.empty();
    int last_user_idx = -1;

    std::string system;
    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        const std::string role = json_value(m, "role", std::string("user"));
        if (!dsv4_role_is_system(role)) {
            continue;
        }
        if (!system.empty()) {
            system += "\n\n";
        }
        system += dsv4_message_content_text(m);
    }

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        const std::string role = json_value(m, "role", std::string("user"));
        if (dsv4_role_is_user_like(role)) {
            last_user_idx = (int) i;
        }
        if ((role == "assistant" && m.contains("tool_calls") && m["tool_calls"].is_array() && !m["tool_calls"].empty()) || role == "tool" || role == "function") {
            tool_context = true;
        }
    }

    if (!tool_schemas.empty()) {
        if (!system.empty()) {
            system += "\n\n";
        }
        system += dsv4_tools_prompt_block(tool_schemas);
    }

    std::string out = "<｜begin▁of▁sentence｜>";
    if (think_max) {
        out += dsv4_think_max_prefix();
    }
    out += system;

    bool pending_assistant = false;
    bool pending_tool_result = false;

    for (size_t i = 0; i < messages.size(); ++i) {
        const auto & m = messages[i];
        const std::string role = json_value(m, "role", std::string("user"));
        const std::string content = dsv4_message_content_text(m);

        if (dsv4_role_is_system(role)) {
            continue;
        }

        if (role == "user") {
            out += "<｜User｜>";
            out += content;
            pending_assistant = true;
            pending_tool_result = false;
            continue;
        }

        if (role == "tool" || role == "function") {
            if (!pending_tool_result) {
                out += "<｜User｜>";
            }
            out += "<tool_result>";
            dsv4_append_text_escaped(out, content);
            out += "</tool_result>";
            pending_assistant = true;
            pending_tool_result = true;
            continue;
        }

        if (role == "assistant") {
            if (pending_assistant) {
                out += "<｜Assistant｜>";
                if (think_enabled) {
                    if (tool_context || (int) i > last_user_idx) {
                        out += "<think>";
                        out += json_value(m, "reasoning_content", std::string());
                        out += "</think>";
                    } else {
                        out += "</think>";
                    }
                } else {
                    out += "</think>";
                }
            }
            out += content;
            if (m.contains("tool_calls")) {
                dsv4_append_tool_calls_text(out, m["tool_calls"], orders);
            }
            out += "<｜end▁of▁sentence｜>";
            pending_assistant = false;
            pending_tool_result = false;
            continue;
        }
    }

    if (pending_assistant) {
        out += "<｜Assistant｜>";
        out += think_enabled ? "<think>" : "</think>";
    }

    return out;
}

static llama_tokens dsv4_tokenize_rendered_chat(const llama_vocab * vocab, const std::string & text) {
    struct special_tok {
        const char * text;
    };

    static const std::array<special_tok, 7> specials = {{
        {"<｜begin▁of▁sentence｜>"},
        {"<｜end▁of▁sentence｜>"},
        {"<｜User｜>"},
        {"<｜Assistant｜>"},
        {"<think>"},
        {"</think>"},
        {"｜DSML｜"},
    }};

    llama_tokens out;

    size_t span_start = 0;
    size_t i = 0;
    while (i < text.size()) {
        bool matched = false;
        size_t matched_len = 0;

        for (const auto & sp : specials) {
            const std::string token_text = sp.text;
            if (text.compare(i, token_text.size(), token_text) == 0) {
                matched = true;
                matched_len = token_text.size();
                break;
            }
        }

        if (!matched) {
            ++i;
            continue;
        }

        if (i > span_start) {
            const std::string span = text.substr(span_start, i - span_start);
            auto span_tokens = common_tokenize(vocab, span, false, false);
            out.insert(out.end(), span_tokens.begin(), span_tokens.end());
        }

        const std::string s = text.substr(i, matched_len);
        auto special_tokens = common_tokenize(vocab, s, false, true);
        out.insert(out.end(), special_tokens.begin(), special_tokens.end());

        i += matched_len;
        span_start = i;
    }

    if (span_start < text.size()) {
        const std::string span = text.substr(span_start);
        auto span_tokens = common_tokenize(vocab, span, false, false);
        out.insert(out.end(), span_tokens.begin(), span_tokens.end());
    }

    return out;
}

} // namespace

json dsv4_build_prompt_tokens(
    const json & messages,
    const json & tools,
    const std::map<std::string, std::string> & chat_template_kwargs,
    bool enable_thinking,
    bool tool_choice_none,
    const llama_vocab * vocab) {
    const std::string reasoning_effort = json_value(chat_template_kwargs, "reasoning_effort", std::string(""));
    const bool think_max = reasoning_effort == "\"max\"" || reasoning_effort == "max";

    const std::string prompt_text = dsv4_render_chat_prompt(messages, tools, enable_thinking, think_max, tool_choice_none);
    const auto prompt_tokens = dsv4_tokenize_rendered_chat(vocab, prompt_text);

    json prompt = json::array();
    for (const auto tok : prompt_tokens) {
        prompt.push_back(tok);
    }
    return prompt;
}

void dsv4_apply_sampling_defaults(
    json & llama_params,
    const json & body,
    bool enable_thinking) {
    if (!body.contains("temperature") || enable_thinking) {
        llama_params["temperature"] = 1.0f;
    }
    if (!body.contains("top_k") || enable_thinking) {
        llama_params["top_k"] = 0;
    }
    if (!body.contains("top_p") || enable_thinking) {
        llama_params["top_p"] = 1.0f;
    }
    if (!body.contains("min_p") || enable_thinking) {
        llama_params["min_p"] = 0.0f;
    }
}
