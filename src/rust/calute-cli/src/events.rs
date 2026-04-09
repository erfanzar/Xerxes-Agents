/// Protocol event types mirroring Python's streaming events.
///
/// These are deserialized from newline-delimited JSON coming from the Python bridge server.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;


#[derive(Debug, Serialize)]
pub struct Request {
    pub method: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl Request {
    pub fn init(model: &str, permission_mode: &str, base_url: Option<&str>, api_key: Option<&str>) -> Self {
        let mut params = serde_json::json!({
            "model": model,
            "permission_mode": permission_mode,
        });
        if let Some(url) = base_url {
            params["base_url"] = serde_json::Value::String(url.to_string());
        }
        if let Some(key) = api_key {
            params["api_key"] = serde_json::Value::String(key.to_string());
        }
        Self {
            method: "init",
            params: Some(params),
        }
    }

    pub fn query(text: &str) -> Self {
        Self {
            method: "query",
            params: Some(serde_json::json!({"text": text})),
        }
    }

    pub fn permission_response(granted: bool) -> Self {
        Self {
            method: "permission_response",
            params: Some(serde_json::json!({"granted": granted})),
        }
    }

    pub fn cancel() -> Self {
        Self {
            method: "cancel",
            params: None,
        }
    }

    pub fn slash(command: &str) -> Self {
        Self {
            method: "slash",
            params: Some(serde_json::json!({"command": command})),
        }
    }

    pub fn provider_list() -> Self {
        Self {
            method: "provider_list",
            params: None,
        }
    }

    pub fn fetch_models(base_url: &str, api_key: &str) -> Self {
        Self {
            method: "fetch_models",
            params: Some(serde_json::json!({"base_url": base_url, "api_key": api_key})),
        }
    }

    pub fn provider_save(name: &str, base_url: &str, api_key: &str, model: &str) -> Self {
        Self {
            method: "provider_save",
            params: Some(serde_json::json!({
                "name": name,
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
            })),
        }
    }

    pub fn provider_select(name: &str) -> Self {
        Self {
            method: "provider_select",
            params: Some(serde_json::json!({"name": name})),
        }
    }
}


#[derive(Debug, Deserialize)]
pub struct RawEvent {
    pub event: String,
    #[serde(default)]
    pub data: serde_json::Value,
}

/// Typed events parsed from the Python bridge.
#[derive(Debug, Clone)]
pub enum Event {
    Ready {
        model: String,
        provider: String,
        tools: usize,
        permission_mode: String,
        has_profile: bool,
    },
    TextChunk {
        text: String,
    },
    ThinkingChunk {
        text: String,
    },
    ToolStart {
        name: String,
        inputs: HashMap<String, serde_json::Value>,
        tool_call_id: String,
    },
    ToolEnd {
        name: String,
        result: String,
        permitted: bool,
        tool_call_id: String,
        duration_ms: f64,
    },
    PermissionRequest {
        tool_name: String,
        description: String,
        inputs: HashMap<String, serde_json::Value>,
    },
    TurnDone {
        input_tokens: u64,
        output_tokens: u64,
        tool_calls_count: u64,
        model: String,
    },
    QueryDone,
    SlashResult {
        output: String,
    },
    State {
        turn_count: u64,
        total_input_tokens: u64,
        total_output_tokens: u64,
        message_count: u64,
        cost_usd: f64,
    },
    Error {
        message: String,
    },
    ProviderList {
        profiles: Vec<serde_json::Value>,
    },
    ModelsList {
        models: Vec<String>,
        base_url: String,
    },
    ProviderSaved {
        message: String,
        model: String,
        provider: String,
    },
    Exit,
    Unknown(String),
}

impl Event {
    /// Parse a raw JSON event into a typed Event.
    pub fn parse(raw: RawEvent) -> Self {
        let d = &raw.data;
        match raw.event.as_str() {
            "ready" => Event::Ready {
                model: d["model"].as_str().unwrap_or("").to_string(),
                provider: d["provider"].as_str().unwrap_or("").to_string(),
                tools: d["tools"].as_u64().unwrap_or(0) as usize,
                permission_mode: d["permission_mode"].as_str().unwrap_or("auto").to_string(),
                has_profile: d["has_profile"].as_bool().unwrap_or(false),
            },
            "text_chunk" => Event::TextChunk {
                text: d["text"].as_str().unwrap_or("").to_string(),
            },
            "thinking_chunk" => Event::ThinkingChunk {
                text: d["text"].as_str().unwrap_or("").to_string(),
            },
            "tool_start" => {
                let inputs: HashMap<String, serde_json::Value> = d
                    .get("inputs")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                Event::ToolStart {
                    name: d["name"].as_str().unwrap_or("").to_string(),
                    inputs,
                    tool_call_id: d["tool_call_id"].as_str().unwrap_or("").to_string(),
                }
            }
            "tool_end" => Event::ToolEnd {
                name: d["name"].as_str().unwrap_or("").to_string(),
                result: d["result"].as_str().unwrap_or("").to_string(),
                permitted: d["permitted"].as_bool().unwrap_or(true),
                tool_call_id: d["tool_call_id"].as_str().unwrap_or("").to_string(),
                duration_ms: d["duration_ms"].as_f64().unwrap_or(0.0),
            },
            "permission_request" => {
                let inputs: HashMap<String, serde_json::Value> = d
                    .get("inputs")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                Event::PermissionRequest {
                    tool_name: d["tool_name"].as_str().unwrap_or("").to_string(),
                    description: d["description"].as_str().unwrap_or("").to_string(),
                    inputs,
                }
            }
            "turn_done" => Event::TurnDone {
                input_tokens: d["input_tokens"].as_u64().unwrap_or(0),
                output_tokens: d["output_tokens"].as_u64().unwrap_or(0),
                tool_calls_count: d["tool_calls_count"].as_u64().unwrap_or(0),
                model: d["model"].as_str().unwrap_or("").to_string(),
            },
            "query_done" => Event::QueryDone,
            "slash_result" => Event::SlashResult {
                output: d["output"].as_str().unwrap_or("").to_string(),
            },
            "state" => Event::State {
                turn_count: d["turn_count"].as_u64().unwrap_or(0),
                total_input_tokens: d["total_input_tokens"].as_u64().unwrap_or(0),
                total_output_tokens: d["total_output_tokens"].as_u64().unwrap_or(0),
                message_count: d["message_count"].as_u64().unwrap_or(0),
                cost_usd: d["cost_usd"].as_f64().unwrap_or(0.0),
            },
            "error" => Event::Error {
                message: d["message"].as_str().unwrap_or("").to_string(),
            },
            "provider_list" => {
                let profiles = d.get("profiles")
                    .and_then(|v| v.as_array())
                    .map(|a| a.clone())
                    .unwrap_or_default();
                Event::ProviderList { profiles }
            }
            "models_list" => {
                let models = d.get("models")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|m| m.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                Event::ModelsList {
                    models,
                    base_url: d["base_url"].as_str().unwrap_or("").to_string(),
                }
            }
            "provider_saved" => {
                let profile = d.get("profile").unwrap_or(d);
                Event::ProviderSaved {
                    message: d["message"].as_str().unwrap_or("").to_string(),
                    model: profile.get("model").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    provider: profile.get("provider").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                }
            }
            "exit" => Event::Exit,
            other => Event::Unknown(other.to_string()),
        }
    }
}
