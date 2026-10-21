// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming XML-ish tool-call parser for MiniMax-M3.
//!
//! MiniMax-M3 prefixes every tag with the namespace token `]<]minimax[>[` and
//! names parameters by their TAG (not a `name=` attribute):
//!   `]<]minimax[>[<tool_call>
//!    ]<]minimax[>[<invoke name="NAME">
//!    ]<]minimax[>[<KEY>value]<]minimax[>[</KEY>
//!    ]<]minimax[>[</invoke>
//!    ]<]minimax[>[</tool_call>`
//! plus a bare `]<]minimax[>[<invoke ...>...</invoke>` back-off form when the
//! outer wrapper is absent.
//!
//! The streaming concern (buffering, chunk-split marker safety, normal_text
//! suppression) is owned by the shared [`scan::WrappedBlockScanner`]; all
//! three markers begin with the namespace token, so the shared holdback also
//! retains a split `]<]minimax[>[` run. The per-invoke value typing is
//! delegated to the v1 batch parser `try_tool_call_parse_minimax_m3` driven by
//! the same config `dynamo_parsers` uses for batch parsing, so a streamed call
//! matches exactly what the batch parser produces. Arguments are re-serialized
//! in source parameter-tag order because the v1 parser builds them from a
//! `HashMap` whose key order is non-deterministic.
//!
//! While an invoke is open, schema-declared plain-string parameters stream as
//! JSON string fragments; the first parameter of any other shape stops
//! streaming, and the rest of the call arrives at the invoke close.

use crate::tool_calling::scan::{
    BareRecoveryLatch, InvokeEmitter, InvokeLatch, WrappedBlockScanner, WrappedBlockSpec,
    marker_prefix_suffix_len, reorder_arguments,
};
use crate::tool_calling::v1core::xml::{minimax_m3_html_unescape, minimax_m3_invoke_name};
use crate::tool_calling::v1core::{
    MiniMaxM3ParserConfig, ToolDefinition, try_tool_call_parse_minimax_m3,
};

use crate::tool_calling::traits::{Tool, ToolCallDelta, ToolParseResult, ToolParser};
use std::collections::HashSet;

/// The namespace token emitted before every M3 tag.
const NS: &str = "]<]minimax[>[";
/// Opens a parameter tag, or a nested element inside a value.
const TAG_OPEN: &str = "]<]minimax[>[<";
/// Longest input the batch entity decoder rewrites as one unit (`&amp;quot;`).
const MAX_ENTITY_SPAN: usize = 10;
const BLOCK_START: &str = "]<]minimax[>[<tool_call>";
const BLOCK_END: &str = "]<]minimax[>[</tool_call>";
/// Bare `<invoke` (no trailing `>`): matches both `<invoke name="...">` and the
/// malformed nameless `<invoke>` (whose parse yields no call and is dropped).
const FUNCTION_START: &str = "]<]minimax[>[<invoke";
const FUNCTION_END: &str = "]<]minimax[>[</invoke>";

fn spec() -> WrappedBlockSpec {
    WrappedBlockSpec {
        family: "minimax_m3",
        block_starts: vec![BLOCK_START.to_string()],
        block_ends: vec![BLOCK_END.to_string()],
        invoke_start: FUNCTION_START.to_string(),
        invoke_end: FUNCTION_END.to_string(),
        orphan_markers: vec![BLOCK_END.to_string()],
        // BLOCK_END is held back too: after a bare-invoke recovery latches
        // suppression, a split closing marker must be retained whole so the
        // orphan-close path (which clears the latch) can match it.
        holdback_markers: vec![
            BLOCK_START.to_string(),
            FUNCTION_START.to_string(),
            BLOCK_END.to_string(),
        ],
        bare_recovery_latch: BareRecoveryLatch::Set,
        invoke_latch: InvokeLatch::IfEmitted,
        drop_invoke_crossing_block_end: false,
        // Every wrapped family's markers are special tokens today.
        preserve_special_tokens: true,
        ..Default::default()
    }
}

/// Value-typing hook: wraps one complete invoke run in the M3 `<tool_call>`
/// block so the v1 parser takes its normal wrapped path, then re-orders the
/// arguments to source parameter-tag order.
struct M3Emitter {
    config: MiniMaxM3ParserConfig,
    tools: Vec<ToolDefinition>,
    partial: Option<PartialInvoke>,
}

/// Arguments already streamed for the open invoke.
struct PartialInvoke {
    tool_index: usize,
    /// `<invoke ...>` header and the tail of the input seen so far, so state
    /// from an invoke the scanner dropped is never applied to a new one.
    header: String,
    seen_len: usize,
    seen_tail: String,
    name: String,
    emitted_json: String,
    scan_cursor: usize,
    seen_parameters: HashSet<String>,
    active: Option<ActiveString>,
    blocked: bool,
}

struct ActiveString {
    closer: String,
    /// Raw value bytes before this offset have been emitted.
    value_cursor: usize,
    search_cursor: usize,
    /// `{"key":"` or `,"key":"`, sent with the first non-whitespace value byte
    /// so a value that turns out to be nested XML has emitted nothing.
    opener: String,
    started: bool,
}

enum ValueScan {
    Open(usize),
    Closed(usize),
    Nested,
}

impl PartialInvoke {
    fn continues(&self, invoke: &str, tool_index: usize) -> bool {
        self.tool_index == tool_index
            && invoke.starts_with(&self.header)
            && invoke
                .get(..self.seen_len)
                .is_some_and(|seen| seen.ends_with(&self.seen_tail))
    }

    fn mark_seen(&mut self, invoke: &str) {
        let mut tail = invoke.len().saturating_sub(16);
        while !invoke.is_char_boundary(tail) {
            tail -= 1;
        }
        self.seen_len = invoke.len();
        self.seen_tail = invoke[tail..].to_string();
    }
}

fn is_plain_string(tools: &[ToolDefinition], tool: &str, parameter: &str) -> bool {
    tools
        .iter()
        .find(|definition| definition.name == tool)
        .and_then(|definition| definition.parameters.as_ref())
        .and_then(|schema| schema.get("properties"))
        .and_then(|properties| properties.get(parameter))
        .is_some_and(|schema| {
            schema.get("type").and_then(serde_json::Value::as_str) == Some("string")
                && ["$ref", "allOf", "anyOf", "oneOf", "enum", "const"]
                    .iter()
                    .all(|key| schema.get(key).is_none())
                && schema.get("nullable").and_then(serde_json::Value::as_bool) != Some(true)
        })
}

impl InvokeEmitter for M3Emitter {
    fn parse_partial_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        if self
            .partial
            .as_ref()
            .is_some_and(|partial| !partial.continues(invoke, tool_index))
        {
            self.partial = None;
        }
        if self.partial.is_none() {
            let Some(header_end) = invoke
                .strip_prefix(FUNCTION_START)
                .and_then(|attributes| attributes.find('>'))
                .map(|offset| FUNCTION_START.len() + offset)
            else {
                return Ok(None);
            };
            let name = minimax_m3_invoke_name(&invoke[FUNCTION_START.len()..header_end])
                .unwrap_or_default();
            let known = self.tools.iter().any(|tool| tool.name == name);
            self.partial = Some(PartialInvoke {
                tool_index,
                header: invoke[..=header_end].to_string(),
                seen_len: 0,
                seen_tail: String::new(),
                name,
                emitted_json: String::new(),
                scan_cursor: header_end + 1,
                seen_parameters: HashSet::new(),
                active: None,
                blocked: !known,
            });
        }

        let partial = self.partial.as_mut().expect("initialized above");
        partial.mark_seen(invoke);
        if partial.blocked {
            return Ok(None);
        }
        let mut arguments = String::new();
        loop {
            if let Some(active) = partial.active.as_mut() {
                match scan_string_value(invoke, active) {
                    ValueScan::Open(end) => {
                        let end = entity_safe_end(invoke, active.value_cursor, end);
                        append_value(active, &invoke[..end], false, &mut arguments);
                        break;
                    }
                    ValueScan::Closed(end) => {
                        append_value(active, &invoke[..end], true, &mut arguments);
                        arguments.push('"');
                        partial.scan_cursor = end + active.closer.len();
                        partial.active = None;
                        continue;
                    }
                    ValueScan::Nested => {
                        partial.blocked = true;
                        break;
                    }
                }
            }

            let Some(relative) = invoke[partial.scan_cursor..].find(TAG_OPEN) else {
                break;
            };
            let tag = partial.scan_cursor + relative + TAG_OPEN.len();
            let Some(tag_len) = invoke[tag..].find('>') else {
                break;
            };
            let parameter = &invoke[tag..tag + tag_len];
            // The batch parser skips closers and malformed tag names.
            if parameter.is_empty()
                || parameter.starts_with('/')
                || parameter.contains(char::is_whitespace)
            {
                partial.scan_cursor = tag + tag_len + 1;
                continue;
            }
            if !partial.seen_parameters.insert(parameter.to_string())
                || !is_plain_string(&self.tools, &partial.name, parameter)
            {
                partial.blocked = true;
                break;
            }
            let mut opener =
                String::from(if partial.emitted_json.is_empty() && arguments.is_empty() {
                    "{"
                } else {
                    ","
                });
            opener.push_str(&serde_json::to_string(parameter)?);
            opener.push_str(":\"");
            let value_start = tag + tag_len + 1;
            partial.active = Some(ActiveString {
                closer: format!("{NS}</{parameter}>"),
                value_cursor: value_start,
                search_cursor: value_start,
                opener,
                started: false,
            });
        }
        if arguments.is_empty() {
            return Ok(None);
        }
        let first = partial.emitted_json.is_empty();
        partial.emitted_json.push_str(&arguments);
        Ok(Some(ToolCallDelta {
            tool_index,
            name: first.then(|| partial.name.clone()),
            arguments,
            complete: false,
        }))
    }

    fn parse_invoke(
        &mut self,
        invoke: &str,
        tool_index: usize,
    ) -> anyhow::Result<Option<ToolCallDelta>> {
        let partial = self
            .partial
            .take()
            .filter(|partial| partial.continues(invoke, tool_index));
        let wrapped = format!("{BLOCK_START}{invoke}{BLOCK_END}");
        let tools_opt = (!self.tools.is_empty()).then_some(self.tools.as_slice());
        let (calls, _content) = try_tool_call_parse_minimax_m3(&wrapped, &self.config, tools_opt)?;
        let Some(call) = calls.into_iter().next() else {
            return Ok(None);
        };
        let arguments =
            reorder_arguments(&call.function.arguments, &source_parameter_order(invoke));
        let emitted = partial
            .map(|partial| partial.emitted_json)
            .unwrap_or_default();
        if emitted.is_empty() {
            return Ok(Some(ToolCallDelta {
                tool_index,
                name: Some(call.function.name),
                arguments,
                complete: true,
            }));
        }
        let Some(suffix) = arguments.strip_prefix(&emitted) else {
            // Already-streamed text cannot be retracted, so leave the call
            // incomplete for the coalescer to discard.
            tracing::warn!(
                why = "minimax_m3_streamed_call_invalidated",
                tool_index,
                "streamed MiniMax-M3 arguments no longer match the completed call"
            );
            return Ok(Some(ToolCallDelta {
                tool_index,
                name: None,
                arguments: String::new(),
                complete: false,
            }));
        };
        Ok(Some(ToolCallDelta {
            tool_index,
            name: None,
            arguments: suffix.to_string(),
            complete: true,
        }))
    }

    fn reset(&mut self) {
        self.partial = None;
    }
}

/// Finds where an open string value ends, or how far it is safe to read.
fn scan_string_value(invoke: &str, active: &mut ActiveString) -> ValueScan {
    loop {
        let Some(relative) = invoke[active.search_cursor..].find(NS) else {
            let held = marker_prefix_suffix_len(&invoke[active.search_cursor..], [NS]);
            active.search_cursor = invoke.len() - held;
            return ValueScan::Open(active.search_cursor);
        };
        let at = active.search_cursor + relative;
        let rest = &invoke[at..];
        if rest.starts_with(&active.closer) {
            return ValueScan::Closed(at);
        }
        if active.closer.starts_with(rest) {
            active.search_cursor = at;
            return ValueScan::Open(at);
        }
        // The batch parser reads any value containing a tag as nested XML.
        if rest.starts_with(TAG_OPEN) {
            return ValueScan::Nested;
        }
        active.search_cursor = at + NS.len();
    }
}

/// Backs `end` off any `&` the batch decoder could still combine with later text.
fn entity_safe_end(text: &str, start: usize, end: usize) -> usize {
    let bytes = text.as_bytes();
    let mut end = end;
    loop {
        let from = end.saturating_sub(MAX_ENTITY_SPAN).max(start);
        match bytes[from..end].iter().position(|&byte| byte == b'&') {
            Some(offset) => end = from + offset,
            None => return end,
        }
    }
}

fn append_value(active: &mut ActiveString, text: &str, closing: bool, output: &mut String) {
    let raw = &text[active.value_cursor..];
    if !active.started {
        if !closing && raw.trim().is_empty() {
            return;
        }
        output.push_str(&active.opener);
        active.started = true;
    }
    let encoded = serde_json::to_string(&minimax_m3_html_unescape(raw))
        .expect("serializing a string cannot fail");
    output.push_str(&encoded[1..encoded.len() - 1]);
    active.value_cursor = text.len();
}

/// Stream parser for MiniMax-M3 tool calls.
pub struct MiniMaxM3ToolStreamParser {
    scanner: WrappedBlockScanner<M3Emitter>,
}

impl MiniMaxM3ToolStreamParser {
    pub fn new(tools: &[Tool]) -> Self {
        Self {
            scanner: WrappedBlockScanner::new(
                spec(),
                M3Emitter {
                    // Identical to `dynamo_parsers`' batch config so the streamed
                    // value typing matches the v1 batch parser exactly.
                    config: MiniMaxM3ParserConfig::default(),
                    tools: tools.iter().map(ToolDefinition::from).collect(),
                    partial: None,
                },
            ),
        }
    }
}

impl ToolParser for MiniMaxM3ToolStreamParser {
    fn create(tools: &[Tool]) -> anyhow::Result<Box<dyn ToolParser>>
    where
        Self: Sized + 'static,
    {
        Ok(Box::new(Self::new(tools)))
    }

    fn preserve_special_tokens(&self) -> bool {
        self.scanner.preserve_special_tokens()
    }

    fn push(&mut self, chunk: &str) -> anyhow::Result<ToolParseResult> {
        self.scanner.push(chunk)
    }

    fn finish(&mut self) -> anyhow::Result<ToolParseResult> {
        self.scanner.finish()
    }
}

/// TOP-LEVEL parameter tag names in the order they appear in an invoke run.
///
/// M3 names parameters by their tag (`]<]minimax[>[<location>value...`), and
/// values may nest further namespaced tags (arrays/objects — batch cases
/// 7.d/7.e), so the scan tracks tag depth and records only the depth-0 openers
/// inside the invoke body; nested tag names never shadow a top-level key.
fn source_parameter_order(function: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut depth: i32 = 0;
    let mut cursor = 0;
    while let Some(rel) = function[cursor..].find(NS) {
        let tag_start = cursor + rel + NS.len();
        let rest = &function[tag_start..];
        let Some(after_lt) = rest.strip_prefix('<') else {
            cursor = tag_start;
            continue;
        };
        if let Some(closer) = after_lt.strip_prefix('/') {
            // `</invoke>` closes the run; any other closer pops one level.
            if !closer.starts_with("invoke") {
                depth -= 1;
            }
            cursor = tag_start + 1;
            continue;
        }
        if after_lt.starts_with("invoke") {
            cursor = tag_start + 1;
            continue;
        }
        let Some(name_end) = after_lt.find('>') else {
            break;
        };
        let name = after_lt[..name_end].trim();
        if !name.is_empty() {
            if depth == 0 {
                names.push(name.to_string());
            }
            depth += 1;
        }
        cursor = tag_start + 1 + name_end + 1;
    }
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weather_tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": { "location": { "type": "string" } }
            }),
            strict: None,
        }]
    }

    fn parse_chunks(tools: &[Tool], chunks: &[&str]) -> ToolParseResult {
        let mut parser = MiniMaxM3ToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for chunk in chunks {
            out.append(parser.push(chunk).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    #[test]
    fn null_constraints_preserve_argument_types() {
        let input = concat!(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"get_weather\">",
            "]<]minimax[>[<location>null]<]minimax[>[</location>",
            "]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
        );
        for (schema, expected) in [
            (
                serde_json::json!({"anyOf": [{"type": "string"}, {"const": null}]}),
                "{\"location\":null}",
            ),
            (
                serde_json::json!({"type": "string", "anyOf": [{"type": "null"}, {"type": "string"}]}),
                "{\"location\":\"null\"}",
            ),
        ] {
            let mut tools = weather_tools();
            tools[0].parameters["properties"]["location"] = schema;
            let output = parse_chunks(&tools, &[input]).coalesce_calls();
            assert_eq!(output.calls.len(), 1);
            assert_eq!(output.calls[0].arguments, expected);
            assert!(output.calls[0].complete);
        }
    }

    #[test]
    fn emits_complete_call_on_close() {
        let out = parse_chunks(
            &weather_tools(),
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">",
                "\n]<]minimax[>[<location>",
                "NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>",
                "\n]<]minimax[>[</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].tool_index, 0);
        assert_eq!(merged.calls[0].name.as_deref(), Some("get_weather"));
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn preserves_prefix_and_trailing_text() {
        // 8.c: prefix before the block AND narration after the close both flow
        // into normal_text verbatim; the block markup is suppressed.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check the weather. ]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">",
                "\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
                " Let me know if you need more.",
            ],
        );
        assert_eq!(
            out.normal_text,
            "I will check the weather.  Let me know if you need more."
        );
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn suppresses_in_block_narration_between_invokes() {
        // 8.d: prose INSIDE the block between two invokes is part of the markup
        // block and is dropped, matching the v1 batch parser.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check both cities. ]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>",
                "\nThen check LA.\n]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>LA]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "I will check both cities. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 2);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
        assert_eq!(merged.calls[1].arguments, r#"{"location":"LA"}"#);
    }

    #[test]
    fn recovers_bare_invoke_without_wrapper() {
        // 5.g-shaped: prose + bare invoke (no `<tool_call>` opener) + orphan close.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check that. ]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>",
                "\n]<]minimax[>[</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "I will check that. ");
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn holds_back_split_orphan_close_after_bare_invoke() {
        // 5.g-shaped, but the orphan `BLOCK_END` close is split across two chunk
        // boundaries while `suppress_normal_text` is latched by the bare-invoke
        // recovery. The split close must be held back whole so the orphan-close
        // path can consume it and clear the latch; otherwise its remainder is
        // unrecognizable, suppression stays on, and the trailing prose is dropped.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "I will check that. ]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax",
                "[>[</tool",
                "_call>Here is the weather.",
            ],
        );
        // The close marker never leaks, and the trailing prose survives.
        assert_eq!(out.normal_text, "I will check that. Here is the weather.");
        assert!(!out.normal_text.contains("tool_call"));
        assert!(!out.normal_text.contains("minimax"));
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(merged.calls[0].arguments, r#"{"location":"NYC"}"#);
    }

    #[test]
    fn suppresses_incomplete_invoke_at_eof() {
        // 5.c: block start + invoke truncated mid-value never leaks as text or
        // assembles a call; the streamed prefix stays incomplete.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">",
                "\n]<]minimax[>[<location>NY",
            ],
        );
        assert_eq!(out.normal_text, "");
        assert!(!out.calls.is_empty());
        assert!(out.calls.iter().all(|call| !call.complete));
        assert!(out.coalesce_calls().calls.is_empty());
    }

    #[test]
    fn holds_back_split_namespace_token() {
        // A chunk boundary inside the namespace token must not leak its prefix.
        let out = parse_chunks(
            &weather_tools(),
            &[
                "text before ]<]mini",
                "max[>[<tool_call>\n]<]minimax[>[<invoke name=\"get_weather\">\n]<]minimax[>[<location>NYC]<]minimax[>[</location>\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        assert_eq!(out.normal_text, "text before ");
        assert_eq!(out.coalesce_calls().calls.len(), 1);
    }

    #[test]
    fn preserves_source_parameter_order_and_types() {
        // 7.a-shaped: multiple typed parameters keep model-emitted order.
        let tools = vec![Tool {
            name: "book_flight".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "destination": { "type": "string" },
                    "passengers": { "type": "integer" },
                    "first_class": { "type": "boolean" }
                }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"book_flight\">",
                "\n]<]minimax[>[<destination>Paris]<]minimax[>[</destination>",
                "\n]<]minimax[>[<passengers>2]<]minimax[>[</passengers>",
                "\n]<]minimax[>[<first_class>true]<]minimax[>[</first_class>",
                "\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        assert_eq!(
            merged.calls[0].arguments,
            r#"{"destination":"Paris","passengers":2,"first_class":true}"#
        );
    }

    #[test]
    fn nested_value_tags_do_not_shadow_top_level_order() {
        // 7.d-shaped: nested array/object tags inside a value must not be taken
        // for top-level parameter names when re-ordering.
        let tools = vec![Tool {
            name: "process_data".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "items": { "type": "array", "items": { "type": "integer" } },
                    "config": { "type": "object" }
                }
            }),
            strict: None,
        }];
        let out = parse_chunks(
            &tools,
            &[
                "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"process_data\">",
                "\n]<]minimax[>[<items>]<]minimax[>[<item>1]<]minimax[>[</item>]<]minimax[>[<item>2]<]minimax[>[</item>]<]minimax[>[</items>",
                "\n]<]minimax[>[<config>]<]minimax[>[<mode>fast]<]minimax[>[</mode>]<]minimax[>[</config>",
                "\n]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
            ],
        );
        let merged = out.coalesce_calls();
        assert_eq!(merged.calls.len(), 1);
        let args: serde_json::Value = serde_json::from_str(&merged.calls[0].arguments).unwrap();
        assert_eq!(args["items"], serde_json::json!([1, 2]));
        assert_eq!(args["config"]["mode"], "fast");
        // Top-level order: items before config.
        assert!(
            merged.calls[0].arguments.find("\"items\"").unwrap()
                < merged.calls[0].arguments.find("\"config\"").unwrap()
        );
    }

    #[test]
    fn nested_union_child_values_follow_payload_shape() {
        let tools = vec![Tool {
            name: "list_notes".into(),
            description: None,
            parameters: serde_json::json!({
                "properties": {"pagination": {"anyOf": [
                    {"type": "object", "properties": {
                        "page": {"type": "integer"},
                        "mode": {"anyOf": [{"type": "string"}, {"type": "object"}]},
                        "after": {"type": ["object", "null"]},
                        "config": {"type": "object", "properties": {"enabled": {"type": "boolean"}}}
                    }},
                    {"type": "null"}
                ]}}
            }),
            strict: Some(true),
        }];
        let input = concat!(
            "]<]minimax[>[<tool_call>",
            "]<]minimax[>[<invoke name=\"list_notes\">",
            "]<]minimax[>[<pagination>",
            "]<]minimax[>[<page>2]<]minimax[>[</page>",
            "]<]minimax[>[<mode>one]<]minimax[>[</mode>",
            "]<]minimax[>[<after>null]<]minimax[>[</after>",
            "]<]minimax[>[<config>]<]minimax[>[<enabled>true]<]minimax[>[</enabled>]<]minimax[>[</config>",
            "]<]minimax[>[</pagination>",
            "]<]minimax[>[</invoke>",
            "]<]minimax[>[</tool_call>"
        );
        for width in [1, input.len()] {
            let chunks: Vec<_> = input
                .as_bytes()
                .chunks(width)
                .map(|chunk| std::str::from_utf8(chunk).unwrap())
                .collect();
            let merged = parse_chunks(&tools, &chunks).coalesce_calls();
            assert_eq!(merged.calls.len(), 1);
            let args: serde_json::Value = serde_json::from_str(&merged.calls[0].arguments).unwrap();
            assert_eq!(
                args,
                serde_json::json!({"pagination": {"page": 2, "mode": "one", "after": null, "config": {"enabled": true}}}),
                "width {width}"
            );
        }
    }

    #[test]
    fn literal_null_branch_does_not_erase_nested_integer_type() {
        for union in ["anyOf", "oneOf"] {
            for alternative in [
                serde_json::json!({"enum":[null]}),
                serde_json::json!({"const":null}),
            ] {
                let tools = vec![Tool {
                    name: "list_notes".into(),
                    description: None,
                    strict: None,
                    parameters: serde_json::json!({"properties":{"pagination":{union:[
                        {"type":"object","properties":{"page":{"type":"integer"}}},alternative
                    ]}}}),
                }];
                let input = concat!(
                    "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"list_notes\">",
                    "]<]minimax[>[<pagination>]<]minimax[>[<page>2]<]minimax[>[</page>",
                    "]<]minimax[>[</pagination>]<]minimax[>[</invoke>]<]minimax[>[</tool_call>"
                );
                for width in [1, input.len()] {
                    let chunks: Vec<_> = input
                        .as_bytes()
                        .chunks(width)
                        .map(|c| std::str::from_utf8(c).unwrap())
                        .collect();
                    let out = parse_chunks(&tools, &chunks).coalesce_calls();
                    assert_eq!(out.calls.len(), 1);
                    assert_eq!(
                        serde_json::from_str::<serde_json::Value>(&out.calls[0].arguments).unwrap(),
                        serde_json::json!({"pagination":{"page":2}}),
                        "{union}, {alternative}, width {width}"
                    );
                }
            }
        }
    }

    #[test]
    fn nested_and_encoded_refs_preserve_types_across_chunk_boundaries() {
        let tools = vec![Tool {
            name: "create_order".into(),
            description: None,
            strict: None,
            parameters: serde_json::json!({
                "$defs": {
                    "postal code": {"type": "integer"},
                    "Address": {"type": "object", "properties": {
                        "zip": {"$ref": "#/$defs/postal%20code"},
                        "primary": {"$ref": "#/$defs/Flag"}
                    }},
                    "Flag": {"type": "boolean"}
                },
                "properties": {"shipping": {"$ref": "#/$defs/Address"}}
            }),
        }];
        let input = concat!(
            "Preparing. ]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"create_order\">",
            "]<]minimax[>[<shipping>]<]minimax[>[<zip>18956]<]minimax[>[</zip>",
            "]<]minimax[>[<primary>true]<]minimax[>[</primary>]<]minimax[>[</shipping>",
            "]<]minimax[>[</invoke>]<]minimax[>[</tool_call> Done."
        );
        for width in [1, 7, input.len()] {
            let chunks: Vec<_> = input
                .as_bytes()
                .chunks(width)
                .map(|chunk| std::str::from_utf8(chunk).unwrap())
                .collect();
            let out = parse_chunks(&tools, &chunks).coalesce_calls();
            assert_eq!(out.normal_text, "Preparing.  Done.");
            assert_eq!(out.calls.len(), 1);
            assert_eq!(out.calls[0].name.as_deref(), Some("create_order"));
            assert!(out.calls[0].complete);
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&out.calls[0].arguments).unwrap(),
                serde_json::json!({"shipping": {"zip": 18956, "primary": true}}),
                "width {width}"
            );
        }
    }

    fn string_tool(name: &str, parameters: &[&str]) -> Tool {
        let properties: serde_json::Map<_, _> = parameters
            .iter()
            .map(|parameter| (parameter.to_string(), serde_json::json!({"type": "string"})))
            .collect();
        Tool {
            name: name.to_string(),
            description: None,
            parameters: serde_json::json!({"type": "object", "properties": properties}),
            strict: None,
        }
    }

    fn write_file_tools() -> Vec<Tool> {
        vec![string_tool("write_file", &["path", "content"])]
    }

    fn stream_every_char(tools: &[Tool], input: &str) -> ToolParseResult {
        let mut parser = MiniMaxM3ToolStreamParser::new(tools);
        let mut out = ToolParseResult::default();
        for character in input.chars() {
            out.append(parser.push(&character.to_string()).expect("push"));
        }
        out.append(parser.finish().expect("finish"));
        out
    }

    fn push_every_char(parser: &mut MiniMaxM3ToolStreamParser, input: &str) -> ToolParseResult {
        let mut out = ToolParseResult::default();
        for character in input.chars() {
            out.append(parser.push(&character.to_string()).expect("push"));
        }
        out
    }

    fn joined_arguments(out: &ToolParseResult) -> String {
        out.calls
            .iter()
            .map(|call| call.arguments.as_str())
            .collect()
    }

    fn char_splits(input: &str) -> impl Iterator<Item = usize> + '_ {
        (0..=input.len()).filter(|&index| input.is_char_boundary(index))
    }

    #[test]
    fn streams_string_arguments_before_invoke_closes_at_every_char_boundary() {
        let input = concat!(
            "]<]minimax[>[<tool_call>\n]<]minimax[>[<invoke name=\"write_file\">\n",
            "]<]minimax[>[<path>reasoning_plan.txt]<]minimax[>[</path>\n",
            "]<]minimax[>[<content>\n## 发布计划\n依赖 \"core\" 与 C:\\path，",
            "回滚窗口 30 分钟，监控告警需要覆盖全部关键指标。]<]minimax[>[</content>\n",
            "]<]minimax[>[</invoke>\n]<]minimax[>[</tool_call>",
        );
        let baseline = parse_chunks(&write_file_tools(), &[input]).coalesce_calls();
        assert_eq!(baseline.calls.len(), 1);

        let invoke_close = input.find("]<]minimax[>[</invoke>").unwrap();
        let mut parser = MiniMaxM3ToolStreamParser::new(&write_file_tools());
        let before_close = push_every_char(&mut parser, &input[..invoke_close]);
        assert_eq!(
            before_close
                .calls
                .first()
                .and_then(|call| call.name.as_deref()),
            Some("write_file"),
            "name must arrive before the invoke closes: {before_close:?}"
        );

        let content_close = input.find("]<]minimax[>[</content>").unwrap();
        let mut parser = MiniMaxM3ToolStreamParser::new(&write_file_tools());
        let during_value = push_every_char(&mut parser, &input[..content_close - 30]);
        let emitted = joined_arguments(&during_value);
        assert!(
            emitted.starts_with(r#"{"path":"reasoning_plan.txt","content":"\n## 发布计划"#),
            "content must stream while the value is still open: {emitted}"
        );

        let out = stream_every_char(&write_file_tools(), input);
        assert_eq!(
            out.calls.iter().filter(|call| call.name.is_some()).count(),
            1,
            "only the first update carries the name"
        );
        assert!(out.calls.last().expect("call updates").complete);
        let fragments = out
            .calls
            .iter()
            .filter(|call| !call.arguments.is_empty())
            .count();
        assert!(
            fragments >= 10,
            "expected many argument fragments, got {fragments}"
        );
        assert!(!joined_arguments(&out).contains("minimax"), "{out:?}");
        assert_eq!(out.coalesce_calls(), baseline);

        for split in char_splits(input) {
            assert_eq!(
                parse_chunks(&write_file_tools(), &[&input[..split], &input[split..]])
                    .coalesce_calls(),
                baseline,
                "split {split}"
            );
        }
    }

    #[test]
    fn streams_each_parallel_invoke_before_it_closes() {
        let tools = vec![
            string_tool("save_background", &["content"]),
            string_tool("save_risks", &["content"]),
        ];
        let input = concat!(
            "]<]minimax[>[<tool_call>\n",
            "]<]minimax[>[<invoke name=\"save_background\">\n",
            "]<]minimax[>[<content>项目背景：统一平台。]<]minimax[>[</content>\n",
            "]<]minimax[>[</invoke>\n",
            "]<]minimax[>[<invoke name=\"save_risks\">\n",
            "]<]minimax[>[<content>主要风险：依赖延期与数据迁移失败。]<]minimax[>[</content>\n",
            "]<]minimax[>[</invoke>\n",
            "]<]minimax[>[</tool_call>",
        );
        let baseline = parse_chunks(&tools, &[input]).coalesce_calls();
        assert_eq!(baseline.calls.len(), 2);

        let mut parser = MiniMaxM3ToolStreamParser::new(&tools);
        let second_value_open = input.find("主要风险").unwrap() + "主要风险".len();
        let before = push_every_char(&mut parser, &input[..second_value_open]);
        let second: Vec<_> = before
            .calls
            .iter()
            .filter(|call| call.tool_index == 1)
            .collect();
        assert_eq!(
            second.first().and_then(|call| call.name.as_deref()),
            Some("save_risks")
        );
        assert!(
            second
                .iter()
                .map(|call| call.arguments.as_str())
                .collect::<String>()
                .contains("主要风险"),
            "second call must stream before it closes: {before:?}"
        );
        assert!(second.iter().all(|call| !call.complete));

        assert_eq!(stream_every_char(&tools, input).coalesce_calls(), baseline);
    }

    #[test]
    fn non_string_parameter_is_deferred_to_invoke_close() {
        let tools = vec![Tool {
            name: "save".to_string(),
            description: None,
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"title": {"type": "string"}, "count": {"type": "integer"}}
            }),
            strict: None,
        }];
        let string_first = concat!(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"save\">",
            "]<]minimax[>[<title>quarterly report]<]minimax[>[</title>",
            "]<]minimax[>[<count>42]<]minimax[>[</count>",
            "]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
        );
        let count_open = string_first.find("42").unwrap();
        let mut parser = MiniMaxM3ToolStreamParser::new(&tools);
        let mut out = push_every_char(&mut parser, &string_first[..count_open + 2]);
        assert_eq!(joined_arguments(&out), r#"{"title":"quarterly report""#);
        out.append(parser.push(&string_first[count_open + 2..]).unwrap());
        out.append(parser.finish().unwrap());
        assert_eq!(
            out.coalesce_calls().calls[0].arguments,
            r#"{"title":"quarterly report","count":42}"#
        );

        let count_first = concat!(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"save\">",
            "]<]minimax[>[<count>42]<]minimax[>[</count>",
            "]<]minimax[>[<title>quarterly report]<]minimax[>[</title>",
            "]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
        );
        let invoke_close = count_first.find("]<]minimax[>[</invoke>").unwrap();
        let mut parser = MiniMaxM3ToolStreamParser::new(&tools);
        assert!(
            push_every_char(&mut parser, &count_first[..invoke_close])
                .calls
                .is_empty()
        );
        let closed = parser.push(&count_first[invoke_close..]).unwrap();
        assert_eq!(closed.calls.len(), 1);
        assert_eq!(
            closed.calls[0].arguments,
            r#"{"count":42,"title":"quarterly report"}"#
        );
    }

    #[test]
    fn entities_decode_like_the_batch_parser_at_every_split() {
        let input = concat!(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"get_weather\">",
            "]<]minimax[>[<location>a &amp; b &lt;x&gt; &amp;quot; &amp;lt; &#39;z&#x27; &am",
            " tail that keeps streaming]<]minimax[>[</location>",
            "]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
        );
        let baseline = parse_chunks(&weather_tools(), &[input]).coalesce_calls();
        assert_eq!(baseline.calls.len(), 1);
        assert_eq!(
            stream_every_char(&weather_tools(), input).coalesce_calls(),
            baseline
        );
        for split in char_splits(input) {
            assert_eq!(
                parse_chunks(&weather_tools(), &[&input[..split], &input[split..]])
                    .coalesce_calls(),
                baseline,
                "split {split}"
            );
        }
    }

    #[test]
    fn nested_tags_in_a_string_parameter_never_stream_a_mismatched_prefix() {
        let input = concat!(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"get_weather\">",
            "]<]minimax[>[<location>\n  ]<]minimax[>[<city>Paris]<]minimax[>[</city>\n]<]minimax[>[</location>",
            "]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
        );
        let baseline = parse_chunks(&weather_tools(), &[input]).coalesce_calls();
        assert_eq!(baseline.calls.len(), 1);
        assert_eq!(
            stream_every_char(&weather_tools(), input).coalesce_calls(),
            baseline
        );
    }

    #[test]
    fn duplicate_streamed_parameter_does_not_assemble_corrupted_arguments() {
        let input = concat!(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"get_weather\">",
            "]<]minimax[>[<location>Paris]<]minimax[>[</location>",
            "]<]minimax[>[<location>London]<]minimax[>[</location>",
            "]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
        );
        let first_value = input.find("Paris").unwrap() + "Paris".len();
        let out = parse_chunks(
            &weather_tools(),
            &[&input[..first_value], &input[first_value..]],
        );
        assert!(
            out.clone().coalesce_calls().calls.is_empty(),
            "streamed duplicate parameter assembled mismatched JSON: {out:?}"
        );
        let complete = parse_chunks(&weather_tools(), &[input]).coalesce_calls();
        assert_eq!(
            complete.calls[0].arguments,
            r#"{"location":["Paris","London"]}"#
        );
    }

    #[test]
    fn unknown_tool_is_emitted_only_at_invoke_close() {
        let input = concat!(
            "]<]minimax[>[<tool_call>]<]minimax[>[<invoke name=\"get_time\">",
            "]<]minimax[>[<zone>UTC]<]minimax[>[</zone>",
            "]<]minimax[>[</invoke>]<]minimax[>[</tool_call>",
        );
        let invoke_close = input.find("]<]minimax[>[</invoke>").unwrap();
        let mut parser = MiniMaxM3ToolStreamParser::new(&weather_tools());
        assert!(
            push_every_char(&mut parser, &input[..invoke_close])
                .calls
                .is_empty()
        );
        let closed = parser.push(&input[invoke_close..]).unwrap();
        assert_eq!(closed.calls.len(), 1);
        assert_eq!(closed.calls[0].arguments, r#"{"zone":"UTC"}"#);
    }
}
