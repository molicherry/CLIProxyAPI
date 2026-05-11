package qoder

import (
	"bytes"
	"context"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var (
	qoderDataTag  = []byte("data:")
	qoderDoneLine = []byte("data: [DONE]")
)

type qoderStreamState struct {
	ResponseID string
	Created    int64
	Model      string
	ToolIndex  int
}

// TranslateQoderStreamToOpenAI translates a single Qoder SSE chunk into OpenAI Chat Completions
// streaming format. Qoder wraps events as: data: {"event":"EVENT_TYPE","data":{...}}.
//
// Supported event types:
//   - content_block_delta → text content delta
//   - tool_use_start → tool call announcement
//   - tool_use_delta → tool call arguments delta
//   - message_delta → finish reason and usage
//   - message_stop → [DONE]
//   - error → error chunk
func TranslateQoderStreamToOpenAI(_ context.Context, modelName string, _, _, rawJSON []byte, param *any) [][]byte {
	if *param == nil {
		*param = &qoderStreamState{
			Model:     modelName,
			ToolIndex: -1,
		}
	}
	state := (*param).(*qoderStreamState)

	if !bytes.HasPrefix(rawJSON, qoderDataTag) {
		return [][]byte{}
	}

	payload := bytes.TrimSpace(rawJSON[5:])

	if bytes.Equal(payload, []byte("[DONE]")) || bytes.Equal(payload, qoderDoneLine[5:]) {
		return [][]byte{qoderDoneLine}
	}

	root := gjson.ParseBytes(payload)
	eventType := root.Get("event").String()
	data := root.Get("data")

	template := []byte(`{"id":"","object":"chat.completion.chunk","created":0,"model":"","choices":[{"index":0,"delta":{},"finish_reason":null}]}`)

	if state.ResponseID != "" {
		template, _ = sjson.SetBytes(template, "id", state.ResponseID)
	}
	if state.Created > 0 {
		template, _ = sjson.SetBytes(template, "created", state.Created)
	}
	if m := data.Get("model"); m.Exists() {
		template, _ = sjson.SetBytes(template, "model", m.String())
		state.Model = m.String()
	} else if state.Model != "" {
		template, _ = sjson.SetBytes(template, "model", state.Model)
	}

	switch eventType {
	case "message_start":
		state.ResponseID = data.Get("id").String()
		state.Created = data.Get("created").Int()
		if m := data.Get("model"); m.Exists() {
			state.Model = m.String()
		}
		return [][]byte{}

	case "content_block_delta":
		text := data.Get("text").String()
		if text == "" {
			text = data.Get("delta.text").String()
		}
		if text == "" {
			return [][]byte{}
		}
		template, _ = sjson.SetBytes(template, "choices.0.delta.role", "assistant")
		template, _ = sjson.SetBytes(template, "choices.0.delta.content", text)

	case "tool_use_start":
		state.ToolIndex++
		toolID := data.Get("id").String()
		toolName := data.Get("name").String()
		funcCall := []byte(`{"index":0,"id":"","type":"function","function":{"name":"","arguments":""}}`)
		funcCall, _ = sjson.SetBytes(funcCall, "index", state.ToolIndex)
		funcCall, _ = sjson.SetBytes(funcCall, "id", toolID)
		funcCall, _ = sjson.SetBytes(funcCall, "function.name", toolName)
		template, _ = sjson.SetBytes(template, "choices.0.delta.role", "assistant")
		template, _ = sjson.SetRawBytes(template, "choices.0.delta.tool_calls", []byte(`[]`))
		template, _ = sjson.SetRawBytes(template, "choices.0.delta.tool_calls.-1", funcCall)

	case "tool_use_delta":
		argsDelta := data.Get("arguments").String()
		if argsDelta == "" {
			argsDelta = data.Get("delta.arguments").String()
		}
		funcCall := []byte(`{"index":0,"function":{"arguments":""}}`)
		funcCall, _ = sjson.SetBytes(funcCall, "index", state.ToolIndex)
		funcCall, _ = sjson.SetBytes(funcCall, "function.arguments", argsDelta)
		template, _ = sjson.SetRawBytes(template, "choices.0.delta.tool_calls", []byte(`[]`))
		template, _ = sjson.SetRawBytes(template, "choices.0.delta.tool_calls.-1", funcCall)

	case "message_delta":
		finishReason := data.Get("stop_reason").String()
		if finishReason == "" {
			finishReason = data.Get("finish_reason").String()
		}
		if finishReason == "tool_use" {
			finishReason = "tool_calls"
		} else if finishReason == "end_turn" || finishReason == "stop" {
			finishReason = "stop"
		}
		if finishReason != "" {
			template, _ = sjson.SetBytes(template, "choices.0.finish_reason", finishReason)
		}
		if usage := data.Get("usage"); usage.Exists() {
			if v := usage.Get("input_tokens"); v.Exists() {
				template, _ = sjson.SetBytes(template, "usage.prompt_tokens", v.Int())
			}
			if v := usage.Get("output_tokens"); v.Exists() {
				template, _ = sjson.SetBytes(template, "usage.completion_tokens", v.Int())
			}
			if v := usage.Get("total_tokens"); v.Exists() {
				template, _ = sjson.SetBytes(template, "usage.total_tokens", v.Int())
			}
		}

	case "message_stop":
		return [][]byte{qoderDoneLine}

	case "error":
		errMsg := data.Get("message").String()
		if errMsg == "" {
			errMsg = data.Raw
		}
		errChunk := []byte(`{"error":{"message":"","type":""}}`)
		errChunk, _ = sjson.SetBytes(errChunk, "error.message", errMsg)
		errChunk, _ = sjson.SetBytes(errChunk, "error.type", "server_error")
		return [][]byte{errChunk}

	default:
		return [][]byte{}
	}

	return [][]byte{template}
}

// TranslateQoderResponseToOpenAI converts a non-streaming Qoder response into OpenAI
// Chat Completions response format.
func TranslateQoderResponseToOpenAI(_ context.Context, _ string, _, _, rawJSON []byte, _ *any) []byte {
	root := gjson.ParseBytes(rawJSON)

	template := []byte(`{"id":"","object":"chat.completion","created":0,"model":"","choices":[{"index":0,"message":{"role":"assistant","content":null},"finish_reason":null}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`)

	if v := root.Get("id"); v.Exists() {
		template, _ = sjson.SetBytes(template, "id", v.String())
	}
	if v := root.Get("model"); v.Exists() {
		template, _ = sjson.SetBytes(template, "model", v.String())
	}
	if v := root.Get("created"); v.Exists() {
		template, _ = sjson.SetBytes(template, "created", v.Int())
	}
	if v := root.Get("created_at"); v.Exists() {
		template, _ = sjson.SetBytes(template, "created", v.Int())
	}

	if usage := root.Get("usage"); usage.Exists() {
		if v := usage.Get("input_tokens"); v.Exists() {
			template, _ = sjson.SetBytes(template, "usage.prompt_tokens", v.Int())
		}
		if v := usage.Get("prompt_tokens"); v.Exists() {
			template, _ = sjson.SetBytes(template, "usage.prompt_tokens", v.Int())
		}
		if v := usage.Get("output_tokens"); v.Exists() {
			template, _ = sjson.SetBytes(template, "usage.completion_tokens", v.Int())
		}
		if v := usage.Get("completion_tokens"); v.Exists() {
			template, _ = sjson.SetBytes(template, "usage.completion_tokens", v.Int())
		}
		if v := usage.Get("total_tokens"); v.Exists() {
			template, _ = sjson.SetBytes(template, "usage.total_tokens", v.Int())
		}
	}

	var contentText string
	var toolCalls [][]byte

	output := root.Get("output")
	if output.IsArray() {
		arr := output.Array()
		for i := 0; i < len(arr); i++ {
			item := arr[i]
			switch item.Get("type").String() {
			case "message":
				if content := item.Get("content"); content.IsArray() {
					for _, part := range content.Array() {
						if part.Get("type").String() == "text" || part.Get("type").String() == "output_text" {
							if contentText != "" {
								contentText += "\n"
							}
							contentText += part.Get("text").String()
						}
					}
				}
			case "text", "output_text":
				if t := item.Get("text").String(); t != "" {
					if contentText != "" {
						contentText += "\n"
					}
					contentText += t
				}
			case "function_call", "tool_use":
				funcCall := []byte(`{"id":"","type":"function","function":{"name":"","arguments":""}}`)
				funcCall, _ = sjson.SetBytes(funcCall, "id", item.Get("id").String())
				funcCall, _ = sjson.SetBytes(funcCall, "function.name", item.Get("name").String())
				funcCall, _ = sjson.SetBytes(funcCall, "function.arguments", item.Get("arguments").String())
				toolCalls = append(toolCalls, funcCall)
			}
		}
	}

	if contentText == "" {
		if c := root.Get("content"); c.Exists() {
			if c.Type == gjson.String {
				contentText = c.String()
			} else if c.IsArray() {
				for _, part := range c.Array() {
					if part.Get("type").String() == "text" {
						if contentText != "" {
							contentText += "\n"
						}
						contentText += part.Get("text").String()
					}
				}
			}
		}
	}

	if contentText != "" {
		template, _ = sjson.SetBytes(template, "choices.0.message.content", contentText)
	}
	if len(toolCalls) > 0 {
		template, _ = sjson.SetRawBytes(template, "choices.0.message.tool_calls", []byte(`[]`))
		for _, tc := range toolCalls {
			template, _ = sjson.SetRawBytes(template, "choices.0.message.tool_calls.-1", tc)
		}
	}

	finishReason := "stop"
	if v := root.Get("stop_reason"); v.Exists() {
		fr := v.String()
		if fr == "tool_use" {
			finishReason = "tool_calls"
		}
	}
	if v := root.Get("finish_reason"); v.Exists() {
		fr := v.String()
		if fr == "tool_use" || fr == "tool_calls" {
			finishReason = "tool_calls"
		} else if fr != "" {
			finishReason = fr
		}
	}
	if len(toolCalls) > 0 {
		finishReason = "tool_calls"
	}
	template, _ = sjson.SetBytes(template, "choices.0.finish_reason", finishReason)

	return template
}
