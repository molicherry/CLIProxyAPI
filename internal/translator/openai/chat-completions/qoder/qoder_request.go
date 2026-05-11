// Package qoder translates between OpenAI Chat Completions and Qoder private API formats.
package qoder

import (
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ConvertOpenAIRequestToQoder converts an OpenAI Chat Completions request JSON into Qoder format.
// System messages are extracted into a top-level "system_prompt" field, message content is converted
// to [{"type":"text","text":"..."}] array format, and tools[].parameters is renamed to tools[].input_schema.
//
// Parameters:
//   - modelName: The model to use
//   - rawJSON: The raw OpenAI request JSON
//   - stream: Whether streaming is requested
//
// Returns the transformed request in Qoder format.
func ConvertOpenAIRequestToQoder(modelName string, rawJSON []byte, stream bool) []byte {
	out := []byte(`{}`)

	out, _ = sjson.SetBytes(out, "model", modelName)
	out, _ = sjson.SetBytes(out, "stream", stream)

	if v := gjson.GetBytes(rawJSON, "max_tokens"); v.Exists() {
		out, _ = sjson.SetBytes(out, "max_tokens", v.Value())
	}
	if v := gjson.GetBytes(rawJSON, "max_completion_tokens"); v.Exists() {
		out, _ = sjson.SetBytes(out, "max_completion_tokens", v.Value())
	}
	if v := gjson.GetBytes(rawJSON, "temperature"); v.Exists() {
		out, _ = sjson.SetBytes(out, "temperature", v.Value())
	}
	if v := gjson.GetBytes(rawJSON, "top_p"); v.Exists() {
		out, _ = sjson.SetBytes(out, "top_p", v.Value())
	}
	if v := gjson.GetBytes(rawJSON, "stop"); v.Exists() {
		out, _ = sjson.SetRawBytes(out, "stop", []byte(v.Raw))
	}

	out, _ = sjson.SetRawBytes(out, "messages", []byte(`[]`))
	messages := gjson.GetBytes(rawJSON, "messages")
	if messages.IsArray() {
		arr := messages.Array()
		for i := 0; i < len(arr); i++ {
			m := arr[i]
			role := m.Get("role").String()

			if role == "system" {
				c := m.Get("content")
				systemText := extractTextContent(c)
				if systemText != "" {
					existing := gjson.GetBytes(out, "system_prompt").String()
					if existing != "" {
						existing += "\n" + systemText
					} else {
						existing = systemText
					}
					out, _ = sjson.SetBytes(out, "system_prompt", existing)
				}
				continue
			}

			msg := []byte(`{"role":"","content":[]}`)
			msg, _ = sjson.SetBytes(msg, "role", role)

			c := m.Get("content")
			if c.Exists() && c.Type == gjson.String && c.String() != "" {
				part := []byte(`{"type":"text","text":""}`)
				part, _ = sjson.SetBytes(part, "text", c.String())
				msg, _ = sjson.SetRawBytes(msg, "content.-1", part)
			} else if c.Exists() && c.IsArray() {
				items := c.Array()
				for j := 0; j < len(items); j++ {
					it := items[j]
					part := convertContentPartToQoder(it)
					if part != nil {
						msg, _ = sjson.SetRawBytes(msg, "content.-1", part)
					}
				}
			}

			if role == "assistant" {
				toolCalls := m.Get("tool_calls")
				if toolCalls.Exists() && toolCalls.IsArray() {
					tcArr := toolCalls.Array()
					for j := 0; j < len(tcArr); j++ {
						tc := tcArr[j]
						if tc.Get("type").String() == "function" {
							toolUse := []byte(`{"type":"tool_use","id":"","name":"","input":{}}`)
							toolUse, _ = sjson.SetBytes(toolUse, "id", tc.Get("id").String())
							toolUse, _ = sjson.SetBytes(toolUse, "name", tc.Get("function.name").String())
							argsStr := tc.Get("function.arguments").String()
							if argsStr != "" && gjson.Valid(argsStr) {
								toolUse, _ = sjson.SetRawBytes(toolUse, "input", []byte(argsStr))
							}
							msg, _ = sjson.SetRawBytes(msg, "content.-1", toolUse)
						}
					}
				}
			}

			if role == "tool" {
				toolCallID := m.Get("tool_call_id").String()
				toolResult := []byte(`{"type":"tool_result","tool_use_id":"","content":""}`)
				toolResult, _ = sjson.SetBytes(toolResult, "tool_use_id", toolCallID)
				toolContent := m.Get("content")
				resultText := extractTextContent(toolContent)
				toolResult, _ = sjson.SetBytes(toolResult, "content", resultText)
				msg, _ = sjson.SetRawBytes(msg, "content.-1", toolResult)
			}

			out, _ = sjson.SetRawBytes(out, "messages.-1", msg)
		}
	}

	tools := gjson.GetBytes(rawJSON, "tools")
	if tools.IsArray() && len(tools.Array()) > 0 {
		out, _ = sjson.SetRawBytes(out, "tools", []byte(`[]`))
		arr := tools.Array()
		for i := 0; i < len(arr); i++ {
			t := arr[i]
			if t.Get("type").String() == "function" {
				fn := t.Get("function")
				if !fn.Exists() {
					continue
				}
				item := []byte(`{"name":"","description":"","input_schema":{}}`)
				item, _ = sjson.SetBytes(item, "name", fn.Get("name").String())
				if v := fn.Get("description"); v.Exists() {
					item, _ = sjson.SetBytes(item, "description", v.String())
				}
				if v := fn.Get("parameters"); v.Exists() {
					item, _ = sjson.SetRawBytes(item, "input_schema", []byte(v.Raw))
				}
				out, _ = sjson.SetRawBytes(out, "tools.-1", item)
			}
		}
	}

	if tc := gjson.GetBytes(rawJSON, "tool_choice"); tc.Exists() {
		switch tc.Type {
		case gjson.String:
			out, _ = sjson.SetBytes(out, "tool_choice", tc.String())
		case gjson.JSON:
			if tc.Get("type").String() == "function" {
				name := tc.Get("function.name").String()
				choice := []byte(`{"type":"function","name":""}`)
				choice, _ = sjson.SetBytes(choice, "name", name)
				out, _ = sjson.SetRawBytes(out, "tool_choice", choice)
			} else {
				out, _ = sjson.SetRawBytes(out, "tool_choice", []byte(tc.Raw))
			}
		}
	}

	if rf := gjson.GetBytes(rawJSON, "response_format"); rf.Exists() {
		out, _ = sjson.SetRawBytes(out, "response_format", []byte(rf.Raw))
	}

	return out
}

func extractTextContent(c gjson.Result) string {
	if !c.Exists() {
		return ""
	}
	if c.Type == gjson.String {
		return c.String()
	}
	if c.IsArray() {
		var text string
		arr := c.Array()
		for i := 0; i < len(arr); i++ {
			it := arr[i]
			if it.Get("type").String() == "text" {
				if text != "" {
					text += "\n"
				}
				text += it.Get("text").String()
			}
		}
		return text
	}
	return c.String()
}

func convertContentPartToQoder(part gjson.Result) []byte {
	t := part.Get("type").String()
	switch t {
	case "text":
		item := []byte(`{"type":"text","text":""}`)
		item, _ = sjson.SetBytes(item, "text", part.Get("text").String())
		return item
	case "image_url":
		url := part.Get("image_url.url").String()
		if url == "" {
			return nil
		}
		item := []byte(`{"type":"image","source":{"type":"url","url":""}}`)
		item, _ = sjson.SetBytes(item, "source.url", url)
		return item
	}
	return nil
}
