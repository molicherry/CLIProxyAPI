package chat_completions

import (
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

// ConvertQoderRequestToOpenAI converts a Qoder-format request JSON into OpenAI Chat Completions
// format. It restores system_prompt into system messages, converts content arrays, and renames
// tools[].input_schema back to tools[].parameters.
func ConvertQoderRequestToOpenAI(modelName string, rawJSON []byte, stream bool) []byte {
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

	if sp := gjson.GetBytes(rawJSON, "system_prompt"); sp.Exists() && sp.String() != "" {
		sysMsg := []byte(`{"role":"system","content":""}`)
		sysMsg, _ = sjson.SetBytes(sysMsg, "content", sp.String())
		out, _ = sjson.SetRawBytes(out, "messages.-1", sysMsg)
	}

	messages := gjson.GetBytes(rawJSON, "messages")
	if messages.IsArray() {
		arr := messages.Array()
		for i := 0; i < len(arr); i++ {
			m := arr[i]
			role := m.Get("role").String()
			msg := []byte(`{"role":"","content":[]}`)
			msg, _ = sjson.SetBytes(msg, "role", role)

			content := m.Get("content")
			if content.IsArray() {
				items := content.Array()
				for j := 0; j < len(items); j++ {
					it := items[j]
					t := it.Get("type").String()
					switch t {
					case "text":
						part := []byte(`{"type":"text","text":""}`)
						part, _ = sjson.SetBytes(part, "text", it.Get("text").String())
						msg, _ = sjson.SetRawBytes(msg, "content.-1", part)
					case "image":
						url := it.Get("source.url").String()
						if url != "" {
							part := []byte(`{"type":"image_url","image_url":{"url":""}}`)
							part, _ = sjson.SetBytes(part, "image_url.url", url)
							msg, _ = sjson.SetRawBytes(msg, "content.-1", part)
						}
					case "tool_use":
						if role == "assistant" {
							tc := []byte(`{"id":"","type":"function","function":{"name":"","arguments":""}}`)
							tc, _ = sjson.SetBytes(tc, "id", it.Get("id").String())
							tc, _ = sjson.SetBytes(tc, "function.name", it.Get("name").String())
							input := it.Get("input")
							if input.Exists() && input.IsObject() {
								tc, _ = sjson.SetBytes(tc, "function.arguments", input.Raw)
							}
							msg, _ = sjson.SetRawBytes(msg, "tool_calls.-1", tc)
						}
					case "tool_result":
						continue
					}
				}
			} else if content.Exists() && content.Type == gjson.String && content.String() != "" {
				part := []byte(`{"type":"text","text":""}`)
				part, _ = sjson.SetBytes(part, "text", content.String())
				msg, _ = sjson.SetRawBytes(msg, "content.-1", part)
			}

			out, _ = sjson.SetRawBytes(out, "messages.-1", msg)

			if content.IsArray() {
				items := content.Array()
				for j := 0; j < len(items); j++ {
					it := items[j]
					if it.Get("type").String() == "tool_result" {
						toolMsg := []byte(`{"role":"tool","tool_call_id":"","content":""}`)
						toolMsg, _ = sjson.SetBytes(toolMsg, "tool_call_id", it.Get("tool_use_id").String())
						toolMsg, _ = sjson.SetBytes(toolMsg, "content", it.Get("content").String())
						out, _ = sjson.SetRawBytes(out, "messages.-1", toolMsg)
					}
				}
			}
		}
	}

	tools := gjson.GetBytes(rawJSON, "tools")
	if tools.IsArray() && len(tools.Array()) > 0 {
		out, _ = sjson.SetRawBytes(out, "tools", []byte(`[]`))
		arr := tools.Array()
		for i := 0; i < len(arr); i++ {
			t := arr[i]
			item := []byte(`{"type":"function","function":{"name":"","description":"","parameters":{}}}`)
			item, _ = sjson.SetBytes(item, "function.name", t.Get("name").String())
			if v := t.Get("description"); v.Exists() {
				item, _ = sjson.SetBytes(item, "function.description", v.String())
			}
			if v := t.Get("input_schema"); v.Exists() {
				item, _ = sjson.SetRawBytes(item, "function.parameters", []byte(v.Raw))
			} else if v := t.Get("parameters"); v.Exists() {
				item, _ = sjson.SetRawBytes(item, "function.parameters", []byte(v.Raw))
			}
			out, _ = sjson.SetRawBytes(out, "tools.-1", item)
		}
	}

	if tc := gjson.GetBytes(rawJSON, "tool_choice"); tc.Exists() {
		switch tc.Type {
		case gjson.String:
			out, _ = sjson.SetBytes(out, "tool_choice", tc.String())
		case gjson.JSON:
			if tc.Get("type").String() == "function" {
				name := tc.Get("name").String()
				choice := []byte(`{"type":"function","function":{"name":""}}`)
				choice, _ = sjson.SetBytes(choice, "function.name", name)
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
