package chat_completions

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

// TranslateOpenAIStreamToQoder translates an OpenAI Chat Completions streaming response chunk
// into Qoder SSE format: data: {"event":"EVENT_TYPE","data":{...}}.
func TranslateOpenAIStreamToQoder(_ context.Context, modelName string, _, _, rawJSON []byte, param *any) [][]byte {
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
	if bytes.Equal(payload, []byte("[DONE]")) {
		return [][]byte{[]byte("data: {\"event\":\"message_stop\",\"data\":{}}")}
	}

	root := gjson.ParseBytes(payload)

	if id := root.Get("id"); id.Exists() && state.ResponseID == "" {
		state.ResponseID = id.String()
	}
	if created := root.Get("created"); created.Exists() && state.Created == 0 {
		state.Created = created.Int()
	}

	choices := root.Get("choices")
	if !choices.IsArray() || len(choices.Array()) == 0 {
		return [][]byte{}
	}
	choice := choices.Array()[0]
	delta := choice.Get("delta")
	finishReason := choice.Get("finish_reason")

	var results [][]byte

	if state.ResponseID != "" && state.Created > 0 {
		startEvent := []byte(`{"event":"message_start","data":{"id":"","model":"","created":0}}`)
		startEvent, _ = sjson.SetBytes(startEvent, "data.id", state.ResponseID)
		if m := root.Get("model"); m.Exists() {
			startEvent, _ = sjson.SetBytes(startEvent, "data.model", m.String())
		}
		startEvent, _ = sjson.SetBytes(startEvent, "data.created", state.Created)
		results = append(results, wrapQoderSSE(startEvent))
		state.ResponseID = ""
	}

	if content := delta.Get("content"); content.Exists() && content.String() != "" {
		event := []byte(`{"event":"content_block_delta","data":{"text":""}}`)
		event, _ = sjson.SetBytes(event, "data.text", content.String())
		results = append(results, wrapQoderSSE(event))
	}

	if toolCalls := delta.Get("tool_calls"); toolCalls.IsArray() {
		tcArr := toolCalls.Array()
		for i := 0; i < len(tcArr); i++ {
			tc := tcArr[i]
			if id := tc.Get("id"); id.Exists() && id.String() != "" {
				state.ToolIndex++
				event := []byte(`{"event":"tool_use_start","data":{"id":"","name":""}}`)
				event, _ = sjson.SetBytes(event, "data.id", id.String())
				event, _ = sjson.SetBytes(event, "data.name", tc.Get("function.name").String())
				results = append(results, wrapQoderSSE(event))
			}
			if args := tc.Get("function.arguments"); args.Exists() && args.String() != "" {
				event := []byte(`{"event":"tool_use_delta","data":{"arguments":""}}`)
				event, _ = sjson.SetBytes(event, "data.arguments", args.String())
				results = append(results, wrapQoderSSE(event))
			}
		}
	}

	if finishReason.Exists() && finishReason.Value() != nil {
		event := []byte(`{"event":"message_delta","data":{"stop_reason":"","usage":{}}}`)
		fr := finishReason.String()
		if fr == "tool_calls" {
			fr = "tool_use"
		}
		event, _ = sjson.SetBytes(event, "data.stop_reason", fr)
		if usage := root.Get("usage"); usage.Exists() {
			if v := usage.Get("prompt_tokens"); v.Exists() {
				event, _ = sjson.SetBytes(event, "data.usage.input_tokens", v.Int())
			}
			if v := usage.Get("completion_tokens"); v.Exists() {
				event, _ = sjson.SetBytes(event, "data.usage.output_tokens", v.Int())
			}
			if v := usage.Get("total_tokens"); v.Exists() {
				event, _ = sjson.SetBytes(event, "data.usage.total_tokens", v.Int())
			}
		}
		results = append(results, wrapQoderSSE(event))
	}

	return results
}

// TranslateOpenAIResponseToQoder converts a non-streaming OpenAI Chat Completions response
// into Qoder format.
func TranslateOpenAIResponseToQoder(_ context.Context, _ string, _, _, rawJSON []byte, _ *any) []byte {
	root := gjson.ParseBytes(rawJSON)

	out := []byte(`{}`)

	if v := root.Get("id"); v.Exists() {
		out, _ = sjson.SetBytes(out, "id", v.String())
	}
	if v := root.Get("model"); v.Exists() {
		out, _ = sjson.SetBytes(out, "model", v.String())
	}
	if v := root.Get("created"); v.Exists() {
		out, _ = sjson.SetBytes(out, "created", v.Int())
	}

	if usage := root.Get("usage"); usage.Exists() {
		outUsage := []byte(`{}`)
		if v := usage.Get("prompt_tokens"); v.Exists() {
			outUsage, _ = sjson.SetBytes(outUsage, "input_tokens", v.Int())
		}
		if v := usage.Get("completion_tokens"); v.Exists() {
			outUsage, _ = sjson.SetBytes(outUsage, "output_tokens", v.Int())
		}
		if v := usage.Get("total_tokens"); v.Exists() {
			outUsage, _ = sjson.SetBytes(outUsage, "total_tokens", v.Int())
		}
		out, _ = sjson.SetRawBytes(out, "usage", outUsage)
	}

	out, _ = sjson.SetRawBytes(out, "output", []byte(`[]`))
	choices := root.Get("choices")
	if choices.IsArray() {
		for _, choice := range choices.Array() {
			msg := choice.Get("message")
			if !msg.Exists() {
				continue
			}

			outputItem := []byte(`{"type":"message","content":[]}`)
			outputItem, _ = sjson.SetBytes(outputItem, "role", msg.Get("role").String())

			if content := msg.Get("content"); content.Exists() && content.Value() != nil {
				textPart := []byte(`{"type":"text","text":""}`)
				textPart, _ = sjson.SetBytes(textPart, "text", content.String())
				outputItem, _ = sjson.SetRawBytes(outputItem, "content.-1", textPart)
			}

			if toolCalls := msg.Get("tool_calls"); toolCalls.IsArray() {
				for _, tc := range toolCalls.Array() {
					funcItem := []byte(`{"type":"function_call","id":"","name":"","arguments":""}`)
					funcItem, _ = sjson.SetBytes(funcItem, "id", tc.Get("id").String())
					funcItem, _ = sjson.SetBytes(funcItem, "name", tc.Get("function.name").String())
					funcItem, _ = sjson.SetBytes(funcItem, "arguments", tc.Get("function.arguments").String())
					out, _ = sjson.SetRawBytes(out, "output.-1", funcItem)
				}
			}

			out, _ = sjson.SetRawBytes(out, "output.-1", outputItem)
		}
	}

	if choices.IsArray() && len(choices.Array()) > 0 {
		fr := choices.Array()[0].Get("finish_reason").String()
		if fr == "tool_calls" {
			fr = "tool_use"
		}
		out, _ = sjson.SetBytes(out, "stop_reason", fr)
	}

	return out
}

func wrapQoderSSE(eventJSON []byte) []byte {
	result := make([]byte, 0, len(eventJSON)+6)
	result = append(result, "data: "...)
	result = append(result, eventJSON...)
	return result
}
