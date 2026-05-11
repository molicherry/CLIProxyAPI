package executor

import (
	"bufio"
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	coreusage "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/usage"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
)

const qoderDefaultBaseURL = "https://openapi.qoder.sh"

// QoderExecutor implements ProviderExecutor for the Qoder API.
// It converts OpenAI-format payloads to Qoder's proprietary request format,
// sends them to the Qoder chat completions endpoint, and translates
// Qoder SSE events back into OpenAI-compatible streaming chunks.
type QoderExecutor struct {
	cfg *config.Config
}

// NewQoderExecutor creates an executor bound to the Qoder provider.
func NewQoderExecutor(cfg *config.Config) *QoderExecutor {
	return &QoderExecutor{cfg: cfg}
}

// Identifier implements cliproxyauth.ProviderExecutor.
func (e *QoderExecutor) Identifier() string { return "qoder" }

// PrepareRequest injects Qoder credentials and required headers into the outgoing HTTP request.
func (e *QoderExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	_, apiKey := e.resolveCredentials(auth)
	if strings.TrimSpace(apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	e.applyQoderHeaders(req)
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(req, attrs)
	return nil
}

// HttpRequest injects Qoder credentials into the request and executes it.
func (e *QoderExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("qoder executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(httpReq)
}

// Execute performs a non-streaming chat completion request to Qoder.
func (e *QoderExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	baseURL, apiKey := e.resolveCredentials(auth)
	if baseURL == "" {
		baseURL = qoderDefaultBaseURL
	}

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")
	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, false)
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, false)

	translated, err = thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return resp, err
	}

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	translated = helps.ApplyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", translated, originalTranslated, requestedModel, requestPath)

	qoderBody, err := convertOpenAIToQoder(translated, false)
	if err != nil {
		return resp, fmt.Errorf("qoder executor: payload conversion failed: %w", err)
	}

	url := strings.TrimSuffix(baseURL, "/") + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(qoderBody))
	if err != nil {
		return resp, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	}
	e.applyQoderHeaders(httpReq)
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      qoderBody,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("qoder executor: close response body error: %v", errClose)
		}
	}()
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		helps.AppendAPIResponseChunk(ctx, e.cfg, b)
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return resp, err
	}
	body, err := io.ReadAll(httpResp.Body)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	helps.AppendAPIResponseChunk(ctx, e.cfg, body)

	openaiBody := convertQoderResponseToOpenAI(body)
	reporter.Publish(ctx, helps.ParseOpenAIUsage(openaiBody))
	reporter.EnsurePublished(ctx)

	var param any
	out := sdktranslator.TranslateNonStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, openaiBody, &param)
	resp = cliproxyexecutor.Response{Payload: out, Headers: httpResp.Header.Clone()}
	return resp, nil
}

// ExecuteStream performs a streaming chat completion request to Qoder.
func (e *QoderExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	reporter := helps.NewUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	baseURL, apiKey := e.resolveCredentials(auth)
	if baseURL == "" {
		baseURL = qoderDefaultBaseURL
	}

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")
	originalPayloadSource := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayloadSource = opts.OriginalRequest
	}
	originalPayload := originalPayloadSource
	originalTranslated := sdktranslator.TranslateRequest(from, to, baseModel, originalPayload, true)
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, true)

	translated, err = thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return nil, err
	}

	requestedModel := helps.PayloadRequestedModel(opts, req.Model)
	requestPath := helps.PayloadRequestPath(opts)
	translated = helps.ApplyPayloadConfigWithRoot(e.cfg, baseModel, to.String(), "", translated, originalTranslated, requestedModel, requestPath)

	qoderBody, err := convertOpenAIToQoder(translated, true)
	if err != nil {
		return nil, fmt.Errorf("qoder executor: payload conversion failed: %w", err)
	}

	url := strings.TrimSuffix(baseURL, "/") + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(qoderBody))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("Cache-Control", "no-cache")
	if apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	}
	e.applyQoderHeaders(httpReq)
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      qoderBody,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return nil, err
	}
	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		helps.AppendAPIResponseChunk(ctx, e.cfg, b)
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("qoder executor: close response body error: %v", errClose)
		}
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return nil, err
	}
	out := make(chan cliproxyexecutor.StreamChunk)
	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("qoder executor: close response body error: %v", errClose)
			}
		}()

		var (
			toolCallIndex int
			toolCallID    string
			toolCallName  string
			param         any
		)

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, 52_428_800)
		for scanner.Scan() {
			line := scanner.Bytes()
			helps.AppendAPIResponseChunk(ctx, e.cfg, line)
			trimmedLine := bytes.TrimSpace(line)
			if len(trimmedLine) == 0 {
				continue
			}
			if !bytes.HasPrefix(trimmedLine, []byte("data:")) {
				if bytes.HasPrefix(trimmedLine, []byte(":")) || bytes.HasPrefix(trimmedLine, []byte("event:")) ||
					bytes.HasPrefix(trimmedLine, []byte("id:")) || bytes.HasPrefix(trimmedLine, []byte("retry:")) {
					continue
				}
				if bytes.HasPrefix(trimmedLine, []byte("{")) || bytes.HasPrefix(trimmedLine, []byte("[")) {
					streamErr := statusErr{code: http.StatusBadGateway, msg: string(trimmedLine)}
					helps.RecordAPIResponseError(ctx, e.cfg, streamErr)
					reporter.PublishFailure(ctx, streamErr)
					select {
					case out <- cliproxyexecutor.StreamChunk{Err: streamErr}:
					case <-ctx.Done():
					}
					return
				}
				continue
			}

			payload := bytes.TrimSpace(trimmedLine[len("data:"):])

			if bytes.Equal(payload, []byte("[DONE]")) {
				chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, []byte("data: [DONE]"), &param)
				for i := range chunks {
					select {
					case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
					case <-ctx.Done():
						return
					}
				}
				break
			}

			if len(payload) == 0 || payload[0] != '{' {
				continue
			}

			eventType := gjson.GetBytes(payload, "event").String()
			dataField := gjson.GetBytes(payload, "data")

			switch eventType {
			case "content_block_delta":
				text := dataField.Get("text").String()
				if text != "" {
					chunk := fmt.Sprintf(`data: {"choices":[{"delta":{"content":%s}}]}`, mustJSONString(text))
					chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, []byte(chunk), &param)
					for i := range chunks {
						select {
						case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
						case <-ctx.Done():
							return
						}
					}
				}

			case "tool_use_start":
				toolCallID = dataField.Get("id").String()
				toolCallName = dataField.Get("name").String()
				idx := toolCallIndex
				chunk := fmt.Sprintf(`data: {"choices":[{"delta":{"tool_calls":[{"index":%d,"id":%s,"type":"function","function":{"name":%s,"arguments":""}}]}}]}`,
					idx, mustJSONString(toolCallID), mustJSONString(toolCallName))
				chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, []byte(chunk), &param)
				for i := range chunks {
					select {
					case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
					case <-ctx.Done():
						return
					}
				}

			case "tool_use_delta":
				arguments := dataField.Get("arguments").String()
				if arguments != "" {
					idx := toolCallIndex
					chunk := fmt.Sprintf(`data: {"choices":[{"delta":{"tool_calls":[{"index":%d,"function":{"arguments":%s}}]}}]}`,
						idx, mustJSONString(arguments))
					chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, []byte(chunk), &param)
					for i := range chunks {
						select {
						case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
						case <-ctx.Done():
							return
						}
					}
				}

			case "tool_use_stop":
				toolCallIndex++

			case "message_delta":
				finishReason := dataField.Get("finish_reason").String()
				usageNode := dataField.Get("usage")
				if finishReason != "" {
					detail := parseQoderUsageFromNode(usageNode)
					if detail.InputTokens > 0 || detail.OutputTokens > 0 {
						reporter.Publish(ctx, detail)
					}
					var usageJSON string
					if usageNode.Exists() {
						usageJSON = fmt.Sprintf(`,"usage":{"prompt_tokens":%d,"completion_tokens":%d,"total_tokens":%d}`,
							usageNode.Get("input_tokens").Int(),
							usageNode.Get("output_tokens").Int(),
							usageNode.Get("input_tokens").Int()+usageNode.Get("output_tokens").Int())
					}
					chunk := fmt.Sprintf(`data: {"choices":[{"finish_reason":"%s"}]%s}`, finishReason, usageJSON)
					chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, []byte(chunk), &param)
					for i := range chunks {
						select {
						case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
						case <-ctx.Done():
							return
						}
					}
				}

			case "message_stop":
				chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, []byte("data: [DONE]"), &param)
				for i := range chunks {
					select {
					case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
					case <-ctx.Done():
						return
					}
				}

			case "error":
				errMsg := dataField.Get("message").String()
				if errMsg == "" {
					errMsg = "qoder stream error"
				}
				streamErr := statusErr{code: http.StatusBadGateway, msg: errMsg}
				helps.RecordAPIResponseError(ctx, e.cfg, streamErr)
				reporter.PublishFailure(ctx, streamErr)
				select {
				case out <- cliproxyexecutor.StreamChunk{Err: streamErr}:
				case <-ctx.Done():
				}
				return

			default:
				log.Debugf("qoder executor: unknown SSE event type: %s", eventType)
			}
		}
		if errScan := scanner.Err(); errScan != nil {
			helps.RecordAPIResponseError(ctx, e.cfg, errScan)
			reporter.PublishFailure(ctx, errScan)
			select {
			case out <- cliproxyexecutor.StreamChunk{Err: errScan}:
			case <-ctx.Done():
			}
		} else {
			chunks := sdktranslator.TranslateStream(ctx, to, from, req.Model, opts.OriginalRequest, translated, []byte("data: [DONE]"), &param)
			for i := range chunks {
				select {
				case out <- cliproxyexecutor.StreamChunk{Payload: chunks[i]}:
				case <-ctx.Done():
					return
				}
			}
		}
		reporter.EnsurePublished(ctx)
	}()
	return &cliproxyexecutor.StreamResult{Headers: httpResp.Header.Clone(), Chunks: out}, nil
}

// Refresh is a no-op for API-key based Qoder credentials.
func (e *QoderExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	log.Debugf("qoder executor: refresh called")
	return auth, nil
}

// CountTokens returns the local token count for the given request.
func (e *QoderExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")
	translated := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, false)

	modelForCounting := baseModel

	translated, err := thinking.ApplyThinking(translated, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}

	enc, err := helps.TokenizerForModel(modelForCounting)
	if err != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("qoder executor: tokenizer init failed: %w", err)
	}

	count, err := helps.CountOpenAIChatTokens(enc, translated)
	if err != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("qoder executor: token counting failed: %w", err)
	}

	usageJSON := helps.BuildOpenAIUsageJSON(count)
	translatedUsage := sdktranslator.TranslateTokenCount(ctx, to, from, count, usageJSON)
	return cliproxyexecutor.Response{Payload: translatedUsage}, nil
}

// resolveCredentials extracts the base URL and API key from auth attributes.
func (e *QoderExecutor) resolveCredentials(auth *cliproxyauth.Auth) (baseURL, apiKey string) {
	if auth == nil {
		return "", ""
	}
	if auth.Attributes != nil {
		baseURL = strings.TrimSpace(auth.Attributes["base_url"])
		apiKey = strings.TrimSpace(auth.Attributes["api_key"])
	}
	return
}

// applyQoderHeaders sets the required Qoder-specific HTTP headers.
func (e *QoderExecutor) applyQoderHeaders(req *http.Request) {
	req.Header.Set("X-Request-Id", generateRequestID())
	req.Header.Set("X-Machine-Id", getMachineID())
	req.Header.Set("X-Machine-OS", runtime.GOOS)
	req.Header.Set("X-Client-Timestamp", time.Now().UTC().Format(time.RFC3339))
}

// getMachineID returns a stable machine identifier for the X-Machine-Id header.
func getMachineID() string {
	h, err := os.Hostname()
	if err != nil || h == "" {
		h = "cliproxy-unknown"
	}
	src := []byte("qoder-machine-" + h)
	b := make([]byte, hex.EncodedLen(len(src)))
	hex.Encode(b, src)
	return string(b[:32])
}

// convertOpenAIToQoder converts an OpenAI-format chat completions payload
// into Qoder's proprietary request format.
func convertOpenAIToQoder(openaiPayload []byte, stream bool) ([]byte, error) {
	if len(openaiPayload) == 0 {
		return openaiPayload, nil
	}

	result := make(map[string]any)
	if err := json.Unmarshal(openaiPayload, &result); err != nil {
		return nil, fmt.Errorf("failed to parse OpenAI payload: %w", err)
	}

	if messages, ok := result["messages"].([]any); ok {
		for i, msg := range messages {
			msgMap, ok := msg.(map[string]any)
			if !ok {
				continue
			}
			content, exists := msgMap["content"]
			if !exists || content == nil {
				continue
			}
			switch c := content.(type) {
			case string:
				msgMap["content"] = []map[string]any{
					{"type": "text", "text": c},
				}
				result["messages"].([]any)[i] = msgMap
			case []any:
			}
		}
	}

	if tools, ok := result["tools"].([]any); ok {
		for i, tool := range tools {
			toolMap, ok := tool.(map[string]any)
			if !ok {
				continue
			}
			if function, ok := toolMap["function"].(map[string]any); ok {
				if params, ok := function["parameters"]; ok {
					toolMap["input_schema"] = params
					delete(function, "parameters")
				}
				if name, ok := function["name"]; ok {
					toolMap["name"] = name
				}
				if desc, ok := function["description"]; ok {
					toolMap["description"] = desc
				}
				delete(toolMap, "function")
				delete(toolMap, "type")
			}
			tools[i] = toolMap
		}
	}

	delete(result, "stream_options")

	if stream {
		result["stream"] = true
	} else {
		delete(result, "stream")
	}

	if model, ok := result["model"].(string); ok && strings.TrimSpace(model) == "" {
		result["model"] = "auto"
	}

	out, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal Qoder payload: %w", err)
	}
	return out, nil
}

// convertQoderResponseToOpenAI converts a non-streaming Qoder response body
// into OpenAI chat completions format.
func convertQoderResponseToOpenAI(qoderBody []byte) []byte {
	if len(qoderBody) == 0 {
		return qoderBody
	}

	root := gjson.ParseBytes(qoderBody)

	if root.Get("choices").Exists() {
		return qoderBody
	}

	out := map[string]any{
		"id":    root.Get("id").String(),
		"model": root.Get("model").String(),
	}

	var messageContent string
	var toolCalls []map[string]any
	contentBlocks := root.Get("content")
	if contentBlocks.Exists() && contentBlocks.IsArray() {
		contentBlocks.ForEach(func(_, block gjson.Result) bool {
			blockType := block.Get("type").String()
			switch blockType {
			case "text":
				messageContent += block.Get("text").String()
			case "tool_use":
				tc := map[string]any{
					"id":   block.Get("id").String(),
					"type": "function",
					"function": map[string]any{
						"name":      block.Get("name").String(),
						"arguments": block.Get("input").Raw,
					},
				}
				toolCalls = append(toolCalls, tc)
			}
			return true
		})
	}

	choice := map[string]any{
		"index": 0,
		"message": map[string]any{
			"role":    "assistant",
			"content": messageContent,
		},
	}
	if len(toolCalls) > 0 {
		choice["message"].(map[string]any)["tool_calls"] = toolCalls
	}

	finishReason := root.Get("stop_reason").String()
	if finishReason == "" {
		finishReason = root.Get("finish_reason").String()
	}
	if finishReason == "" {
		finishReason = "stop"
	}
	choice["finish_reason"] = finishReason

	out["choices"] = []any{choice}

	usageNode := root.Get("usage")
	if usageNode.Exists() {
		inputTokens := usageNode.Get("input_tokens").Int()
		outputTokens := usageNode.Get("output_tokens").Int()
		out["usage"] = map[string]any{
			"prompt_tokens":     inputTokens,
			"completion_tokens": outputTokens,
			"total_tokens":      inputTokens + outputTokens,
		}
	}

	result, _ := json.Marshal(out)
	return result
}

// parseQoderUsageFromNode extracts token usage from a Qoder usage JSON node.
func parseQoderUsageFromNode(node gjson.Result) (detail coreusage.Detail) {
	if !node.Exists() {
		return
	}
	detail.InputTokens = node.Get("input_tokens").Int()
	detail.OutputTokens = node.Get("output_tokens").Int()
	detail.TotalTokens = detail.InputTokens + detail.OutputTokens
	return
}

// mustJSONString returns a JSON-encoded string value safe for embedding in JSON.
func mustJSONString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}
