package openai

import (
	"context"
	"errors"
	"net/http"
)

var (
	ErrO1MaxTokensDeprecated                   = errors.New("this model is not supported MaxTokens, please use MaxCompletionTokens")                               //nolint:lll
	ErrCompletionUnsupportedModel              = errors.New("this model is not supported with this method, please use CreateChatCompletion client method instead") //nolint:lll
	ErrCompletionStreamNotSupported            = errors.New("streaming is not supported with this method, please use CreateCompletionStream")                      //nolint:lll
	ErrCompletionRequestPromptTypeNotSupported = errors.New("the type of CompletionRequest.Prompt only supports string and []string")                              //nolint:lll
)

var (
	ErrO1BetaLimitationsMessageTypes = errors.New("this model has beta-limitations, user and assistant messages only, system messages are not supported")                                  //nolint:lll
	ErrO1BetaLimitationsTools        = errors.New("this model has beta-limitations, tools, function calling, and response format parameters are not supported")                            //nolint:lll
	ErrO1BetaLimitationsLogprobs     = errors.New("this model has beta-limitations, logprobs not supported")                                                                               //nolint:lll
	ErrO1BetaLimitationsOther        = errors.New("this model has beta-limitations, temperature, top_p and n are fixed at 1, while presence_penalty and frequency_penalty are fixed at 0") //nolint:lll
)

// GPT3 Defines the models provided by OpenAI to use when generating
// completions from OpenAI.
// GPT3 Models are designed for text-based tasks. For code-specific
// tasks, please refer to the Codex series of models.
const (
	Gemini1Dot5Flash    = "gemini-1.5-flash"
	Gemini2Dot0FlashExp = "gemini-2.0-flash-exp"
	Gemini1Dot5Pro      = "gemini-1.5-pro"
)

var disabledModelsForEndpoints = map[string]map[string]bool{
	"/completions": {
		Gemini1Dot5Flash:    true,
		Gemini2Dot0FlashExp: true,
		Gemini1Dot5Pro:      true,
	},
	chatCompletionsSuffix: {},
}

func checkEndpointSupportsModel(endpoint, model string) bool {
	return !disabledModelsForEndpoints[endpoint][model]
}

func checkPromptType(prompt any) bool {
	_, isString := prompt.(string)
	_, isStringSlice := prompt.([]string)
	if isString || isStringSlice {
		return true
	}

	// check if it is prompt is []string hidden under []any
	slice, isSlice := prompt.([]any)
	if !isSlice {
		return false
	}

	for _, item := range slice {
		_, itemIsString := item.(string)
		if !itemIsString {
			return false
		}
	}
	return true // all items in the slice are string, so it is []string
}

var unsupportedToolsForO1Models = map[ToolType]struct{}{
	ToolTypeFunction: {},
}

var availableMessageRoleForO1Models = map[string]struct{}{
	ChatMessageRoleUser:      {},
	ChatMessageRoleAssistant: {},
}

// CompletionRequest represents a request structure for completion API.
type CompletionRequest struct {
	Model            string  `json:"model"`
	Prompt           any     `json:"prompt,omitempty"`
	BestOf           int     `json:"best_of,omitempty"`
	Echo             bool    `json:"echo,omitempty"`
	FrequencyPenalty float32 `json:"frequency_penalty,omitempty"`
	// LogitBias is must be a token id string (specified by their token ID in the tokenizer), not a word string.
	// incorrect: `"logit_bias":{"You": 6}`, correct: `"logit_bias":{"1639": 6}`
	// refs: https://platform.openai.com/docs/api-reference/completions/create#completions/create-logit_bias
	LogitBias map[string]int `json:"logit_bias,omitempty"`
	// Store can be set to true to store the output of this completion request for use in distillations and evals.
	// https://platform.openai.com/docs/api-reference/chat/create#chat-create-store
	Store bool `json:"store,omitempty"`
	// Metadata to store with the completion.
	Metadata        map[string]string `json:"metadata,omitempty"`
	LogProbs        int               `json:"logprobs,omitempty"`
	MaxTokens       int               `json:"max_tokens,omitempty"`
	N               int               `json:"n,omitempty"`
	PresencePenalty float32           `json:"presence_penalty,omitempty"`
	Seed            *int              `json:"seed,omitempty"`
	Stop            []string          `json:"stop,omitempty"`
	Stream          bool              `json:"stream,omitempty"`
	Suffix          string            `json:"suffix,omitempty"`
	Temperature     float32           `json:"temperature,omitempty"`
	TopP            float32           `json:"top_p,omitempty"`
	User            string            `json:"user,omitempty"`
}

// CompletionChoice represents one of possible completions.
type CompletionChoice struct {
	Text         string        `json:"text"`
	Index        int           `json:"index"`
	FinishReason string        `json:"finish_reason"`
	LogProbs     LogprobResult `json:"logprobs"`
}

// LogprobResult represents logprob result of Choice.
type LogprobResult struct {
	Tokens        []string             `json:"tokens"`
	TokenLogprobs []float32            `json:"token_logprobs"`
	TopLogprobs   []map[string]float32 `json:"top_logprobs"`
	TextOffset    []int                `json:"text_offset"`
}

// CompletionResponse represents a response structure for completion API.
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   Usage              `json:"usage"`

	httpHeader
}

// CreateCompletion — API call to create a completion. This is the main endpoint of the API. Returns new text as well
// as, if requested, the probabilities over each alternative token at each position.
//
// If using a fine-tuned model, simply provide the model's ID in the CompletionRequest object,
// and the server will use the model's parameters to generate the completion.
func (c *Client) CreateCompletion(
	ctx context.Context,
	request CompletionRequest,
) (response CompletionResponse, err error) {
	if request.Stream {
		err = ErrCompletionStreamNotSupported
		return
	}

	urlSuffix := "/completions"
	if !checkEndpointSupportsModel(urlSuffix, request.Model) {
		err = ErrCompletionUnsupportedModel
		return
	}

	if !checkPromptType(request.Prompt) {
		err = ErrCompletionRequestPromptTypeNotSupported
		return
	}

	req, err := c.newRequest(
		ctx,
		http.MethodPost,
		c.fullURL(urlSuffix, withModel(request.Model)),
		withBody(request),
	)
	if err != nil {
		return
	}

	err = c.sendRequest(req, &response)
	return
}
