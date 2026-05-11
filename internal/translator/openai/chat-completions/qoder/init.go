package qoder

import (
	. "github.com/router-for-me/CLIProxyAPI/v7/internal/constant"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/interfaces"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/translator/translator"
)

const qoderFormat = "qoder"

func init() {
	translator.Register(
		OpenAI,
		qoderFormat,
		ConvertOpenAIRequestToQoder,
		interfaces.TranslateResponse{
			Stream:    TranslateQoderStreamToOpenAI,
			NonStream: TranslateQoderResponseToOpenAI,
		},
	)
}
