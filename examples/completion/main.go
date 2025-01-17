package main

import (
	"context"
	"fmt"
	"os"

	"github.com/sashabaranov/go-openai"
)

func main() {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
	resp, err := client.CreateCompletion(
		context.Background(),
		openai.CompletionRequest{
			Model:     openai.Gemini1Dot5Flash,
			MaxTokens: 5,
			Prompt:    "Lorem ipsum",
		},
	)
	if err != nil {
		fmt.Printf("Completion error: %v\n", err)
		return
	}
	fmt.Println(resp.Choices[0].Text)
}
