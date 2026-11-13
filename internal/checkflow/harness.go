package checkflow

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"time"

	"voice-timeline-renderer/internal/httpapi"
	"voice-timeline-renderer/internal/render"
	"voice-timeline-renderer/internal/service"
	"voice-timeline-renderer/internal/storage"
)

type Harness struct {
	baseURL string
	client  *http.Client
	tempDir string
	server  *httptest.Server
}

func NewHarness() (*Harness, error) {
	return NewHarnessWithConfig(service.DefaultConfig())
}

// NewHarnessWithConfig starts the real HTTP surface with a custom service
// configuration, so checks can exercise resource refusals and memory bounds.
func NewHarnessWithConfig(config service.Config) (*Harness, error) {
	tempDir, err := os.MkdirTemp("", "voice-timeline-check-*")
	if err != nil {
		return nil, err
	}
	repo, err := storage.NewRepository(tempDir)
	if err != nil {
		_ = os.RemoveAll(tempDir)
		return nil, err
	}
	serviceLayer := service.New(repo, config)
	httpServer := httpapi.New(serviceLayer, httpapi.Options{
		MaxSourceRequestBytes: 48 << 20,
		MaxJSONBytes:          2 << 20,
	})
	server := httptest.NewServer(httpServer.Handler())
	return &Harness{
		baseURL: server.URL,
		client:  &http.Client{Timeout: 45 * time.Second},
		tempDir: tempDir,
		server:  server,
	}, nil
}

func (harness *Harness) prepareStats(ctx context.Context) render.PrepareStats {
	status, body, err := harness.requestJSON(ctx, "GET", "/v1/preparation/stats", nil)
	if err != nil || status != 200 {
		return render.PrepareStats{}
	}
	var stats render.PrepareStats
	if err := decodeBody(body, &stats); err != nil {
		return render.PrepareStats{}
	}
	return stats
}

func (harness *Harness) Close() {
	harness.server.Close()
	_ = os.RemoveAll(harness.tempDir)
}

func (harness *Harness) requestJSON(
	ctx context.Context,
	method string,
	path string,
	payload any,
) (int, []byte, error) {
	var body io.Reader
	if payload != nil {
		encoded, err := json.Marshal(payload)
		if err != nil {
			return 0, nil, err
		}
		body = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, harness.baseURL+path, body)
	if err != nil {
		return 0, nil, err
	}
	if payload != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := harness.client.Do(request)
	if err != nil {
		return 0, nil, err
	}
	defer response.Body.Close()
	result, err := io.ReadAll(response.Body)
	if err != nil {
		return 0, nil, err
	}
	return response.StatusCode, result, nil
}

func decodeBody(body []byte, target any) error {
	if err := json.Unmarshal(body, target); err != nil {
		return fmt.Errorf("decode response %q: %w", string(body), err)
	}
	return nil
}
