package service

import (
	"context"
	"sync"

	"voice-timeline-renderer/internal/domain"
	"voice-timeline-renderer/internal/render"
	"voice-timeline-renderer/internal/storage"
)

type Repository interface {
	SaveSource(domain.SourceAsset, []byte) error
	GetSourceMeta(string) (domain.SourceAsset, error)
	OpenSource(string) (*storage.SourceReader, error)
	SaveTimeline(domain.Timeline) error
	GetTimeline(string) (domain.Timeline, error)
	SaveJob(domain.RenderJob) error
	GetJob(string) (domain.RenderJob, error)
	CommitRenderArtifacts(string, []byte, domain.RenderReceipt) error
	GetRenderOutput(string) ([]byte, domain.RenderReceipt, error)
}

type Config struct {
	MaxSourceBytes     int64
	MaxSpans           int
	MaxOutputFrames    int64
	PrepareCacheBytes  int64
	PrepareBudgetBytes int64
	ResidentBytes      int64
}

func DefaultConfig() Config {
	return Config{
		MaxSourceBytes:  32 << 20,
		MaxSpans:        64,
		MaxOutputFrames: 48000 * 180,
		// Cache plus active renders stay below the 192 MiB service ceiling;
		// the remainder covers the Go runtime and the HTTP layer. The cache
		// ceiling is a sub-limit of the active budget, so retention and live
		// work cannot each claim the same bytes.
		PrepareCacheBytes:  48 << 20,
		PrepareBudgetBytes: 112 << 20,
	}
}

type Service struct {
	repo   Repository
	config Config
	cache  *render.PrepareCache
	mu     sync.Mutex
}

func New(repo Repository, config Config) *Service {
	defaults := DefaultConfig()
	if config.MaxSourceBytes <= 0 {
		config.MaxSourceBytes = defaults.MaxSourceBytes
	}
	if config.MaxSpans <= 0 {
		config.MaxSpans = defaults.MaxSpans
	}
	if config.MaxOutputFrames <= 0 {
		config.MaxOutputFrames = defaults.MaxOutputFrames
	}
	if config.PrepareBudgetBytes <= 0 {
		config.PrepareBudgetBytes = defaults.PrepareBudgetBytes
	}
	if config.PrepareCacheBytes <= 0 {
		config.PrepareCacheBytes = defaults.PrepareCacheBytes
	}
	if config.ResidentBytes <= 0 {
		config.ResidentBytes = 192 << 20
	}
	if config.PrepareCacheBytes > config.PrepareBudgetBytes {
		config.PrepareCacheBytes = config.PrepareBudgetBytes
	}
	budget := render.NewBudget(config.PrepareBudgetBytes)
	return &Service{
		repo:   repo,
		config: config,
		cache:  render.NewPrepareCache(budget, config.PrepareCacheBytes),
	}
}

// PrepareStats reports live preparation reuse and the shared memory budget.
func (s *Service) PrepareStats() render.PrepareStats {
	return s.cache.Stats()
}

// ResidentCeiling is the documented maximum for cache plus active preparation.
func (s *Service) ResidentCeiling() int64 {
	return s.config.ResidentBytes
}

type SourceWriteRequest struct {
	Name      string `json:"name"`
	WAVBase64 string `json:"wav_base64"`
}

type TimelineWriteRequest struct {
	Name           string                `json:"name"`
	SampleRate     int                   `json:"sample_rate"`
	OutputChannels int                   `json:"output_channels"`
	TargetPeakDBFS float64               `json:"target_peak_dbfs"`
	Spans          []domain.TimelineSpan `json:"spans"`
}

type RenderWriteRequest struct {
	TimelineID string `json:"timeline_id"`
}

type RenderResult struct {
	Job domain.RenderJob
	WAV []byte
}

func (s *Service) GetSource(ctx context.Context, id string) (domain.SourceAsset, error) {
	if err := ctx.Err(); err != nil {
		return domain.SourceAsset{}, err
	}
	return s.repo.GetSourceMeta(id)
}

func (s *Service) GetTimeline(ctx context.Context, id string) (domain.Timeline, error) {
	if err := ctx.Err(); err != nil {
		return domain.Timeline{}, err
	}
	return s.repo.GetTimeline(id)
}

func (s *Service) GetRender(ctx context.Context, id string) (domain.RenderJob, error) {
	if err := ctx.Err(); err != nil {
		return domain.RenderJob{}, err
	}
	return s.repo.GetJob(id)
}

func (s *Service) GetReceipt(ctx context.Context, id string) (domain.RenderReceipt, error) {
	job, err := s.GetRender(ctx, id)
	if err != nil {
		return domain.RenderReceipt{}, err
	}
	if job.State != domain.RenderCompleted {
		return domain.RenderReceipt{}, domain.E(domain.KindUnready, "render_not_completed", "render_id", "render receipt is available only after completion")
	}
	_, receipt, err := s.repo.GetRenderOutput(id)
	return receipt, err
}

func (s *Service) GetOutput(ctx context.Context, id string) ([]byte, domain.RenderReceipt, error) {
	job, err := s.GetRender(ctx, id)
	if err != nil {
		return nil, domain.RenderReceipt{}, err
	}
	if job.State != domain.RenderCompleted {
		return nil, domain.RenderReceipt{}, domain.E(domain.KindUnready, "render_not_completed", "render_id", "render output is available only after completion")
	}
	return s.repo.GetRenderOutput(id)
}
