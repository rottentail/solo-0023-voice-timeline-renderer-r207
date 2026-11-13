package service

import (
	"context"
	"errors"
	"time"

	"voice-timeline-renderer/internal/domain"
	"voice-timeline-renderer/internal/render"
	"voice-timeline-renderer/internal/storage"
)

func (s *Service) CreateRender(
	ctx context.Context,
	id string,
	request RenderWriteRequest,
) (domain.RenderJob, bool, error) {
	if err := ctx.Err(); err != nil {
		return domain.RenderJob{}, false, err
	}
	if err := domain.ValidateResourceID("render_id", id); err != nil {
		return domain.RenderJob{}, false, err
	}
	if err := domain.ValidateResourceID("timeline_id", request.TimelineID); err != nil {
		return domain.RenderJob{}, false, err
	}

	plan, job, created, err := s.beginRender(id, request.TimelineID)
	if err != nil {
		return job, created, err
	}

	contents := make(map[string]render.SourceInput)
	opened := make([]*storage.SourceReader, 0, len(plan.Spans))
	defer func() {
		for _, handle := range opened {
			_ = handle.Close()
		}
	}()
	for _, span := range plan.Spans {
		if _, loaded := contents[span.SourceID]; loaded {
			continue
		}
		handle, loadErr := s.repo.OpenSource(span.SourceID)
		if loadErr != nil {
			return s.failRender(job, loadErr)
		}
		opened = append(opened, handle)
		contents[span.SourceID] = render.SourceInput{
			Asset:  handle.Asset(),
			Reader: handle,
			Size:   handle.Size(),
		}
	}
	result, renderErr := render.Render(
		ctx,
		id,
		plan,
		contents,
		s.cache,
		render.Limits{MaxOutputFrames: s.config.MaxOutputFrames},
		time.Now().UTC(),
	)
	if renderErr != nil {
		switch {
		case errors.Is(renderErr, render.ErrPrepareMemory):
			return s.failRender(job, domain.Wrap(domain.KindTooLarge, "render_memory_limit", "", "timeline preparation exceeds the configured memory budget", renderErr))
		case errors.Is(renderErr, context.Canceled), errors.Is(renderErr, context.DeadlineExceeded):
			return s.failRender(job, domain.Wrap(domain.KindInternal, "render_cancelled", "", "render was cancelled before publication", renderErr))
		default:
			return s.failRender(job, domain.Wrap(domain.KindInternal, "render_failed", "", "timeline rendering failed", renderErr))
		}
	}
	if err := s.advanceJob(&job, domain.RenderFinalizing, nil); err != nil {
		return job, true, err
	}
	if err := s.repo.CommitRenderArtifacts(id, result.WAV, result.Receipt); err != nil {
		return s.failRender(job, err)
	}
	job.Output = &result.Output
	job.ReceiptDigest = result.Output.Digest
	job.ReusedPreparedFrames = result.ReusedPrepared
	if err := s.advanceJob(&job, domain.RenderCompleted, nil); err != nil {
		return job, true, err
	}
	return job, true, nil
}

// beginRender serializes only the bookkeeping phase: identity checks, job
// creation, and the queued-to-rendering transition. Preparation and mixing then
// run without the service mutex so concurrent renders can share prepared
// intervals instead of queueing behind one another.
func (s *Service) beginRender(id, timelineID string) (domain.Timeline, domain.RenderJob, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	existing, getErr := s.repo.GetJob(id)
	if getErr == nil {
		if existing.TimelineID != timelineID {
			return domain.Timeline{}, domain.RenderJob{}, false, domain.E(domain.KindConflict, "render_conflict", "render_id", "identifier is already bound to a different timeline")
		}
		return domain.Timeline{}, existing, false, nil
	}
	if appErr := domain.AsAppError(getErr); appErr.Kind != domain.KindNotFound {
		return domain.Timeline{}, domain.RenderJob{}, false, getErr
	}
	plan, err := s.repo.GetTimeline(timelineID)
	if err != nil {
		return domain.Timeline{}, domain.RenderJob{}, false, err
	}
	now := time.Now().UTC()
	job := domain.RenderJob{
		ID:         id,
		TimelineID: plan.ID,
		State:      domain.RenderQueued,
		CreatedAt:  now,
		UpdatedAt:  now,
	}
	if err := s.repo.SaveJob(job); err != nil {
		return domain.Timeline{}, domain.RenderJob{}, false, err
	}
	if err := s.advanceJob(&job, domain.RenderRendering, nil); err != nil {
		return domain.Timeline{}, job, true, err
	}
	return plan, job, true, nil
}

func (s *Service) advanceJob(job *domain.RenderJob, state domain.RenderState, detail *domain.ErrorDetail) error {
	if err := job.Transition(state, detail, time.Now().UTC()); err != nil {
		return err
	}
	if err := s.repo.SaveJob(*job); err != nil {
		return err
	}
	return nil
}

func (s *Service) failRender(job domain.RenderJob, cause error) (domain.RenderJob, bool, error) {
	appErr := domain.AsAppError(cause)
	detail := &domain.ErrorDetail{
		Code:    appErr.Detail.Code,
		Message: appErr.Detail.Message,
		Field:   appErr.Detail.Field,
	}
	transitionErr := job.Transition(domain.RenderFailed, detail, time.Now().UTC())
	if transitionErr != nil {
		return job, true, transitionErr
	}
	job.Output = nil
	job.ReceiptDigest = ""
	if saveErr := s.repo.SaveJob(job); saveErr != nil {
		return job, true, saveErr
	}
	return job, true, appErr
}
