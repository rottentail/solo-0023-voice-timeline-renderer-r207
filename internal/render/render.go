package render

import (
	"context"
	"errors"
	"time"

	"voice-timeline-renderer/internal/audiocodec"
	"voice-timeline-renderer/internal/domain"
)

type Limits struct {
	MaxOutputFrames int64
}

type Result struct {
	WAV         []byte
	Receipt     domain.RenderReceipt
	Output      domain.OutputAsset
	Preparation PrepareStats
	// ReusedPrepared is how many prepared output frames this render served
	// from the shared cache instead of converting again.
	ReusedPrepared int64
}

// WorkingSet is a conservative estimate of the bytes one render has live at its
// peak. Retained converted intervals accumulate in the cache; while a build
// runs, the raw PCM read (2 bytes per source sample), its float32 selection
// (4 bytes), and its float32 output (4 bytes per output sample) coexist. After
// preparation, the mix buffer and encoded output appear alongside all retained
// intervals. Duplicate span keys are counted once because they share memory.
func WorkingSet(plan domain.Timeline, sources map[string]SourceInput) int64 {
	seen := make(map[PreparedKey]struct{}, len(plan.Spans))
	retained := int64(0)
	largestBuild := int64(0)
	outputFrames := int64(0)
	for _, span := range plan.Spans {
		source, ok := sources[span.SourceID]
		if !ok {
			continue
		}
		key := PreparedKey{
			SourceDigest:     source.Asset.Digest,
			SourceStartFrame: span.SourceStartFrame,
			SourceEndFrame:   span.SourceEndFrame,
			SampleRate:       plan.SampleRate,
			Channels:         plan.OutputChannels,
		}
		output := domain.ScaledFrames(span.SourceEndFrame-span.SourceStartFrame, source.Asset.SampleRate, plan.SampleRate)
		outputSamples := output * int64(plan.OutputChannels)
		if _, exists := seen[key]; !exists {
			seen[key] = struct{}{}
			sourceSamples := (span.SourceEndFrame - span.SourceStartFrame) * int64(source.Asset.Channels)
			retained += outputSamples * 4
			build := sourceSamples*6 + outputSamples*4
			if build > largestBuild {
				largestBuild = build
			}
		}
		outputFrames += output - span.CrossfadeFrames
	}
	if outputFrames < 0 {
		outputFrames = 0
	}
	outputSamples := outputFrames * int64(plan.OutputChannels)
	// Mix buffer (float32), encoded output (2 bytes per sample), and the
	// receipt's scan of the encoded bytes (2 bytes per sample).
	mixPhase := outputSamples*4 + outputSamples*4 + 4096
	duringBuild := retained + largestBuild
	duringMix := retained + mixPhase
	if duringMix > duringBuild {
		duringBuild = duringMix
	}
	return duringBuild
}

// Render converts a timeline using the shared preparation cache. Cached
// intervals are only read: the mix buffer is a fresh allocation, gain and
// equal-power weights are applied while copying, and peak normalization writes
// to that separate mix buffer. No cache entry is ever modified.
//
// The render reserves its estimated working set in the shared budget before any
// allocation, so concurrent renders cannot overcommit memory and cannot wait on
// memory they themselves hold.
func Render(
	ctx context.Context,
	renderID string,
	plan domain.Timeline,
	sources map[string]SourceInput,
	cache *PrepareCache,
	limits Limits,
	now time.Time,
) (Result, error) {
	if cache == nil {
		return Result{}, errors.New("preparation cache is required")
	}
	reservation := WorkingSet(plan, sources)
	if err := cache.budget.Acquire(ctx, reservation); err != nil {
		return Result{}, err
	}
	defer cache.budget.Release(reservation)

	_, reusedBefore := cache.Baseline()
	spans, totalFrames, release, err := prepareSpans(ctx, plan, sources, cache)
	if err != nil {
		return Result{}, err
	}
	if release != nil {
		defer release()
	}
	if totalFrames < 1 {
		return Result{}, errors.New("timeline produced no output frames")
	}
	if limits.MaxOutputFrames > 0 && totalFrames > limits.MaxOutputFrames {
		return Result{}, errors.New("rendered timeline exceeds the configured frame limit")
	}
	mix, _, err := mixPrepared(ctx, spans, totalFrames, plan.OutputChannels)
	if err != nil {
		return Result{}, err
	}
	normalized := normalizePeak(mix, plan.TargetPeakDBFS)
	encoded, err := audiocodec.EncodePCM16(plan.SampleRate, plan.OutputChannels, normalized.Samples)
	if err != nil {
		return Result{}, err
	}
	receipt, output, err := buildReceipt(renderID, plan, encoded, normalized, spans, now)
	if err != nil {
		return Result{}, err
	}
	_, reusedAfter := cache.Baseline()
	reusedPrepared := reusedAfter - reusedBefore
	if reusedPrepared < 0 {
		reusedPrepared = 0
	}
	return Result{
		WAV:            encoded,
		Receipt:        receipt,
		Output:         output,
		Preparation:    cache.Stats(),
		ReusedPrepared: reusedPrepared,
	}, nil
}
