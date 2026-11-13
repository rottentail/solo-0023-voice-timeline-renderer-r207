package render

import (
	"bytes"
	"context"
	"errors"
	"io"

	"voice-timeline-renderer/internal/audiocodec"
	"voice-timeline-renderer/internal/domain"
)

// SourceInput is an opened, integrity-checked source. Span intervals are
// decoded through its random-access reader so only the selected frames and the
// prepared result are resident.
type SourceInput struct {
	Asset  domain.SourceAsset
	Reader io.ReaderAt
	Size   int64
}

type preparedSpan struct {
	Index            int
	SourceID         string
	SourceDigest     string
	SourceStartFrame int64
	SourceEndFrame   int64
	OutputStartFrame int64
	OutputEndFrame   int64
	OutputFrames     int64
	CrossfadeFrames  int64
	GainDB           float64
	ChannelAction    string
	Clip             *audiocodec.Clip
}

// prepareSpans resolves every span against the shared preparation cache.
//
// The key includes the exact source interval, so a different crop edge or
// sample grid never reuses another interval's samples. The returned release
// function must be called once the spans are no longer being mixed.
func prepareSpans(
	ctx context.Context,
	plan domain.Timeline,
	sources map[string]SourceInput,
	cache *PrepareCache,
) ([]preparedSpan, int64, func(), error) {
	if cache == nil {
		return nil, 0, nil, errors.New("preparation cache is required")
	}
	layouts := make(map[string]audiocodec.WAVLayout, len(sources))
	releases := make([]func(), 0, len(plan.Spans))
	releaseAll := func() {
		for _, release := range releases {
			release()
		}
	}
	prepared := make([]preparedSpan, 0, len(plan.Spans))
	cursor := int64(0)
	for index, span := range plan.Spans {
		source, ok := sources[span.SourceID]
		if !ok {
			releaseAll()
			return nil, 0, nil, errors.New("timeline references a missing source")
		}
		layout, ok := layouts[span.SourceID]
		if !ok {
			parsed, err := audiocodec.ParseWAV(source.Reader, source.Size)
			if err != nil {
				releaseAll()
				return nil, 0, nil, err
			}
			layout = parsed
			layouts[span.SourceID] = layout
		}
		if span.SourceEndFrame > layout.FrameCount {
			releaseAll()
			return nil, 0, nil, errors.New("source interval is outside the source frame count")
		}

		key := PreparedKey{
			SourceDigest:     source.Asset.Digest,
			SourceStartFrame: span.SourceStartFrame,
			SourceEndFrame:   span.SourceEndFrame,
			SampleRate:       plan.SampleRate,
			Channels:         plan.OutputChannels,
		}
		clip, release, err := cache.Acquire(
			ctx,
			key,
			SourcePayload{Reader: source.Reader, Layout: layout},
			buildPreparedClip,
		)
		if err != nil {
			releaseAll()
			return nil, 0, nil, err
		}
		releases = append(releases, release)

		frames := clip.FrameCount()
		if frames < 1 {
			releaseAll()
			return nil, 0, nil, errors.New("span produced no output frames")
		}
		start := cursor
		if index > 0 {
			start -= span.CrossfadeFrames
		}
		if start < 0 {
			releaseAll()
			return nil, 0, nil, errors.New("crossfade produced a negative output position")
		}
		end := start + frames
		cursor = end

		prepared = append(prepared, preparedSpan{
			Index:            index,
			SourceID:         span.SourceID,
			SourceDigest:     source.Asset.Digest,
			SourceStartFrame: span.SourceStartFrame,
			SourceEndFrame:   span.SourceEndFrame,
			OutputStartFrame: start,
			OutputEndFrame:   end,
			OutputFrames:     frames,
			CrossfadeFrames:  span.CrossfadeFrames,
			GainDB:           span.GainDB,
			ChannelAction:    channelAction(source.Asset.Channels, plan.OutputChannels),
			Clip:             clip,
		})
	}
	return prepared, cursor, releaseAll, nil
}

func buildPreparedClip(
	ctx context.Context,
	key PreparedKey,
	source SourcePayload,
) (*audiocodec.Clip, error) {
	selected, err := audiocodec.DecodeWAVRangeReader(
		ctx,
		source.Reader,
		source.Layout,
		key.SourceStartFrame,
		key.SourceEndFrame,
	)
	if err != nil {
		return nil, err
	}
	return audiocodec.PrepareFrames(
		ctx,
		selected,
		0,
		selected.FrameCount(),
		key.SampleRate,
		key.Channels,
	)
}

// BuildPreparedClipForCheck exposes the production conversion to the
// verification harness without duplicating its arithmetic.
var BuildPreparedClipForCheck PrepareBuilder = buildPreparedClip

// PrepareReference converts one interval directly, with no cache, so a check
// can compare a retained entry with fresh selection-first conversion.
func PrepareReference(
	ctx context.Context,
	wav []byte,
	startFrame, endFrame int64,
	sampleRate, channels int,
) (*audiocodec.Clip, error) {
	layout, err := audiocodec.ParseWAVBytes(wav)
	if err != nil {
		return nil, err
	}
	return buildPreparedClip(
		ctx,
		PreparedKey{SourceStartFrame: startFrame, SourceEndFrame: endFrame, SampleRate: sampleRate, Channels: channels},
		SourcePayload{Reader: bytes.NewReader(wav), Layout: layout},
	)
}

func channelAction(sourceChannels, outputChannels int) string {
	switch {
	case sourceChannels == outputChannels:
		return "unchanged"
	case sourceChannels == 1:
		return "mono_to_stereo"
	default:
		return "stereo_to_mono"
	}
}
