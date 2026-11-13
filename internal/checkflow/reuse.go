package checkflow

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"voice-timeline-renderer/internal/audiocodec"
	"voice-timeline-renderer/internal/domain"
	"voice-timeline-renderer/internal/render"
	"voice-timeline-renderer/internal/service"
)

// runPrepareReuse proves that shared preparation changes how much audio work is
// done without changing a single PCM sample, that different crop edges, sample
// grids, and channel layouts never collide, and that cancellation, eviction,
// and budget pressure cannot corrupt a mix that is still reading cached sound.
func runPrepareReuse(ctx context.Context, harness *Harness, final *render.PrepareStats) error {
	sources, err := prepareReuseSources()
	if err != nil {
		return err
	}
	for id, wav := range sources {
		asset, status, err := harness.createSource(ctx, id, "reuse source "+id, wav)
		if err != nil {
			return err
		}
		if status != 201 || asset.Digest == "" {
			return fmt.Errorf("source %s setup failed: status=%d", id, status)
		}
	}

	// Independent control: the same timelines rendered without preparation
	// sharing and without any cache; these must be PCM-identical.
	if err := checkPrepareEquivalence(ctx, harness, sources); err != nil {
		return err
	}
	if err := checkDifferentKeysDoNotCollide(ctx, harness, sources); err != nil {
		return err
	}
	if err := checkStreamingMeasurementMatchesFullDecode(sources); err != nil {
		return err
	}
	if err := checkReuseReducesWork(ctx, harness); err != nil {
		return err
	}
	if err := checkCancellationAndRecycling(ctx, sources); err != nil {
		return err
	}
	if err := checkWholeSourceCropIsNotTheAnswer(ctx, harness, sources); err != nil {
		return err
	}
	if err := checkReceiptCoordinatesMatchSampling(ctx, harness); err != nil {
		return err
	}
	if err := checkBudgetBoundsActiveWork(ctx, harness, sources); err != nil {
		return err
	}
	if err := checkPreparationIsNotFinalOutput(ctx, harness, sources); err != nil {
		return err
	}
	if err := checkConcurrentSharedPreparation(ctx, harness, sources); err != nil {
		return err
	}
	largeStats, err := checkLargeReuseAndResident(ctx)
	if err != nil {
		return err
	}
	if final != nil {
		*final = largeStats
	}
	if err := checkBudgetRefusalIsObservable(ctx); err != nil {
		return err
	}
	return checkIntegrityNotMasked(ctx, harness, sources)
}

func prepareReuseSources() (map[string][]byte, error) {
	mono48, err := sineWAV(48000, 1, 400, 233, 0.37)
	if err != nil {
		return nil, err
	}
	stereo44, err := sineWAV(44100, 2, 300, 311, 0.29)
	if err != nil {
		return nil, err
	}
	mono32, err := sineWAV(32000, 1, 250, 197, 0.41)
	if err != nil {
		return nil, err
	}
	return map[string][]byte{
		"reuse_mono_48":   mono48,
		"reuse_stereo_44": stereo44,
		"reuse_mono_32":   mono32,
	}, nil
}

// checkPrepareEquivalence renders a set of timelines through the shared cache
// and again through a completely independent full-source path: decode the whole
// source, slice, resample, map channels, mix, normalize. Both outputs must be
// byte-identical after encoding.
func checkPrepareEquivalence(
	ctx context.Context,
	harness *Harness,
	sources map[string][]byte,
) error {
	cases := []struct {
		id         string
		sampleRate int
		channels   int
		targetPeak float64
		spanIDs    []string
		spans      []domain.TimelineSpan
	}{
		{
			id: "reuse_case_edges", sampleRate: 48000, channels: 2, targetPeak: -1,
			spanIDs: []string{"reuse_mono_48", "reuse_stereo_44"},
			spans: []domain.TimelineSpan{
				{SourceID: "reuse_mono_48", SourceStartFrame: 137, SourceEndFrame: 19137, GainDB: -3, CrossfadeFrames: 0},
				{SourceID: "reuse_stereo_44", SourceStartFrame: 441, SourceEndFrame: 13230, GainDB: 1.5, CrossfadeFrames: 2400},
			},
		},
		{
			id: "reuse_case_grid", sampleRate: 44100, channels: 1, targetPeak: -2,
			spanIDs: []string{"reuse_stereo_44", "reuse_mono_32"},
			spans: []domain.TimelineSpan{
				{SourceID: "reuse_stereo_44", SourceStartFrame: 0, SourceEndFrame: 13230, GainDB: 0, CrossfadeFrames: 0},
				{SourceID: "reuse_mono_32", SourceStartFrame: 7, SourceEndFrame: 7999, GainDB: -1, CrossfadeFrames: 1100},
			},
		},
		{
			id: "reuse_case_tiny", sampleRate: 96000, channels: 2, targetPeak: -3,
			spanIDs: []string{"reuse_mono_48", "reuse_mono_32"},
			spans: []domain.TimelineSpan{
				{SourceID: "reuse_mono_48", SourceStartFrame: 9000, SourceEndFrame: 9061, GainDB: 2, CrossfadeFrames: 0},
				{SourceID: "reuse_mono_32", SourceStartFrame: 4001, SourceEndFrame: 4002, GainDB: 0, CrossfadeFrames: 0},
			},
		},
	}

	for _, testCase := range cases {
		plan, status, err := harness.createTimeline(ctx, "timeline_"+testCase.id, testCase.id, testCase.sampleRate, testCase.channels, testCase.targetPeak, testCase.spans)
		if err != nil {
			return err
		}
		if status != 201 {
			return fmt.Errorf("%s timeline setup failed: status=%d", testCase.id, status)
		}
		job, status, err := harness.createRender(ctx, "render_"+testCase.id, plan.ID)
		if err != nil {
			return err
		}
		if status != 201 || job.State != domain.RenderCompleted || job.Output == nil {
			return fmt.Errorf("%s shared render failed: status=%d state=%s error=%+v", testCase.id, status, job.State, job.Error)
		}
		sharedWAV, status, err := harness.getOutput(ctx, job.ID)
		if err != nil || status != 200 {
			return fmt.Errorf("%s shared output unavailable: status=%d err=%v", testCase.id, status, err)
		}

		controlWAV, err := fullRemixReference(ctx, plan, sources)
		if err != nil {
			return fmt.Errorf("%s full reference failed: %w", testCase.id, err)
		}
		sharedDigest := sha256.Sum256(sharedWAV)
		controlDigest := sha256.Sum256(controlWAV)
		if sharedDigest != controlDigest {
			firstDiff := firstByteDifference(sharedWAV, controlWAV)
			return fmt.Errorf(
				"%s prepared output differs from the full remix at byte %d (shared %d bytes, control %d bytes)",
				testCase.id, firstDiff, len(sharedWAV), len(controlWAV),
			)
		}
		if hex.EncodeToString(sharedDigest[:]) != job.Output.Digest {
			return fmt.Errorf("%s receipt digest does not match shared output", testCase.id)
		}
	}
	return nil
}

// fullRemixReference is deliberately independent of the preparation cache. It
// decodes every whole source, then for each span slices, resamples, maps
// channels, gains, and mixes with the same float32 order as render.mixPrepared.
// It never shares converted samples and never reuses a cache entry.
func fullRemixReference(
	ctx context.Context,
	plan domain.Timeline,
	sources map[string][]byte,
) ([]byte, error) {
	decoded := make(map[string]*audiocodec.Clip, len(sources))
	prepared := make([]*audiocodec.Clip, 0, len(plan.Spans))
	starts := make([]int64, 0, len(plan.Spans))
	cursor := int64(0)
	for index, span := range plan.Spans {
		clip, ok := decoded[span.SourceID]
		if !ok {
			parsed, _, err := audiocodec.DecodeWAV(sources[span.SourceID])
			if err != nil {
				return nil, err
			}
			clip = parsed
			decoded[span.SourceID] = parsed
		}
		selected, err := audiocodec.SliceFrames(ctx, clip, span.SourceStartFrame, span.SourceEndFrame)
		if err != nil {
			return nil, err
		}
		resampled, err := audiocodec.Resample(ctx, selected, plan.SampleRate)
		if err != nil {
			return nil, err
		}
		mapped, _, err := audiocodec.MapChannels(ctx, resampled, plan.OutputChannels)
		if err != nil {
			return nil, err
		}
		start := cursor
		if index > 0 {
			start -= span.CrossfadeFrames
		}
		cursor = start + mapped.FrameCount()
		prepared = append(prepared, mapped)
		starts = append(starts, start)
	}

	mix := make([]float32, int(cursor)*plan.OutputChannels)
	channels := plan.OutputChannels
	for index, clip := range prepared {
		span := plan.Spans[index]
		gain := float32(audiocodec.DBToLinear(span.GainDB))
		frames := clip.FrameCount()
		for frame := int64(0); frame < frames; frame++ {
			incoming := float32(1)
			outgoing := float32(1)
			if span.CrossfadeFrames > 0 && frame < span.CrossfadeFrames {
				progress := (float64(frame) + 0.5) / float64(span.CrossfadeFrames)
				angle := progress * math.Pi / 2
				incoming = float32(math.Sin(angle))
				outgoing = float32(math.Cos(angle))
			}
			inStart := int(frame) * channels
			outStart := int(starts[index]+frame) * channels
			for channel := 0; channel < channels; channel++ {
				value := clip.Samples[inStart+channel] * gain * incoming
				if span.CrossfadeFrames > 0 && frame < span.CrossfadeFrames {
					mix[outStart+channel] *= outgoing
				}
				mix[outStart+channel] += value
			}
		}
	}
	normalized := normalizeReference(mix, plan.TargetPeakDBFS)
	return audiocodec.EncodePCM16(plan.SampleRate, plan.OutputChannels, normalized)
}

func normalizeReference(samples []float32, targetDBFS float64) []float32 {
	target := audiocodec.DBToLinear(targetDBFS)
	peak := 0.0
	for _, sample := range samples {
		value := math.Abs(float64(sample))
		if value > peak {
			peak = value
		}
	}
	gain := 1.0
	if peak > 0 {
		gain = target / peak
	}
	if math.IsNaN(gain) || math.IsInf(gain, 0) || gain <= 0 {
		gain = 1
	}
	result := make([]float32, len(samples))
	for index, sample := range samples {
		value := float64(sample) * gain
		if value > 1 {
			value = 1
		}
		if value < -1 {
			value = -1
		}
		result[index] = float32(value)
	}
	return result
}

// checkDifferentKeysDoNotCollide renders overlapping intervals that differ only
// in one crop edge, in sample grid, or in channel layout. Each result must equal
// its own slow reference, which proves nothing was served from a different
// interval's prepared samples.
func checkDifferentKeysDoNotCollide(
	ctx context.Context,
	harness *Harness,
	sources map[string][]byte,
) error {
	variants := []struct {
		id         string
		start      int64
		end        int64
		sampleRate int
		channels   int
		source     string
	}{
		{id: "edge_start", start: 1000, end: 9000, sampleRate: 44100, channels: 2, source: "reuse_mono_48"},
		{id: "edge_start_plus_one", start: 1001, end: 9000, sampleRate: 44100, channels: 2, source: "reuse_mono_48"},
		{id: "edge_end_minus_one", start: 1000, end: 8999, sampleRate: 44100, channels: 2, source: "reuse_mono_48"},
		{id: "grid_48k", start: 1000, end: 9000, sampleRate: 48000, channels: 2, source: "reuse_mono_48"},
		{id: "grid_32k", start: 1000, end: 9000, sampleRate: 32000, channels: 2, source: "reuse_mono_48"},
		{id: "layout_mono", start: 1000, end: 9000, sampleRate: 44100, channels: 1, source: "reuse_mono_48"},
	}
	for _, variant := range variants {
		planID := "timeline_collision_" + variant.id
		_, status, err := harness.createTimeline(
			ctx,
			planID,
			"collision "+variant.id,
			variant.sampleRate,
			variant.channels,
			-1.5,
			[]domain.TimelineSpan{{
				SourceID:         variant.source,
				SourceStartFrame: variant.start,
				SourceEndFrame:   variant.end,
			}},
		)
		if err != nil {
			return err
		}
		if status != 201 {
			return fmt.Errorf("collision case %s rejected: status=%d", variant.id, status)
		}
		renderID := "render_collision_" + variant.id
		job, status, err := harness.createRender(ctx, renderID, planID)
		if err != nil {
			return err
		}
		if status != 201 || job.State != domain.RenderCompleted {
			return fmt.Errorf("collision render %s failed: status=%d state=%s", variant.id, status, job.State)
		}
		sharedWAV, status, err := harness.getOutput(ctx, job.ID)
		if err != nil || status != 200 {
			return fmt.Errorf("collision output %s unavailable: status=%d err=%v", variant.id, status, err)
		}
		plan, status, err := harness.getTimeline(ctx, planID)
		if err != nil || status != 200 {
			return fmt.Errorf("collision plan %s read-back failed: status=%d err=%v", variant.id, status, err)
		}
		controlWAV, err := fullRemixReference(ctx, plan, sources)
		if err != nil {
			return err
		}
		if sha256.Sum256(sharedWAV) != sha256.Sum256(controlWAV) {
			return fmt.Errorf(
				"case %s differs from its own full remix at byte %d: a differently keyed interval was reused",
				variant.id, firstByteDifference(sharedWAV, controlWAV),
			)
		}
	}
	return nil
}

// checkStreamingMeasurementMatchesFullDecode proves ingestion's chunked PCM
// measurement agrees with a full decode plus Measure for every source fixture,
// including those where the 16-bit conversion is most sensitive.
func checkStreamingMeasurementMatchesFullDecode(sources map[string][]byte) error {
	for id, wav := range sources {
		streamed, info, err := audiocodec.MeasureWAVPCM16Bytes(wav)
		if err != nil {
			return err
		}
		clip, decodedInfo, err := audiocodec.DecodeWAV(wav)
		if err != nil {
			return err
		}
		measured := audiocodec.Measure(clip.Samples)
		if streamed.PeakDBFS != measured.PeakDBFS || streamed.RMSDBFS != measured.RMSDBFS {
			return fmt.Errorf("source %s streamed metrics %+v differ from full decode %+v", id, streamed, measured)
		}
		if info.FrameCount != decodedInfo.FrameCount || info.ByteCount != decodedInfo.ByteCount {
			return fmt.Errorf("source %s streamed layout differs from full decode", id)
		}
	}
	return nil
}

// checkReuseReducesWork repeats the same timeline and then a timeline mixing
// repeated and distinct intervals. The second render must perform no new
// conversions for already prepared intervals, and the observed cache hit count
// must rise while output stays identical.
func checkReuseReducesWork(ctx context.Context, harness *Harness) error {
	spans := []domain.TimelineSpan{
		{SourceID: "reuse_mono_48", SourceStartFrame: 500, SourceEndFrame: 9500, GainDB: -1, CrossfadeFrames: 0},
		{SourceID: "reuse_stereo_44", SourceStartFrame: 200, SourceEndFrame: 8200, GainDB: 0.5, CrossfadeFrames: 1200},
	}
	if _, status, err := harness.createTimeline(ctx, "timeline_reuse_once", "reuse once", 48000, 2, -1, spans); err != nil || status != 201 {
		return fmt.Errorf("reuse timeline setup failed: status=%d err=%v", status, err)
	}
	repeated := []domain.TimelineSpan{
		spans[0],
		{SourceID: "reuse_mono_48", SourceStartFrame: 500, SourceEndFrame: 9500, GainDB: 2, CrossfadeFrames: 0},
		{SourceID: "reuse_stereo_44", SourceStartFrame: 200, SourceEndFrame: 8200, GainDB: 0.5, CrossfadeFrames: 1200},
	}
	if _, status, err := harness.createTimeline(ctx, "timeline_reuse_twice", "reuse twice", 48000, 2, -1, repeated); err != nil || status != 201 {
		return fmt.Errorf("repeat timeline setup failed: status=%d err=%v", status, err)
	}

	first, status, err := harness.createRender(ctx, "render_reuse_first", "timeline_reuse_once")
	if err != nil || status != 201 || first.State != domain.RenderCompleted {
		return fmt.Errorf("first reuse render failed: status=%d err=%v", status, err)
	}
	firstOutput, status, err := harness.getOutput(ctx, first.ID)
	if err != nil || status != 200 {
		return fmt.Errorf("first reuse output unavailable: err=%v", err)
	}
	second, status, err := harness.createRender(ctx, "render_reuse_second", "timeline_reuse_once")
	if err != nil || status != 201 || second.State != domain.RenderCompleted {
		return fmt.Errorf("second reuse render failed: status=%d err=%v", status, err)
	}
	secondOutput, status, err := harness.getOutput(ctx, second.ID)
	if err != nil || status != 200 {
		return fmt.Errorf("second reuse output unavailable: err=%v", err)
	}
	if sha256.Sum256(firstOutput) != sha256.Sum256(secondOutput) {
		return fmt.Errorf("repeat render of the same timeline changed its PCM output")
	}
	statsBefore := harness.prepareStats(ctx)
	_, status, err = harness.createRender(ctx, "render_reuse_third", "timeline_reuse_twice")
	if err != nil || status != 201 {
		return fmt.Errorf("repeated-interval render failed: status=%d err=%v", status, err)
	}
	statsAfter := harness.prepareStats(ctx)
	if statsAfter.Hits <= statsBefore.Hits {
		return fmt.Errorf("repeated preparation reported no cache hits: before=%d after=%d", statsBefore.Hits, statsAfter.Hits)
	}
	if statsAfter.ReusedPCM <= statsBefore.ReusedPCM {
		return fmt.Errorf("repeated preparation reused no PCM frames: before=%d after=%d", statsBefore.ReusedPCM, statsAfter.ReusedPCM)
	}
	return nil
}

// checkCancellationAndRecycling exercises the shared preparation cache directly:
// one caller is cancelled while another still needs the sound, and eviction plus
// budget pressure run while a mix holds a published entry. The surviving mix
// must still equal its full remix exactly.
func checkCancellationAndRecycling(ctx context.Context, sources map[string][]byte) error {
	wav := sources["reuse_stereo_44"]
	layout, err := audiocodec.ParseWAVBytes(wav)
	if err != nil {
		return err
	}
	budget := render.NewBudget(8 << 20)
	cache := render.NewPrepareCache(budget, 90_000)
	key := render.PreparedKey{
		SourceDigest:     digestOf(wav),
		SourceStartFrame: 300,
		SourceEndFrame:   7300,
		SampleRate:       48000,
		Channels:         2,
	}
	payload := render.SourcePayload{Reader: bytes.NewReader(wav), Layout: layout}
	// Small selections keep this scenario independent of the large-material
	// checks while still forcing the cache ceiling to cycle entries.

	cancelledContext, cancel := context.WithCancel(ctx)
	cancel()
	if _, _, err := cache.Acquire(cancelledContext, key, payload, render.BuildPreparedClipForCheck); err == nil {
		return fmt.Errorf("a cancelled caller joined a shared preparation instead of failing")
	}

	var shared *audiocodec.Clip
	release, err := acquireWithSurvivor(ctx, cache, key, payload, &shared)
	if err != nil {
		return err
	}
	defer release()
	if shared == nil {
		return fmt.Errorf("surviving caller did not receive prepared sound after the other caller was cancelled")
	}
	expected, err := audiocodec.PrepareFrames(ctx, mustDecode(wav), 300, 7300, 48000, 2)
	if err != nil {
		return err
	}
	if !samplesEqual(shared.Samples, expected.Samples) {
		return fmt.Errorf("cancellation left the surviving preparation different from a fresh conversion")
	}

	// Publish a second entry large enough to force eviction while the first is
	// pinned, then apply budget pressure from a fake active mix.
	beforeCacheSnapshot := append([]float32(nil), shared.Samples...)
	fillerKey := key
	fillerKey.SourceStartFrame = 8000
	fillerKey.SourceEndFrame = 11000
	if _, releaseFiller, err := cache.Acquire(ctx, fillerKey, payload, render.BuildPreparedClipForCheck); err != nil {
		return fmt.Errorf("filler preparation failed: %w", err)
	} else {
		releaseFiller()
	}
	if err := budget.Acquire(ctx, 7<<20); err != nil {
		return fmt.Errorf("active mix reservation failed: %w", err)
	}
	filler2 := fillerKey
	filler2.SourceStartFrame = 11000
	filler2.SourceEndFrame = 13000
	if _, releaseFiller2, err := cache.Acquire(ctx, filler2, payload, render.BuildPreparedClipForCheck); err != nil {
		return fmt.Errorf("budget-pressure preparation failed: %w", err)
	} else {
		releaseFiller2()
	}
	budget.Release(7 << 20)

	if !samplesEqual(shared.Samples, beforeCacheSnapshot) {
		return fmt.Errorf("eviction or budget pressure modified samples held by a live mixed preparation")
	}
	if !samplesEqual(shared.Samples, expected.Samples) {
		return fmt.Errorf("recycled cache entry no longer matches its source interval")
	}
	stats := cache.Stats()
	if stats.Evictions == 0 {
		return fmt.Errorf("recycling never evicted an unpinned entry under budget pressure: %+v", stats)
	}
	if stats.Entries < 1 {
		return fmt.Errorf("recycling dropped the pinned entry that a live mix still reads: %+v", stats)
	}
	// The pinned entry must still be the exact same retained object, and a
	// fresh lookup for it must still be a hit, not a rebuild.
	hitsBefore := stats.Hits
	again, releaseAgain, err := cache.Acquire(ctx, key, payload, render.BuildPreparedClipForCheck)
	if err != nil {
		return err
	}
	releaseAgain()
	if !samplesEqual(again.Samples, expected.Samples) {
		return fmt.Errorf("pinned preparation changed after recycling")
	}
	if cache.Stats().Hits != hitsBefore+1 {
		return fmt.Errorf("pinned preparation was rebuilt instead of retained after recycling")
	}
	return nil
}

func acquireWithSurvivor(
	ctx context.Context,
	cache *render.PrepareCache,
	key render.PreparedKey,
	payload render.SourcePayload,
	target **audiocodec.Clip,
) (func(), error) {
	// Use a gated builder so the second caller deterministically joins an
	// in-flight shared build rather than racing its completion.
	started := make(chan struct{})
	releaseBuild := make(chan struct{})
	var startOnce sync.Once
	gated := func(buildCtx context.Context, buildKey render.PreparedKey, source render.SourcePayload) (*audiocodec.Clip, error) {
		startOnce.Do(func() { close(started) })
		select {
		case <-releaseBuild:
		case <-buildCtx.Done():
			return nil, buildCtx.Err()
		}
		return render.BuildPreparedClipForCheck(buildCtx, buildKey, source)
	}

	firstContext, cancelFirst := context.WithCancel(ctx)
	defer cancelFirst()
	type outcome struct {
		clip    *audiocodec.Clip
		release func()
		err     error
	}
	survivor := make(chan outcome, 1)
	go func() {
		clip, release, err := cache.Acquire(ctx, key, payload, gated)
		survivor <- outcome{clip: clip, release: release, err: err}
	}()

	// Join the still-gated build, then cancel this second caller. The wait
	// counter proves the second caller actually attached to the shared build.
	firstErrored := make(chan error, 1)
	go func() {
		_, _, err := cache.Acquire(firstContext, key, payload, gated)
		firstErrored <- err
	}()
	<-started
	deadline := time.Now().Add(time.Second)
	for cache.Stats().Waits == 0 && time.Now().Before(deadline) {
		if cache.Stats().Waits == 0 {
			if cache.Stats().Builds >= 2 {
				return nil, fmt.Errorf("second caller started its own build instead of joining: %+v", cache.Stats())
			}
		}
		time.Sleep(time.Millisecond)
	}
	if cache.Stats().Waits == 0 {
		return nil, fmt.Errorf("second caller did not join the in-flight shared build: %+v", cache.Stats())
	}
	cancelFirst()
	_ = firstContext
	if err := <-firstErrored; err == nil || !errors.Is(err, context.Canceled) {
		return nil, fmt.Errorf("cancelled joined caller returned %v, want context cancellation", err)
	}
	// The cancelled waiter must not abort the build the survivor still needs.
	close(releaseBuild)
	result := <-survivor
	if result.err != nil {
		return nil, result.err
	}
	*target = result.clip
	return result.release, nil
}

// checkIntegrityNotMasked renders a timeline, then damages the stored source
// bytes on disk. A repeat render of the same timeline must fail with a visible
// integrity error instead of serving previously successful output.
func checkIntegrityNotMasked(ctx context.Context, harness *Harness, sources map[string][]byte) error {
	wav := sources["reuse_mono_32"]
	if _, status, err := harness.createSource(ctx, "reuse_integrity", "integrity take", wav); err != nil || status != 201 {
		return fmt.Errorf("integrity source setup failed: status=%d err=%v", status, err)
	}
	spans := []domain.TimelineSpan{{
		SourceID:         "reuse_integrity",
		SourceStartFrame: 100,
		SourceEndFrame:   6100,
	}}
	if _, status, err := harness.createTimeline(ctx, "timeline_integrity_v1", "integrity", 44100, 1, -1, spans); err != nil || status != 201 {
		return fmt.Errorf("integrity timeline setup failed: status=%d err=%v", status, err)
	}
	first, status, err := harness.createRender(ctx, "render_integrity_first", "timeline_integrity_v1")
	if err != nil || status != 201 || first.State != domain.RenderCompleted {
		return fmt.Errorf("integrity baseline render failed: status=%d err=%v", status, err)
	}

	mediaPath := filepath.Join(harness.tempDir, "sources", "reuse_integrity", "source.wav")
	damaged := append([]byte(nil), wav...)
	damaged[len(damaged)-4] ^= 0xFF
	if err := os.WriteFile(mediaPath, damaged, 0o640); err != nil {
		return fmt.Errorf("cannot damage stored media: %w", err)
	}

	second, status, err := harness.createRender(ctx, "render_integrity_second", "timeline_integrity_v1")
	if err != nil {
		return err
	}
	if status != 500 {
		return fmt.Errorf("damaged source render returned status %d, want 500 job=%+v", status, second)
	}
	if second.ID == "" {
		return fmt.Errorf("damaged source render response did not include the job record")
	}
	if second.State != domain.RenderFailed || second.Error == nil || second.Error.Code != "source_integrity_error" {
		return fmt.Errorf("damaged source was masked by an earlier success: state=%s error=%+v", second.State, second.Error)
	}
	outputStatus, outputBody, err := harness.requestJSON(ctx, "GET", "/v1/renders/"+second.ID+"/output", nil)
	if err != nil {
		return err
	}
	if outputStatus != 409 || failureCode(outputBody) != "render_not_completed" {
		return fmt.Errorf("failed integrity render exposed output: status=%d code=%s", outputStatus, failureCode(outputBody))
	}
	receiptStatus, receiptBody, err := harness.requestJSON(ctx, "GET", "/v1/renders/"+second.ID+"/receipt", nil)
	if err != nil {
		return err
	}
	if receiptStatus != 409 || failureCode(receiptBody) != "render_not_completed" {
		return fmt.Errorf("failed integrity render exposed a receipt: status=%d code=%s", receiptStatus, failureCode(receiptBody))
	}
	return nil
}

// checkWholeSourceCropIsNotTheAnswer demonstrates the tension the shared cache
// must resolve. Resampling the whole source and cropping afterwards is not
// equivalent to cropping and then resampling; only the second behavior may be
// reused, and a rendered timeline must match it exactly.
func checkWholeSourceCropIsNotTheAnswer(
	ctx context.Context,
	harness *Harness,
	sources map[string][]byte,
) error {
	wav := sources["reuse_mono_48"]
	startFrame, endFrame := int64(1237), int64(2237)
	sampleRate, channels := 44100, 2

	whole, _, err := audiocodec.DecodeWAV(wav)
	if err != nil {
		return err
	}
	wholeResampled, err := audiocodec.Resample(ctx, whole, sampleRate)
	if err != nil {
		return err
	}
	croppedAfter, _, err := audiocodec.MapChannels(ctx, wholeResampled, channels)
	if err != nil {
		return err
	}
	// Cropping the converted whole source must use the rounded frame positions
	// of the target grid, which is the cheapest plausible "convert first,
	// crop later" shortcut a producer might attempt.
	ratio := float64(sampleRate) / float64(whole.SampleRate)
	cropStart := int(math.Round(float64(startFrame)*ratio)) * channels
	cropEnd := int(math.Round(float64(endFrame)*ratio)) * channels
	resampleThenCrop := append([]float32(nil), croppedAfter.Samples[cropStart:cropEnd]...)

	cropThenResample, err := audiocodec.PrepareFrames(ctx, whole, startFrame, endFrame, sampleRate, channels)
	if err != nil {
		return err
	}
	if samplesEqual(resampleThenCrop, cropThenResample.Samples) {
		return fmt.Errorf("test fixture cannot distinguish whole-source cropping from selection-first conversion")
	}

	spans := []domain.TimelineSpan{{
		SourceID:         "reuse_mono_48",
		SourceStartFrame: startFrame,
		SourceEndFrame:   endFrame,
	}}
	plan, status, err := harness.createTimeline(ctx, "timeline_crop_answer", "crop answer", sampleRate, channels, -1, spans)
	if err != nil {
		return err
	}
	if status != 201 {
		return fmt.Errorf("crop-answer timeline setup failed: status=%d", status)
	}
	job, status, err := harness.createRender(ctx, "render_crop_answer", "timeline_crop_answer")
	if err != nil || status != 201 || job.State != domain.RenderCompleted {
		return fmt.Errorf("crop-answer render failed: status=%d err=%v", status, err)
	}
	rendered, status, err := harness.getOutput(ctx, job.ID)
	if err != nil || status != 200 {
		return fmt.Errorf("crop-answer output unavailable: err=%v", err)
	}
	decoded, _, err := audiocodec.DecodeWAV(rendered)
	if err != nil {
		return err
	}
	if decoded.FrameCount() != int64(len(cropThenResample.Samples)/channels) {
		return fmt.Errorf("rendered crop frame count %d does not match selection-first conversion %d",
			decoded.FrameCount(), len(cropThenResample.Samples)/channels)
	}
	receipt, status, err := harness.getReceipt(ctx, job.ID)
	if err != nil || status != 200 {
		return fmt.Errorf("crop-answer receipt unavailable: status=%d err=%v", status, err)
	}
	gain := audiocodec.DBToLinear(receipt.Normalization.AppliedGainDB)
	for index := range cropThenResample.Samples {
		expected := math.Max(-1, math.Min(1, float64(cropThenResample.Samples[index])*gain))
		// The received WAV is 16-bit quantized, so compare within one LSB.
		if math.Abs(float64(decoded.Samples[index])-expected) > 1.0/32768 {
			return fmt.Errorf(
				"render does not match selection-first conversion at sample %d: got %.9f want %.9f (normalization gain %.4f dB, plan %d frames)",
				index, decoded.Samples[index], expected, receipt.Normalization.AppliedGainDB, plan.OutputFrames,
			)
		}
	}
	if err := checkSourceCoordinates(ctx, harness, job.ID, startFrame, endFrame); err != nil {
		return err
	}
	return nil
}

// checkReceiptCoordinatesMatchSampling checks that the receipt's source
// interval is the interval actually sampled, on a timeline whose spans share a
// source but use different crop edges.
func checkReceiptCoordinatesMatchSampling(ctx context.Context, harness *Harness) error {
	spans := []domain.TimelineSpan{
		{SourceID: "reuse_mono_48", SourceStartFrame: 10, SourceEndFrame: 1010, GainDB: -1, CrossfadeFrames: 0},
		{SourceID: "reuse_mono_48", SourceStartFrame: 1010, SourceEndFrame: 2010, GainDB: 1, CrossfadeFrames: 200},
		{SourceID: "reuse_stereo_44", SourceStartFrame: 500, SourceEndFrame: 2500, GainDB: 0, CrossfadeFrames: 300},
	}
	if _, status, err := harness.createTimeline(ctx, "timeline_coordinates", "coordinates", 48000, 2, -1, spans); err != nil || status != 201 {
		return fmt.Errorf("coordinate timeline setup failed: status=%d err=%v", status, err)
	}
	job, status, err := harness.createRender(ctx, "render_coordinates", "timeline_coordinates")
	if err != nil || status != 201 || job.State != domain.RenderCompleted {
		return fmt.Errorf("coordinate render failed: status=%d err=%v", status, err)
	}
	receipt, status, err := harness.getReceipt(ctx, job.ID)
	if err != nil || status != 200 {
		return fmt.Errorf("coordinate receipt unavailable: status=%d err=%v", status, err)
	}
	if len(receipt.Spans) != len(spans) {
		return fmt.Errorf("receipt span count %d != %d", len(receipt.Spans), len(spans))
	}
	for index, expected := range spans {
		actual := receipt.Spans[index]
		if actual.SourceID != expected.SourceID ||
			actual.SourceStartFrame != expected.SourceStartFrame ||
			actual.SourceEndFrame != expected.SourceEndFrame ||
			actual.GainDB != expected.GainDB ||
			actual.CrossfadeFrames != expected.CrossfadeFrames {
			return fmt.Errorf("receipt span %d coordinates %+v do not match requested %+v", index, actual, expected)
		}
		expectedFrames := domain.ScaledFrames(
			expected.SourceEndFrame-expected.SourceStartFrame,
			sourceRateFor(expected.SourceID),
			48000,
		)
		if actual.OutputFrames != expectedFrames {
			return fmt.Errorf("receipt span %d output frames %d != %d", index, actual.OutputFrames, expectedFrames)
		}
		if actual.OutputEndFrame-actual.OutputStartFrame != actual.OutputFrames {
			return fmt.Errorf("receipt span %d placement does not match its frame count", index)
		}
	}
	// The output must actually contain the samples at those source
	// coordinates: mix the expected prepared intervals by hand for the
	// non-overlapping head of the first span and compare exactly.
	outputWAV, status, err := harness.getOutput(ctx, job.ID)
	if err != nil || status != 200 {
		return fmt.Errorf("coordinate output unavailable: err=%v", err)
	}
	decoded, _, err := audiocodec.DecodeWAV(outputWAV)
	if err != nil {
		return err
	}
	if err := verifySampleAtCoordinate(ctx, decoded, receipt); err != nil {
		return err
	}
	return nil
}

func sourceRateFor(sourceID string) int {
	switch sourceID {
	case "reuse_stereo_44":
		return 44100
	case "reuse_mono_32":
		return 32000
	default:
		return 48000
	}
}

// checkBudgetBoundsActiveWork proves the shared budget governs active render
// work, not just cache entries: a timeline whose mix cannot fit is refused with
// a stable error instead of overcommitting memory.
func checkBudgetBoundsActiveWork(ctx context.Context, harness *Harness, sources map[string][]byte) error {
	budget := render.NewBudget(4 << 20)
	cache := render.NewPrepareCache(budget, 1<<20)
	// A mix larger than the whole budget must be refused before allocation.
	if err := budget.Acquire(ctx, budget.Capacity()+1); !errors.Is(err, render.ErrPrepareMemory) {
		return fmt.Errorf("oversized work returned %v, want a memory refusal", err)
	}
	layout, err := audiocodec.ParseWAVBytes(sources["reuse_mono_48"])
	if err != nil {
		return err
	}
	key := render.PreparedKey{
		SourceDigest:     digestOf(sources["reuse_mono_48"]),
		SourceStartFrame: 0,
		SourceEndFrame:   2000,
		SampleRate:       48000,
		Channels:         1,
	}
	payload := render.SourcePayload{Reader: bytes.NewReader(sources["reuse_mono_48"]), Layout: layout}
	clip, release, err := cache.Acquire(ctx, key, payload, render.BuildPreparedClipForCheck)
	if err != nil {
		return err
	}
	if clip == nil || clip.FrameCount() != 2000 {
		release()
		return fmt.Errorf("bounded preparation produced an unexpected clip")
	}
	release()
	if err := budget.Acquire(ctx, budget.Capacity()); err != nil {
		return fmt.Errorf("whole budget reservation was refused: %w", err)
	}
	overContext, cancel := context.WithTimeout(ctx, 50*time.Millisecond)
	defer cancel()
	if err := budget.Acquire(overContext, 1); err == nil {
		budget.Release(1)
		return fmt.Errorf("budget admitted work beyond its fixed ceiling")
	} else if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, render.ErrPrepareMemory) {
		return fmt.Errorf("budget refusal returned an unexpected error: %w", err)
	}
	budget.Release(budget.Capacity())
	if used := budget.Used(); used < 0 || used > budget.Capacity() {
		return fmt.Errorf("budget usage %d left its [0,%d] range", used, budget.Capacity())
	}
	return nil
}

func verifySampleAtCoordinate(
	ctx context.Context,
	decoded *audiocodec.Clip,
	receipt domain.RenderReceipt,
) error {
	// Span 0 has no crossfade and starts at output frame 0, so normalization is
	// the only transform on its samples. Derive the gain from the receipt and
	// compare the first output sample with the prepared source sample.
	span := receipt.Spans[0]
	source := fixtureSources()[span.SourceID]
	clip, _, err := audiocodec.DecodeWAV(source)
	if err != nil {
		return err
	}
	prepared, err := audiocodec.PrepareFrames(ctx, clip, span.SourceStartFrame, span.SourceEndFrame, 48000, int(receipt.Output.Channels))
	if err != nil {
		return err
	}
	gain := audiocodec.DBToLinear(span.GainDB)
	full := audiocodec.DBToLinear(receipt.Normalization.AppliedGainDB)
	index := 0
	for frame := int64(0); frame < prepared.FrameCount() && frame < 16; frame++ {
		for channel := 0; channel < int(receipt.Output.Channels); channel++ {
			expected := math.Max(-1, math.Min(1, float64(prepared.Samples[index])*gain*full))
			actual := float64(decoded.Samples[index])
			if math.Abs(expected-actual) > 0.01 {
				return fmt.Errorf("output sample %d at source coordinate %d does not match the recorded sampling: expected %.5f got %.5f",
					index, span.SourceStartFrame+frame, expected, actual)
			}
			index++
		}
	}
	return nil
}

func fixtureSources() map[string][]byte {
	sources, _ := prepareReuseSources()
	return sources
}

func firstSampleDifference(left, right []float32) int {
	limit := len(left)
	if len(right) < limit {
		limit = len(right)
	}
	for index := 0; index < limit; index++ {
		if math.Float32bits(left[index]) != math.Float32bits(right[index]) {
			return index
		}
	}
	return limit
}

func checkSourceCoordinates(ctx context.Context, harness *Harness, renderID string, startFrame, endFrame int64) error {
	receipt, status, err := harness.getReceipt(ctx, renderID)
	if err != nil || status != 200 {
		return fmt.Errorf("coordinate receipt unavailable: status=%d err=%v", status, err)
	}
	if len(receipt.Spans) != 1 {
		return fmt.Errorf("coordinate receipt has %d spans", len(receipt.Spans))
	}
	span := receipt.Spans[0]
	if span.SourceStartFrame != startFrame || span.SourceEndFrame != endFrame {
		return fmt.Errorf("receipt source coordinates [%d,%d) do not match the sampled interval [%d,%d)",
			span.SourceStartFrame, span.SourceEndFrame, startFrame, endFrame)
	}
	return nil
}

// checkConcurrentSharedPreparation runs several renders that share intervals at
// the same time, cancels some callers mid-flight, and verifies that every
// surviving render still equals its full remix. It also checks that the shared
// budget never reports usage above its ceiling.
func checkConcurrentSharedPreparation(ctx context.Context, harness *Harness, sources map[string][]byte) error {
	sharedSpans := []domain.TimelineSpan{
		{SourceID: "reuse_mono_48", SourceStartFrame: 300, SourceEndFrame: 9300, GainDB: -2, CrossfadeFrames: 0},
		{SourceID: "reuse_stereo_44", SourceStartFrame: 600, SourceEndFrame: 9600, GainDB: 0, CrossfadeFrames: 1600},
	}
	if _, status, err := harness.createTimeline(ctx, "timeline_concurrent", "concurrent", 48000, 2, -1, sharedSpans); err != nil || status != 201 {
		return fmt.Errorf("concurrent timeline setup failed: status=%d err=%v", status, err)
	}
	plan, status, err := harness.getTimeline(ctx, "timeline_concurrent")
	if err != nil || status != 200 {
		return fmt.Errorf("concurrent plan read-back failed: status=%d err=%v", status, err)
	}
	control, err := fullRemixReference(ctx, plan, sources)
	if err != nil {
		return err
	}

	type outcome struct {
		wav       []byte
		err       error
		cancelled bool
	}
	results := make(chan outcome, 8)
	var group sync.WaitGroup
	for index := 0; index < 6; index++ {
		group.Add(1)
		go func(index int) {
			defer group.Done()
			// Two callers are cancelled early while four keep going. The
			// cancellation window must expire while shared preparation is still
			// in flight, which is the case that must not strand other callers.
			callerContext := ctx
			if index < 2 {
				var cancel context.CancelFunc
				callerContext, cancel = context.WithTimeout(ctx, 15*time.Millisecond)
				defer cancel()
			}
			renderID := fmt.Sprintf("render_concurrent_%d", index)
			job, status, err := harness.createRender(callerContext, renderID, "timeline_concurrent")
			if err != nil || status >= 400 || job.State != domain.RenderCompleted {
				// A caller that gave up must fail cleanly. Any other failure is
				// a real regression.
				if callerContext.Err() != nil {
					results <- outcome{cancelled: true}
					return
				}
				results <- outcome{err: fmt.Errorf("render %s failed: status=%d state=%s err=%v", renderID, status, job.State, err)}
				return
			}
			wav, _, err := harness.getOutput(ctx, job.ID)
			results <- outcome{wav: wav, err: err}
		}(index)
	}
	group.Wait()
	close(results)
	completed := 0
	cancelled := 0
	var lastErr error
	for result := range results {
		if result.cancelled {
			cancelled++
			continue
		}
		if result.err != nil {
			lastErr = result.err
			continue
		}
		if sha256.Sum256(result.wav) != sha256.Sum256(control) {
			return fmt.Errorf("a concurrent render sharing prepared intervals diverged from the full remix (%d concurrent outputs, %d bytes, control %d bytes)",
				completed, len(result.wav), len(control))
		}
		completed++
	}
	if completed < 2 {
		return fmt.Errorf("concurrent sharing completed only %d renders (%d cancelled, last error %v)", completed, cancelled, lastErr)
	}
	stats := harness.prepareStats(ctx)
	if stats.BudgetUsed > stats.BudgetBytes {
		return fmt.Errorf("shared budget reported %d used of %d", stats.BudgetUsed, stats.BudgetBytes)
	}
	// Either a wait on an in-flight build or a hit on a finished entry proves
	// the concurrent renders shared conversion work.
	if stats.Hits == 0 && stats.Waits == 0 {
		return fmt.Errorf("concurrent renders recording no sharing: %+v", stats)
	}
	return nil
}

// checkLargeReuseAndResident renders a long, multi-span timeline twice. The
// first render performs the conversions; the second must perform none for the
// same intervals, while still producing byte-identical output. It then runs
// several such renders concurrently and samples the process resident set, which
// must stay under the 192 MiB service ceiling.
func checkLargeReuseAndResident(ctx context.Context) (render.PrepareStats, error) {
	big, err := sineWAV(48000, 2, 18000, 217, 0.32)
	if err != nil {
		return render.PrepareStats{}, err
	}
	harness, err := NewHarness()
	if err != nil {
		return render.PrepareStats{}, err
	}
	defer harness.Close()
	if _, status, err := harness.createSource(ctx, "resident_big", "long take", big); err != nil || status != 201 {
		return render.PrepareStats{}, fmt.Errorf("large source setup failed: status=%d err=%v", status, err)
	}
	spans := []domain.TimelineSpan{
		{SourceID: "resident_big", SourceStartFrame: 0, SourceEndFrame: 180000, GainDB: -2, CrossfadeFrames: 0},
		{SourceID: "resident_big", SourceStartFrame: 180000, SourceEndFrame: 360000, GainDB: -1, CrossfadeFrames: 6000},
		{SourceID: "resident_big", SourceStartFrame: 240000, SourceEndFrame: 480000, GainDB: 0, CrossfadeFrames: 6000},
		{SourceID: "resident_big", SourceStartFrame: 421000, SourceEndFrame: 720000, GainDB: 1, CrossfadeFrames: 8000},
	}
	plan, status, err := harness.createTimeline(ctx, "timeline_resident", "long assembly", 48000, 2, -1, spans)
	if err != nil || status != 201 {
		return render.PrepareStats{}, fmt.Errorf("large timeline setup failed: status=%d err=%v", status, err)
	}

	beforeFirst := harness.prepareStats(ctx)
	first, status, err := harness.createRender(ctx, "render_resident_first", plan.ID)
	if err != nil || status != 201 || first.State != domain.RenderCompleted {
		return render.PrepareStats{}, fmt.Errorf("large first render failed: status=%d err=%v", status, err)
	}
	afterFirst := harness.prepareStats(ctx)
	firstWAV, status, err := harness.getOutput(ctx, first.ID)
	if err != nil || status != 200 {
		return render.PrepareStats{}, fmt.Errorf("large first output unavailable: err=%v", err)
	}

	beforeSecond := harness.prepareStats(ctx)
	second, status, err := harness.createRender(ctx, "render_resident_second", plan.ID)
	if err != nil || status != 201 || second.State != domain.RenderCompleted {
		return render.PrepareStats{}, fmt.Errorf("large second render failed: status=%d err=%v", status, err)
	}
	afterSecond := harness.prepareStats(ctx)
	secondWAV, status, err := harness.getOutput(ctx, second.ID)
	if err != nil || status != 200 {
		return render.PrepareStats{}, fmt.Errorf("large second output unavailable: err=%v", err)
	}
	if sha256.Sum256(firstWAV) != sha256.Sum256(secondWAV) {
		return render.PrepareStats{}, fmt.Errorf("reusing prepared intervals changed the large output")
	}
	if afterFirst.Builds <= beforeFirst.Builds {
		return render.PrepareStats{}, fmt.Errorf("first large render performed no preparation builds")
	}
	if released := afterSecond.Builds - beforeSecond.Builds; released != 0 {
		// Later concurrent renders reuse these entries too; a single
		// repeat performing no new conversion is the required property.
		return render.PrepareStats{}, fmt.Errorf("second large render repeated %d builds instead of reusing all intervals", released)
	}
	if afterSecond.ReusedPCM <= 0 {
		return render.PrepareStats{}, fmt.Errorf("second large render reused no PCM frames")
	}
	if second.ReusedPreparedFrames <= 0 {
		return render.PrepareStats{}, fmt.Errorf("second large render reported no reused frames in its job record")
	}
	if first.ReusedPreparedFrames != 0 {
		return render.PrepareStats{}, fmt.Errorf("first large render reported %d reused frames despite a cold cache", first.ReusedPreparedFrames)
	}
	// The large timeline's distinct intervals must be retained across renders;
	// if they were dropped, the zero-build assertion above would already fail.
	// This confirms the bytes are genuinely held in the same shared cache.
	if afterSecond.CachedBytes < afterFirst.CachedBytes {
		return render.PrepareStats{}, fmt.Errorf("retained prepared sound shrank after a pure reuse render: %d -> %d",
			afterFirst.CachedBytes, afterSecond.CachedBytes)
	}
	if !samplesEqual(decodedSamplesForCheck(firstWAV), decodedSamplesForCheck(secondWAV)) {
		return render.PrepareStats{}, fmt.Errorf("large outputs differ at the sample level")
	}

	// Concurrent renders of the same long timeline. Prepare the in-flight
	// sharing before the first one completes, and sample RSS while they run.
	peak := residentBytes()
	stopSampling := make(chan struct{})
	sampled := make(chan int64, 1)
	go func() {
		high := residentBytes()
		for {
			select {
			case <-stopSampling:
				sampled <- high
				return
			default:
				if current := residentBytes(); current > high {
					high = current
				}
				time.Sleep(2 * time.Millisecond)
			}
		}
	}()
	var group sync.WaitGroup
	errs := make(chan error, 4)
	for index := 0; index < 4; index++ {
		group.Add(1)
		go func(index int) {
			defer group.Done()
			renderID := fmt.Sprintf("render_resident_parallel_%d", index)
			job, status, err := harness.createRender(ctx, renderID, plan.ID)
			if err != nil {
				errs <- err
				return
			}
			if status != 201 || job.State != domain.RenderCompleted {
				errs <- fmt.Errorf("%s ended %s", renderID, job.State)
			}
		}(index)
	}
	group.Wait()
	close(stopSampling)
	if high := <-sampled; high > peak {
		peak = high
	}
	close(errs)
	for err := range errs {
		if err != nil {
			return render.PrepareStats{}, err
		}
	}
	// The Go heap is the part of the resident set this service controls.
	// Resident pages can exceed it (runtime, code, page cache), so the ceiling
	// check uses the larger of the two: if heap is bounded and resident is
	// bounded, both promises hold.
	var memory runtime.MemStats
	runtime.ReadMemStats(&memory)
	heap := int64(memory.HeapInuse)
	if heap > peak {
		peak = heap
	}
	_ = heap
	final := harness.prepareStats(ctx)
	if final.Hits == 0 && final.Waits == 0 {
		return render.PrepareStats{}, fmt.Errorf("large concurrent renders recorded no prepared-interval sharing: %+v", final)
	}
	if peak > 192<<20 && !raceInstrumented() {
		return render.PrepareStats{}, fmt.Errorf("resident set reached %d bytes, above the 192 MiB ceiling", peak)
	}
	return harness.prepareStats(ctx), nil
}

func decodedSamplesForCheck(wav []byte) []float32 {
	clip, _, err := audiocodec.DecodeWAV(wav)
	if err != nil {
		return nil
	}
	return clip.Samples
}

func residentBytes() int64 {
	payload, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	fields := strings.Fields(strings.TrimSpace(string(payload)))
	if len(fields) < 2 {
		return 0
	}
	pages, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil {
		return 0
	}
	return pages * int64(os.Getpagesize())
}

// checkPreparationIsNotFinalOutput renders the same intervals under different
// gain and normalization targets. The prepared sound is shared, while each
// output differs and each matches its own full remix. This proves the cache
// holds intermediate converted intervals, not finished products, and that
// normalization never writes back into shared samples.
func checkPreparationIsNotFinalOutput(ctx context.Context, harness *Harness, sources map[string][]byte) error {
	baseSpans := []domain.TimelineSpan{
		{SourceID: "reuse_mono_48", SourceStartFrame: 1400, SourceEndFrame: 9400, GainDB: 0, CrossfadeFrames: 0},
		{SourceID: "reuse_stereo_44", SourceStartFrame: 900, SourceEndFrame: 4500, GainDB: 0, CrossfadeFrames: 800},
	}
	hotSpans := append([]domain.TimelineSpan(nil), baseSpans...)
	hotSpans[0].GainDB = 6
	hotSpans[1].GainDB = -6

	_, status, err := harness.createTimeline(ctx, "timeline_not_final_quiet", "quiet mix", 48000, 2, -6, baseSpans)
	if err != nil || status != 201 {
		return fmt.Errorf("quiet timeline setup failed: status=%d err=%v", status, err)
	}
	_, status, err = harness.createTimeline(ctx, "timeline_not_final_loud", "loud mix", 48000, 2, -1, hotSpans)
	if err != nil || status != 201 {
		return fmt.Errorf("loud timeline setup failed: status=%d err=%v", status, err)
	}

	before := harness.prepareStats(ctx)
	quiet, status, err := harness.createRender(ctx, "render_not_final_quiet", "timeline_not_final_quiet")
	if err != nil || status != 201 || quiet.State != domain.RenderCompleted {
		return fmt.Errorf("quiet render failed: status=%d err=%v", status, err)
	}
	loud, status, err := harness.createRender(ctx, "render_not_final_loud", "timeline_not_final_loud")
	if err != nil || status != 201 || loud.State != domain.RenderCompleted {
		return fmt.Errorf("loud render failed: status=%d err=%v", status, err)
	}
	after := harness.prepareStats(ctx)
	if after.Hits <= before.Hits {
		return fmt.Errorf("second timeline with the same intervals reused no prepared sound")
	}
	quietWAV, status, err := harness.getOutput(ctx, quiet.ID)
	if err != nil || status != 200 {
		return fmt.Errorf("quiet output unavailable: err=%v", err)
	}
	loudWAV, status, err := harness.getOutput(ctx, loud.ID)
	if err != nil || status != 200 {
		return fmt.Errorf("loud output unavailable: err=%v", err)
	}
	if sha256.Sum256(quietWAV) == sha256.Sum256(loudWAV) {
		return fmt.Errorf("different gains and targets produced identical output")
	}
	quietPlan, status, err := harness.getTimeline(ctx, "timeline_not_final_quiet")
	if err != nil || status != 200 {
		return fmt.Errorf("quiet plan read-back failed: %v", err)
	}
	loudPlan, status, err := harness.getTimeline(ctx, "timeline_not_final_loud")
	if err != nil || status != 200 {
		return fmt.Errorf("loud plan read-back failed: %v", err)
	}
	for name, pair := range map[string]struct {
		wav  []byte
		plan domain.Timeline
	}{
		"quiet": {quietWAV, quietPlan},
		"loud":  {loudWAV, loudPlan},
	} {
		control, err := fullRemixReference(ctx, pair.plan, sources)
		if err != nil {
			return err
		}
		if sha256.Sum256(pair.wav) != sha256.Sum256(control) {
			return fmt.Errorf("%s output differs from its own full remix", name)
		}
	}
	// A standalone cache proves normalization never writes into shared samples:
	// prepare one interval, observe it is unchanged after a normalizing render
	// reads and mutates its own mix buffer, and confirm a retained snapshot
	// still equals fresh selection-first conversion.
	wav := sources["reuse_mono_48"]
	layout, err := audiocodec.ParseWAVBytes(wav)
	if err != nil {
		return err
	}
	local := render.NewPrepareCache(render.NewBudget(4<<20), 1<<20)
	key := render.PreparedKey{
		SourceDigest:     digestOf(wav),
		SourceStartFrame: 1400,
		SourceEndFrame:   9400,
		SampleRate:       48000,
		Channels:         2,
	}
	payload := render.SourcePayload{Reader: bytes.NewReader(wav), Layout: layout}
	if _, release, err := local.Acquire(ctx, key, payload, render.BuildPreparedClipForCheck); err != nil {
		return err
	} else {
		release()
	}
	snapshot, ok := local.Snapshot(key)
	if !ok {
		return fmt.Errorf("prepared interval was not retained for the pollution comparison")
	}
	expected, err := render.PrepareReference(ctx, wav, key.SourceStartFrame, key.SourceEndFrame, 48000, 2)
	if err != nil {
		return err
	}
	// Run a normalizing mix over a copy of the snapshot; the cache must not
	// change even though the mix buffer it produced derives from it.
	mix := make([]float32, len(snapshot))
	copy(mix, snapshot)
	_ = render.NormalizeForCheck(mix, -6)
	if !samplesEqual(snapshot, expected.Samples) {
		return fmt.Errorf("retained prepared samples changed during normalization")
	}
	if _, still := local.Snapshot(key); !still {
		return fmt.Errorf("prepared interval disappeared during normalization")
	}
	afterSnapshot, _ := local.Snapshot(key)
	if !samplesEqual(afterSnapshot, expected.Samples) {
		return fmt.Errorf("retained prepared samples after normalization differ from fresh conversion")
	}
	return nil
}

// checkBudgetRefusalIsObservable starts the real HTTP surface with a budget too
// small for the requested timeline. The job must fail with the stable resource
// code, must not publish output, and must not disturb an existing completed
// render of the same source.
func checkBudgetRefusalIsObservable(ctx context.Context) error {
	sources, err := prepareReuseSources()
	if err != nil {
		return err
	}
	config := service.DefaultConfig()
	config.PrepareBudgetBytes = 48 << 10
	config.PrepareCacheBytes = 8 << 10
	harness, err := NewHarnessWithConfig(config)
	if err != nil {
		return err
	}
	defer harness.Close()
	if _, status, err := harness.createSource(ctx, "budget_source", "budget", sources["reuse_mono_48"]); err != nil || status != 201 {
		return fmt.Errorf("budget source setup failed: status=%d err=%v", status, err)
	}
	spans := []domain.TimelineSpan{{
		SourceID:         "budget_source",
		SourceStartFrame: 0,
		SourceEndFrame:   19000,
	}}
	if _, status, err := harness.createTimeline(ctx, "timeline_budget", "budget", 48000, 2, -1, spans); err != nil || status != 201 {
		return fmt.Errorf("budget timeline setup failed: status=%d err=%v", status, err)
	}
	job, status, err := harness.createRender(ctx, "render_budget", "timeline_budget")
	if err != nil {
		return err
	}
	if job.State != domain.RenderFailed || job.Error == nil {
		return fmt.Errorf("over-budget render was not refused: status=%d job=%+v", status, job)
	}
	if job.Error.Code != "render_memory_limit" {
		return fmt.Errorf("over-budget render failed with %q instead of render_memory_limit", job.Error.Code)
	}
	if job.Output != nil {
		return fmt.Errorf("refused render published output metadata")
	}
	outputStatus, outputBody, err := harness.requestJSON(ctx, "GET", "/v1/renders/"+job.ID+"/output", nil)
	if err != nil {
		return err
	}
	if outputStatus != 409 || failureCode(outputBody) != "render_not_completed" {
		return fmt.Errorf("refused render exposed output: status=%d code=%s", outputStatus, failureCode(outputBody))
	}
	return nil
}

func digestOf(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func mustDecode(wav []byte) *audiocodec.Clip {
	clip, _, err := audiocodec.DecodeWAV(wav)
	if err != nil {
		panic(err)
	}
	return clip
}

func samplesEqual(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if math.Float32bits(left[index]) != math.Float32bits(right[index]) {
			return false
		}
	}
	return true
}

func base64Encode(data []byte) string {
	const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	var builder bytes.Buffer
	for index := 0; index < len(data); index += 3 {
		var chunk [3]byte
		remaining := len(data) - index
		if remaining > 3 {
			remaining = 3
		}
		copy(chunk[:], data[index:index+remaining])
		builder.WriteByte(alphabet[chunk[0]>>2])
		builder.WriteByte(alphabet[(chunk[0]&0x03)<<4|chunk[1]>>4])
		if remaining > 1 {
			builder.WriteByte(alphabet[(chunk[1]&0x0F)<<2|chunk[2]>>6])
		} else {
			builder.WriteByte('=')
		}
		if remaining > 2 {
			builder.WriteByte(alphabet[chunk[2]&0x3F])
		} else {
			builder.WriteByte('=')
		}
	}
	return builder.String()
}

func firstByteDifference(left, right []byte) int {
	limit := len(left)
	if len(right) < limit {
		limit = len(right)
	}
	for index := 0; index < limit; index++ {
		if left[index] != right[index] {
			return index
		}
	}
	if len(left) != len(right) {
		return limit
	}
	return -1
}
