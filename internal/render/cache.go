package render

import (
	"context"
	"errors"
	"io"
	"sync"

	"voice-timeline-renderer/internal/audiocodec"
)

// ErrPrepareMemory reports that preparation or mix work does not fit in the
// configured memory budget. It is a resource refusal, not a media failure.
var ErrPrepareMemory = errors.New("audio preparation exceeds the configured memory budget")

// Budget is one process-wide allowance for a render's estimated working set:
// its prepared intervals, mix buffer, and encoded output. A render reserves the
// whole estimate in one atomic acquisition before allocating anything, so two
// renders can never hold part of each other's working set and wait on each
// other. Retained cache entries and active renders draw from this same budget,
// which is why the service keeps cache ceiling plus active estimate under its
// resident limit.
type Budget struct {
	mu       sync.Mutex
	capacity int64
	used     int64
	notify   chan struct{}
}

func NewBudget(capacity int64) *Budget {
	if capacity < 0 {
		capacity = 0
	}
	return &Budget{capacity: capacity, notify: make(chan struct{})}
}

// Acquire reserves the whole byte estimate atomically. It waits for another
// render to finish only when that render holds memory; it never waits on
// memory this render itself holds, because nothing has been reserved yet.
func (budget *Budget) Acquire(ctx context.Context, bytes int64) error {
	if budget == nil {
		return nil
	}
	if bytes < 0 {
		bytes = 0
	}
	if bytes > budget.capacity {
		return ErrPrepareMemory
	}
	for {
		budget.mu.Lock()
		if budget.used+bytes <= budget.capacity {
			budget.used += bytes
			budget.mu.Unlock()
			return nil
		}
		wait := budget.notify
		budget.mu.Unlock()
		select {
		case <-wait:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

func (budget *Budget) Release(bytes int64) {
	if budget == nil || bytes <= 0 {
		return
	}
	budget.mu.Lock()
	budget.used -= bytes
	if budget.used < 0 {
		budget.used = 0
	}
	close(budget.notify)
	budget.notify = make(chan struct{})
	budget.mu.Unlock()
}

func (budget *Budget) Used() int64 {
	if budget == nil {
		return 0
	}
	budget.mu.Lock()
	defer budget.mu.Unlock()
	return budget.used
}

func (budget *Budget) Capacity() int64 {
	if budget == nil {
		return 0
	}
	return budget.capacity
}

// PreparedKey identifies one exact prepared sound. The source interval is part
// of the identity because selection happens before resampling; two spans that
// merely overlap have different edge samples and must not share a result.
type PreparedKey struct {
	SourceDigest     string
	SourceStartFrame int64
	SourceEndFrame   int64
	SampleRate       int
	Channels         int
}

// SourcePayload carries the verified random-access source bytes and layout a
// build reads from.
type SourcePayload struct {
	Reader io.ReaderAt
	Layout audiocodec.WAVLayout
}

// PrepareBuilder converts a verified source interval into the clip identified
// by key. The cache calls it once per key while any caller still waits.
type PrepareBuilder func(ctx context.Context, key PreparedKey, source SourcePayload) (*audiocodec.Clip, error)

// PrepareStats is a snapshot of preparation reuse and memory. Hits are reuse of
// an already prepared immutable clip; builds are actual conversions.
type PrepareStats struct {
	Requests      int64
	Builds        int64
	BuildFailures int64
	Hits          int64
	Waits         int64
	Evictions     int64
	Transient     int64
	ReusedPCM     int64
	Entries       int
	CachedBytes   int64
	BudgetUsed    int64
	BudgetBytes   int64
	CacheBytes    int64
}

type prepareEntry struct {
	key       PreparedKey
	done      chan struct{}
	cancel    context.CancelFunc
	clip      *audiocodec.Clip
	err       error
	frames    int64
	bytes     int64
	pins      int
	settled   bool
	abandoned bool
	transient bool
}

// PrepareCache shares immutable prepared intervals between renders. Cached
// samples are never modified in place: callers only read them, eviction drops
// the cache's reference, and the mix plus normalization work on separate
// buffers.
type PrepareCache struct {
	budget   *Budget
	maxBytes int64

	mu      sync.Mutex
	entries map[PreparedKey]*prepareEntry
	order   []*prepareEntry
	total   int64
	stats   PrepareStats

	buildGroup sync.WaitGroup
}

func NewPrepareCache(budget *Budget, maxBytes int64) *PrepareCache {
	if maxBytes < 0 {
		maxBytes = 0
	}
	if budget != nil && maxBytes > budget.Capacity() {
		maxBytes = budget.Capacity()
	}
	return &PrepareCache{
		budget:   budget,
		maxBytes: maxBytes,
		entries:  make(map[PreparedKey]*prepareEntry),
	}
}

func (cache *PrepareCache) Stats() PrepareStats {
	if cache == nil {
		return PrepareStats{}
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	stats := cache.stats
	stats.Entries = len(cache.entries)
	stats.CachedBytes = cache.total
	stats.CacheBytes = cache.maxBytes
	if cache.budget != nil {
		stats.BudgetUsed = cache.budget.Used()
		stats.BudgetBytes = cache.budget.Capacity()
	}
	return stats
}

// Baseline lets a render measure what one execution actually reused.
func (cache *PrepareCache) Baseline() (hits, reusedPCM int64) {
	if cache == nil {
		return 0, 0
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	return cache.stats.Hits, cache.stats.ReusedPCM
}

// Acquire returns a prepared clip for key plus a release function. The clip is
// shared and read-only. Cancelling ctx stops this caller waiting; a build still
// needed by another caller continues, and a build with no remaining callers is
// cancelled.
func (cache *PrepareCache) Acquire(
	ctx context.Context,
	key PreparedKey,
	source SourcePayload,
	build PrepareBuilder,
) (*audiocodec.Clip, func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if cache == nil {
		clip, err := build(ctx, key, source)
		if err != nil {
			return nil, nil, err
		}
		return clip, func() {}, nil
	}

	for {
		cache.mu.Lock()
		if entry, exists := cache.entries[key]; exists {
			entry.pins++
			cache.stats.Requests++
			if entry.settled {
				cache.stats.Hits++
				cache.stats.ReusedPCM += entry.frames
			} else {
				cache.stats.Waits++
			}
			cache.mu.Unlock()
			return cache.wait(ctx, key, entry)
		}

		buildContext, cancel := context.WithCancel(context.Background())
		entry := &prepareEntry{
			key:    key,
			done:   make(chan struct{}),
			cancel: cancel,
			pins:   1,
		}
		cache.entries[key] = entry
		cache.stats.Requests++
		cache.stats.Builds++
		cache.mu.Unlock()

		cache.buildGroup.Add(1)
		go func() {
			defer cache.buildGroup.Done()
			cache.runBuild(buildContext, key, source, build, entry)
		}()
		return cache.wait(ctx, key, entry)
	}
}

func (cache *PrepareCache) wait(
	ctx context.Context,
	key PreparedKey,
	entry *prepareEntry,
) (*audiocodec.Clip, func(), error) {
	select {
	case <-entry.done:
		cache.mu.Lock()
		err := entry.err
		clip := entry.clip
		cache.mu.Unlock()
		if err != nil {
			return nil, nil, err
		}
		return clip, cache.releaseFunc(entry), nil
	case <-ctx.Done():
		cache.mu.Lock()
		entry.pins--
		if entry.pins == 0 && !entry.settled && !entry.abandoned {
			entry.abandoned = true
			if cache.entries[key] == entry {
				delete(cache.entries, key)
			}
			entry.cancel()
		}
		cache.mu.Unlock()
		return nil, nil, ctx.Err()
	}
}

func (cache *PrepareCache) runBuild(
	ctx context.Context,
	key PreparedKey,
	source SourcePayload,
	build PrepareBuilder,
	entry *prepareEntry,
) {
	// The render that requested this build already reserved its estimated
	// working set, so the entry itself consumes no additional budget. The
	// cache ceiling separately bounds how much prepared sound is retained.
	clip, buildErr := build(ctx, key, source)

	cache.mu.Lock()
	if entry.abandoned {
		cache.mu.Unlock()
		close(entry.done)
		return
	}
	if buildErr != nil {
		delete(cache.entries, key)
		entry.err = buildErr
		entry.settled = true
		cache.stats.BuildFailures++
		cache.mu.Unlock()
		close(entry.done)
		return
	}
	actual := int64(len(clip.Samples)) * 4
	entry.clip = clip
	entry.frames = clip.FrameCount()
	entry.bytes = actual
	entry.settled = true

	// Retain only entries that fit the ceiling on their own, after recycling
	// unpinned entries. A larger interval is still served to its callers but
	// is not retained; it stays shared only while pinned.
	admitted := actual > 0 && actual <= cache.maxBytes
	if admitted {
		cache.evictLocked(actual)
		admitted = cache.total+actual <= cache.maxBytes
	}
	if admitted {
		cache.total += actual
		cache.order = append(cache.order, entry)
	} else {
		entry.transient = true
		delete(cache.entries, key)
		cache.stats.Transient++
	}
	cache.mu.Unlock()
	close(entry.done)
}

// releaseFunc drops one pin. A transient (never-retained) entry is already
// absent from the map, so dropping the last pin simply makes it collectable;
// nothing about the samples is rewritten here.
func (cache *PrepareCache) releaseFunc(entry *prepareEntry) func() {
	return func() {
		cache.mu.Lock()
		if entry.pins > 0 {
			entry.pins--
		}
		cache.mu.Unlock()
	}
}

// evictLocked makes room for an entry of need bytes inside the cache ceiling by
// dropping oldest unpinned entries. Pinned entries are never removed because a
// live render may still be reading them.
func (cache *PrepareCache) evictLocked(need int64) {
	overflow := cache.total + need - cache.maxBytes
	if overflow > 0 {
		cache.evictForLocked(overflow)
	}
}

func (cache *PrepareCache) evictForLocked(need int64) {
	freed := int64(0)
	for freed < need && len(cache.order) > 0 {
		index := -1
		for candidate := 0; candidate < len(cache.order); candidate++ {
			entry := cache.order[candidate]
			if entry.pins == 0 && !entry.transient && entry.settled {
				index = candidate
				break
			}
		}
		if index < 0 {
			return
		}
		entry := cache.order[index]
		cache.order = append(cache.order[:index], cache.order[index+1:]...)
		delete(cache.entries, entry.key)
		cache.total -= entry.bytes
		freed += entry.bytes
		cache.stats.Evictions++
	}
}

// Snapshot returns a copy of the cached samples for a key, if that exact
// interval is retained. It lets a verification harness prove a cached entry
// still matches fresh conversion after other renders normalized their outputs.
func (cache *PrepareCache) Snapshot(key PreparedKey) ([]float32, bool) {
	if cache == nil {
		return nil, false
	}
	cache.mu.Lock()
	defer cache.mu.Unlock()
	entry, exists := cache.entries[key]
	if !exists || !entry.settled || entry.err != nil {
		return nil, false
	}
	samples := make([]float32, len(entry.clip.Samples))
	copy(samples, entry.clip.Samples)
	return samples, true
}
