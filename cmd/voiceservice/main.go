package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"math"
	"net/http"
	"os"
	"os/signal"
	"runtime/debug"
	"syscall"
	"time"

	"voice-timeline-renderer/internal/httpapi"
	"voice-timeline-renderer/internal/service"
	"voice-timeline-renderer/internal/storage"
)

func main() {
	address := flag.String("addr", ":8093", "HTTP listen address")
	dataDir := flag.String("data", "./var/voice-timeline", "persistent data directory")
	cacheMiB := flag.Int("prepare-cache-mib", 36, "preparation cache ceiling in MiB")
	budgetMiB := flag.Int("prepare-budget-mib", 112, "shared active-render budget in MiB")
	flag.Parse()

	// Cache plus active work must fit under the service's resident ceiling.
	// The Go runtime keeps some heap slack, so cap it explicitly; a garbage
	// collection target below the hard limit keeps the resident set bounded.
	const residentCeiling = 192 << 20
	if *cacheMiB <= 0 || *budgetMiB <= 0 {
		slog.Error("preparation memory settings must be positive")
		os.Exit(2)
	}
	if int64(*cacheMiB+*budgetMiB)<<20 > residentCeiling {
		slog.Error("preparation cache plus active budget exceeds the resident ceiling",
			"cache_mib", *cacheMiB, "budget_mib", *budgetMiB, "ceiling_mib", residentCeiling>>20)
		os.Exit(2)
	}
	goMemoryLimit := int64(math.Round(float64(residentCeiling) * 0.8))
	debug.SetMemoryLimit(goMemoryLimit)

	if err := os.MkdirAll(*dataDir, 0o750); err != nil {
		slog.Error("cannot create data directory", "error", err)
		os.Exit(1)
	}
	repo, err := storage.NewRepository(*dataDir)
	if err != nil {
		slog.Error("cannot initialize repository", "error", err)
		os.Exit(1)
	}
	config := service.DefaultConfig()
	config.PrepareCacheBytes = int64(*cacheMiB) << 20
	config.PrepareBudgetBytes = int64(*budgetMiB) << 20
	serviceLayer := service.New(repo, config)
	api := httpapi.New(serviceLayer, httpapi.Options{
		MaxSourceRequestBytes: 48 << 20,
		MaxJSONBytes:          2 << 20,
	})
	httpServer := &http.Server{
		Addr:              *address,
		Handler:           api.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       60 * time.Second,
		WriteTimeout:      180 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	runContext, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		<-runContext.Done()
		shutdownContext, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdownContext)
	}()

	slog.Info(
		"voice timeline service listening",
		"addr", *address,
		"data", *dataDir,
		"prepare_cache_bytes", config.PrepareCacheBytes,
		"prepare_budget_bytes", config.PrepareBudgetBytes,
		"go_memory_limit_bytes", goMemoryLimit,
		"resident_ceiling_bytes", residentCeiling,
	)
	if err := httpServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		slog.Error("HTTP service stopped", "error", err)
		os.Exit(1)
	}
}
