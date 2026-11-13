package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"voice-timeline-renderer/internal/checkflow"
)

func main() {
	workflow := flag.String("workflow", "", "workflow identifier to execute")
	stats := flag.Bool("stats", false, "print the preparation reuse snapshot")
	flag.Parse()
	if *workflow == "" {
		fmt.Fprintln(os.Stderr, "-workflow is required")
		os.Exit(2)
	}
	snapshot, err := checkflow.RunWithStats(context.Background(), *workflow)
	if err != nil {
		fmt.Fprintf(os.Stderr, "workflow %s failed: %v\n", *workflow, err)
		os.Exit(1)
	}
	if *stats {
		fmt.Printf("preparation builds=%d hits=%d waits=%d evictions=%d transient=%d reused_pcm_frames=%d cached_bytes=%d budget=%d/%d\n",
			snapshot.Builds, snapshot.Hits, snapshot.Waits, snapshot.Evictions, snapshot.Transient,
			snapshot.ReusedPCM, snapshot.CachedBytes, snapshot.BudgetUsed, snapshot.BudgetBytes)
	}
	fmt.Printf("workflow %s passed\n", *workflow)
}
