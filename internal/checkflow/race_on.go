//go:build race

package checkflow

// raceInstrumented reports that the binary carries the race detector, whose
// shadow memory dominates the resident set and is not service-accounted.
func raceInstrumented() bool { return true }
