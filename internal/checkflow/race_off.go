//go:build !race

package checkflow

func raceInstrumented() bool { return false }
