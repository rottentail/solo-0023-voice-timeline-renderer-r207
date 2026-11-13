package audiocodec

import (
	"context"
	"errors"
	"math"
)

// PrepareFrames produces the exact samples obtained by selecting
// [startFrame, endFrame) from the decoded source at its own sample grid and
// only then resampling that selection to targetRate and mapping it to
// outputChannels.
//
// The order matters. Resampling a whole source and cropping it afterwards
// samples a different interpolation grid at the selection edges, which can
// make short selections and splice points audibly different. This function is
// therefore a fused implementation of SliceFrames, Resample, and MapChannels,
// with the same float32 operations in the same order.
//
// The returned samples are fresh memory owned by the caller. Callers that
// place the result in a cache must treat the sample slice as immutable.
func PrepareFrames(
	ctx context.Context,
	clip *Clip,
	startFrame int64,
	endFrame int64,
	targetRate int,
	outputChannels int,
) (*Clip, error) {
	if clip == nil || clip.Channels <= 0 {
		return nil, errors.New("clip is empty")
	}
	if targetRate < 8000 || targetRate > 192000 {
		return nil, errors.New("target sample rate is outside the accepted range")
	}
	if outputChannels != 1 && outputChannels != 2 {
		return nil, errors.New("output channel count must be one or two")
	}
	sourceFrames := clip.FrameCount()
	if startFrame < 0 || endFrame <= startFrame || endFrame > sourceFrames {
		return nil, errors.New("frame interval is outside the clip")
	}

	selectedFrames := endFrame - startFrame
	targetFrames := int64(math.Round(float64(selectedFrames) * float64(targetRate) / float64(clip.SampleRate)))
	if targetFrames < 1 {
		targetFrames = 1
	}
	if targetFrames > int64(maxInt()/outputChannels) {
		return nil, errors.New("prepared clip is too large for this process")
	}
	samples := make([]float32, int(targetFrames)*outputChannels)
	lastSourceFrame := selectedFrames - 1

	// Matching rates are a verbatim copy in the existing transform pipeline;
	// preserve that path so no interpolation arithmetic is introduced.
	if clip.SampleRate == targetRate && targetFrames == selectedFrames {
		for frame := int64(0); frame < selectedFrames; frame++ {
			if frame%4096 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			sourceBase := int(startFrame+frame) * clip.Channels
			targetBase := int(frame) * outputChannels
			switch {
			case clip.Channels == outputChannels:
				copy(samples[targetBase:targetBase+outputChannels], clip.Samples[sourceBase:sourceBase+clip.Channels])
			case clip.Channels == 1:
				value := clip.Samples[sourceBase]
				samples[targetBase] = value
				samples[targetBase+1] = value
			default:
				samples[targetBase] = (clip.Samples[sourceBase] + clip.Samples[sourceBase+1]) * 0.5
			}
		}
		return &Clip{SampleRate: targetRate, Channels: outputChannels, Samples: samples}, nil
	}

	step := float64(clip.SampleRate) / float64(targetRate)
	for frame := int64(0); frame < targetFrames; frame++ {
		if frame%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		position := float64(frame) * step
		left := int64(position)
		if left >= lastSourceFrame {
			left = lastSourceFrame
		}
		right := left + 1
		if right >= selectedFrames {
			right = lastSourceFrame
		}
		weight := float32(position - float64(left))
		sourceLeft := int(startFrame+left) * clip.Channels
		sourceRight := int(startFrame+right) * clip.Channels
		targetBase := int(frame) * outputChannels

		switch {
		case clip.Channels == 1 && outputChannels == 1:
			first := clip.Samples[sourceLeft]
			second := clip.Samples[sourceRight]
			samples[targetBase] = first + (second-first)*weight
		case clip.Channels == 1 && outputChannels == 2:
			first := clip.Samples[sourceLeft]
			second := clip.Samples[sourceRight]
			value := first + (second-first)*weight
			samples[targetBase] = value
			samples[targetBase+1] = value
		case clip.Channels == 2 && outputChannels == 2:
			for channel := 0; channel < 2; channel++ {
				first := clip.Samples[sourceLeft+channel]
				second := clip.Samples[sourceRight+channel]
				samples[targetBase+channel] = first + (second-first)*weight
			}
		default:
			leftFirst := clip.Samples[sourceLeft]
			leftSecond := clip.Samples[sourceRight]
			rightFirst := clip.Samples[sourceLeft+1]
			rightSecond := clip.Samples[sourceRight+1]
			leftValue := leftFirst + (leftSecond-leftFirst)*weight
			rightValue := rightFirst + (rightSecond-rightFirst)*weight
			samples[targetBase] = (leftValue + rightValue) * 0.5
		}
	}
	return &Clip{SampleRate: targetRate, Channels: outputChannels, Samples: samples}, nil
}

func maxInt() int {
	return int(^uint(0) >> 1)
}
