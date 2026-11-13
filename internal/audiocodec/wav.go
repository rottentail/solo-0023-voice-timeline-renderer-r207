package audiocodec

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

const (
	pcmFormatCode = 1
	pcmBitWidth   = 16
)

// WAVLayout describes where the PCM payload lives inside a validated corpus of
// WAV bytes. It allows a caller to decode any frame interval without holding or
// decoding the whole source.
type WAVLayout struct {
	WAVInfo
	DataOffset int64
	DataLength int64
}

func (layout WAVLayout) blockWidth() int64 {
	return int64(layout.Channels) * 2
}

// ParseWAV reads the RIFF container structure through reader without reading
// the sample payload. size is the total byte length of the file.
func ParseWAV(reader io.ReaderAt, size int64) (WAVLayout, error) {
	if size < 12 {
		return WAVLayout{}, errors.New("file is shorter than a RIFF header")
	}
	header := make([]byte, 12)
	if _, err := reader.ReadAt(header, 0); err != nil {
		return WAVLayout{}, err
	}
	if string(header[0:4]) != "RIFF" || string(header[8:12]) != "WAVE" {
		return WAVLayout{}, errors.New("file is not a RIFF/WAVE payload")
	}
	riffSize := uint64(binary.LittleEndian.Uint32(header[4:8]))
	if riffSize+8 > uint64(size) {
		return WAVLayout{}, errors.New("RIFF size exceeds the supplied payload")
	}

	var formatChunk []byte
	formatFound := false
	dataOffset, dataLength := int64(-1), int64(0)
	offset := int64(12)
	chunkHeader := make([]byte, 8)
	for offset+8 <= size {
		if _, err := reader.ReadAt(chunkHeader, offset); err != nil {
			return WAVLayout{}, err
		}
		chunkID := string(chunkHeader[0:4])
		chunkSize := int64(binary.LittleEndian.Uint32(chunkHeader[4:8]))
		payloadStart := offset + 8
		payloadEnd := payloadStart + chunkSize
		if chunkSize < 0 || payloadEnd < payloadStart || payloadEnd > size {
			return WAVLayout{}, fmt.Errorf("chunk %q exceeds the payload", chunkID)
		}
		switch chunkID {
		case "fmt ":
			if chunkSize > 1<<16 {
				return WAVLayout{}, errors.New("fmt chunk is unreasonably large")
			}
			formatChunk = make([]byte, chunkSize)
			if _, err := reader.ReadAt(formatChunk, payloadStart); err != nil {
				return WAVLayout{}, err
			}
			formatFound = true
		case "data":
			dataOffset = payloadStart
			dataLength = chunkSize
		}
		offset = payloadEnd
		if chunkSize%2 == 1 {
			offset++
		}
	}
	if !formatFound || len(formatChunk) < 16 {
		return WAVLayout{}, errors.New("fmt chunk is missing or too short")
	}
	if dataOffset < 0 {
		return WAVLayout{}, errors.New("data chunk is missing")
	}

	audioFormat := binary.LittleEndian.Uint16(formatChunk[0:2])
	channels := int(binary.LittleEndian.Uint16(formatChunk[2:4]))
	sampleRate := int(binary.LittleEndian.Uint32(formatChunk[4:8]))
	byteRate := binary.LittleEndian.Uint32(formatChunk[8:12])
	blockAlign := int(binary.LittleEndian.Uint16(formatChunk[12:14]))
	bitsPerSample := int(binary.LittleEndian.Uint16(formatChunk[14:16]))

	if audioFormat != pcmFormatCode {
		return WAVLayout{}, fmt.Errorf("unsupported WAV format code %d", audioFormat)
	}
	if bitsPerSample != pcmBitWidth {
		return WAVLayout{}, fmt.Errorf("unsupported PCM width %d", bitsPerSample)
	}
	if channels != 1 && channels != 2 {
		return WAVLayout{}, fmt.Errorf("unsupported channel count %d", channels)
	}
	if sampleRate < 8000 || sampleRate > 192000 {
		return WAVLayout{}, fmt.Errorf("sample rate %d is outside the accepted range", sampleRate)
	}
	expectedBlockAlign := channels * (bitsPerSample / 8)
	if blockAlign != expectedBlockAlign {
		return WAVLayout{}, errors.New("block alignment does not match channels and sample width")
	}
	expectedByteRate := uint32(sampleRate * blockAlign)
	if byteRate != expectedByteRate {
		return WAVLayout{}, errors.New("byte rate does not match sample rate and block alignment")
	}
	if dataLength == 0 || dataLength%int64(blockAlign) != 0 {
		return WAVLayout{}, errors.New("sample data is empty or not frame-aligned")
	}

	return WAVLayout{
		WAVInfo: WAVInfo{
			SampleRate:    sampleRate,
			Channels:      channels,
			BitsPerSample: bitsPerSample,
			FrameCount:    dataLength / int64(blockAlign),
			ByteCount:     size,
		},
		DataOffset: dataOffset,
		DataLength: dataLength,
	}, nil
}

func ParseWAVBytes(data []byte) (WAVLayout, error) {
	return ParseWAV(bytes.NewReader(data), int64(len(data)))
}

// DecodeWAV decodes the full payload. It retains the whole-file behavior used
// by ingestion, which needs signal measurements over every frame.
func DecodeWAV(data []byte) (*Clip, WAVInfo, error) {
	layout, err := ParseWAVBytes(data)
	if err != nil {
		return nil, WAVInfo{}, err
	}
	clip, err := decodeFrames(context.Background(), bytes.NewReader(data), layout, 0, layout.FrameCount, 4096)
	if err != nil {
		return nil, WAVInfo{}, err
	}
	return clip, layout.WAVInfo, nil
}

// DecodeWAVRangeReader decodes selected frames through a random-access reader.
// Only the selected PCM bytes are read, so a large source does not need to be
// resident in memory.
func DecodeWAVRangeReader(
	ctx context.Context,
	reader io.ReaderAt,
	layout WAVLayout,
	startFrame, endFrame int64,
) (*Clip, error) {
	return decodeFrames(ctx, reader, layout, startFrame, endFrame, 8192)
}

func decodeFrames(
	ctx context.Context,
	reader io.ReaderAt,
	layout WAVLayout,
	startFrame, endFrame int64,
	checkEvery int64,
) (*Clip, error) {
	if startFrame < 0 || endFrame <= startFrame || endFrame > layout.FrameCount {
		return nil, errors.New("frame interval is outside the clip")
	}
	width := layout.blockWidth()
	byteCount := (endFrame - startFrame) * width
	if byteCount > int64(^uint(0)>>1)/4 {
		return nil, errors.New("decoded interval is too large for this process")
	}
	raw := make([]byte, byteCount)
	if _, err := reader.ReadAt(raw, layout.DataOffset+startFrame*width); err != nil {
		return nil, err
	}
	samples := make([]float32, len(raw)/2)
	for index := range samples {
		if int64(index) == 0 || int64(index)%checkEvery == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		value := int16(binary.LittleEndian.Uint16(raw[index*2 : index*2+2]))
		if value < 0 {
			samples[index] = float32(value) / 32768
		} else {
			samples[index] = float32(value) / 32767
		}
	}
	return &Clip{SampleRate: layout.SampleRate, Channels: layout.Channels, Samples: samples}, nil
}

// MeasureWAVPCM16 measures 16-bit PCM samples by streaming the payload in
// chunks instead of materializing a decoded float32 clip. The numeric result
// matches DecodeWAV followed by Measure, using the same conversion and the same
// summation order.
func MeasureWAVPCM16(reader io.ReaderAt, layout WAVLayout) (Metrics, error) {
	peak := 0.0
	sumSquares := 0.0
	sampleCount := int(layout.FrameCount) * layout.Channels
	const chunkBytes = 1 << 20
	buffer := make([]byte, 0, chunkBytes)
	if chunk := layout.DataLength; chunk < chunkBytes {
		buffer = make([]byte, 0, chunk)
	}
	position := int64(0)
	for index := 0; index < sampleCount; index++ {
		byteOffset := int64(index) * 2
		if byteOffset < position || byteOffset+2 > position+int64(len(buffer)) {
			readStart := byteOffset
			readLength := int64(chunkBytes)
			if remaining := layout.DataLength - readStart; remaining < readLength {
				readLength = remaining
			}
			if readLength <= 0 {
				break
			}
			buffer = buffer[:cap(buffer)][:readLength]
			if _, err := reader.ReadAt(buffer, layout.DataOffset+readStart); err != nil {
				return Metrics{}, err
			}
			position = readStart
		}
		raw := buffer[byteOffset-position : byteOffset-position+2]
		value := int16(binary.LittleEndian.Uint16(raw))
		var sample float32
		if value < 0 {
			sample = float32(value) / 32768
		} else {
			sample = float32(value) / 32767
		}
		converted := float64(sample)
		absolute := converted
		if absolute < 0 {
			absolute = -absolute
		}
		if absolute > peak {
			peak = absolute
		}
		sumSquares += converted * converted
	}
	rms := 0.0
	if sampleCount > 0 {
		rms = math.Sqrt(sumSquares / float64(sampleCount))
	}
	return Metrics{
		PeakDBFS: amplitudeDBFS(peak),
		RMSDBFS:  amplitudeDBFS(rms),
	}, nil
}

func MeasureWAVPCM16Bytes(data []byte) (Metrics, WAVInfo, error) {
	layout, err := ParseWAVBytes(data)
	if err != nil {
		return Metrics{}, WAVInfo{}, err
	}
	metrics, err := MeasureWAVPCM16(bytes.NewReader(data), layout)
	if err != nil {
		return Metrics{}, WAVInfo{}, err
	}
	return metrics, layout.WAVInfo, nil
}
