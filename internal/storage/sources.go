package storage

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"path/filepath"

	"voice-timeline-renderer/internal/domain"
)

func (repo *Repository) SaveSource(asset domain.SourceAsset, wav []byte) error {
	repo.mu.Lock()
	defer repo.mu.Unlock()

	final := repo.path("sources", asset.ID)
	if _, err := os.Stat(final); err == nil {
		return domain.E(domain.KindConflict, "source_conflict", "source_id", "source identifier already exists")
	} else if !errors.Is(err, os.ErrNotExist) {
		return domain.Wrap(domain.KindInternal, "storage_error", "", "cannot inspect source destination", err)
	}
	temp, err := os.MkdirTemp(repo.path("tmp"), "source-*")
	if err != nil {
		return domain.Wrap(domain.KindInternal, "storage_error", "", "cannot create source staging directory", err)
	}
	defer func() {
		_ = os.RemoveAll(temp)
	}()
	if err := writeBytes(filepath.Join(temp, "source.wav"), wav, 0o640); err != nil {
		return domain.Wrap(domain.KindInternal, "storage_error", "", "cannot write source media", err)
	}
	if err := writeJSONAtomic(filepath.Join(temp, "metadata.json"), asset, 0o640); err != nil {
		return domain.Wrap(domain.KindInternal, "storage_error", "", "cannot write source metadata", err)
	}
	if err := os.Rename(temp, final); err != nil {
		return domain.Wrap(domain.KindInternal, "storage_error", "", "cannot commit source", err)
	}
	if err := syncDirectory(repo.path("sources")); err != nil {
		return domain.Wrap(domain.KindInternal, "storage_error", "", "cannot sync source index", err)
	}
	return nil
}

func (repo *Repository) GetSourceMeta(id string) (domain.SourceAsset, error) {
	repo.mu.RLock()
	defer repo.mu.RUnlock()
	var asset domain.SourceAsset
	if err := readJSON(repo.path("sources", id, "metadata.json"), &asset); err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return domain.SourceAsset{}, domain.E(domain.KindNotFound, "source_not_found", "source_id", "source does not exist")
		}
		return domain.SourceAsset{}, domain.Wrap(domain.KindInternal, "storage_error", "", "cannot read source metadata", err)
	}
	return asset, nil
}

// SourceReader is an immutable, digest-verified source payload. Frame ranges can
// be read without keeping the whole source resident. Every access re-reads and
// re-hashes the full bytes, so damaged storage bytes are reported on every use
// rather than being masked by an earlier success.
type SourceReader struct {
	asset  domain.SourceAsset
	path   string
	handle *os.File
	size   int64
}

func (source *SourceReader) Asset() domain.SourceAsset { return source.asset }

func (source *SourceReader) Size() int64 { return source.size }

func (source *SourceReader) ReadAt(target []byte, offset int64) (int, error) {
	return source.handle.ReadAt(target, offset)
}

func (source *SourceReader) Close() error {
	if source == nil || source.handle == nil {
		return nil
	}
	return source.handle.Close()
}

// OpenSource loads source metadata, opens the media file, and verifies the
// stored bytes against the recorded digest before the reader is returned.
func (repo *Repository) OpenSource(id string) (*SourceReader, error) {
	asset, err := repo.GetSourceMeta(id)
	if err != nil {
		return nil, err
	}
	handle, err := os.Open(repo.path("sources", id, "source.wav"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, domain.E(domain.KindNotFound, "source_not_found", "source_id", "source media does not exist")
		}
		return nil, domain.Wrap(domain.KindInternal, "storage_error", "", "cannot open source media", err)
	}
	info, err := handle.Stat()
	if err != nil {
		_ = handle.Close()
		return nil, domain.Wrap(domain.KindInternal, "storage_error", "", "cannot inspect source media", err)
	}
	hasher := sha256.New()
	if _, err := io.Copy(hasher, handle); err != nil {
		_ = handle.Close()
		return nil, domain.Wrap(domain.KindInternal, "storage_error", "", "cannot verify source media", err)
	}
	if hex.EncodeToString(hasher.Sum(nil)) != asset.Digest {
		_ = handle.Close()
		return nil, domain.E(domain.KindInternal, "source_integrity_error", "source_id", "stored source digest does not match metadata")
	}
	return &SourceReader{
		asset:  asset,
		path:   repo.path("sources", id, "source.wav"),
		handle: handle,
		size:   info.Size(),
	}, nil
}

// ReadSource returns the whole verified payload. It is used by callers that
// already bound the source size, such as ingestion idempotency checks.
func (repo *Repository) ReadSource(id string) (domain.SourceAsset, []byte, error) {
	reader, err := repo.OpenSource(id)
	if err != nil {
		return domain.SourceAsset{}, nil, err
	}
	defer reader.Close()
	payload, err := io.ReadAll(io.NewSectionReader(reader, 0, reader.Size()))
	if err != nil {
		return domain.SourceAsset{}, nil, domain.Wrap(domain.KindInternal, "storage_error", "", "cannot read source media", err)
	}
	return reader.Asset(), payload, nil
}
