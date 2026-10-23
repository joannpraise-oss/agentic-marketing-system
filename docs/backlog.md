# Backlog

## Retry failed uploads without paying for the image again

**Problem:** when a creative image is generated (and paid for) but its upload to
Supabase Storage fails, the asset is saved with `file_url = NULL` and the image
bytes are lost. Getting the image means generating, and paying for, it again.

**Idea:** keep Higgsfield's own result URL on the asset, so a later retry can
download from there and upload again.

- Add nullable `source_url` (Higgsfield `images[0].url`) and `provider_request_id`
  columns to `asset`. This is a schema change: write the `ALTER TABLE asset ...`
  and get approval before running it against Supabase.
- `higgsfield.generate_image` returns the result URL alongside the bytes; the
  generate endpoint saves it on every paid asset.
- Add a retry path (endpoint or job) for assets with `source_url` set and
  `file_url` NULL: download, `storage.upload_asset`, set `file_url`.
- Higgsfield keeps outputs for at least 7 days, then may delete them, so retries
  must happen within that window. After that, the asset can only be regenerated.
