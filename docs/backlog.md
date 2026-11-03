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

## Verify rendered on-image text

**Problem:** with `text_mode: "rendered"` the image model draws `on_image_text`
into the image, and image models misspell. Nothing checks the result yet.

**Idea:** after each rendered image is generated, read the text back (a vision
model or OCR), compare it with `on_image_text` (case- and whitespace-
insensitive), and flag mismatches so the asset can't be approved as-is.

- Needs a place to store the result, e.g. a nullable `rendered_text_ok BOOLEAN`
  (and maybe `rendered_text_read VARCHAR`) on `asset`. That's a schema change:
  ask before running it against Supabase.
- The read-back call costs money: include it in the spend estimate.
- A mismatch could block approval the same way placeholders do.

## Product label check: what's still missing

`services/label_check.py` reads the label off each product-photo image and flags
`label_mismatch` when `brand_name` isn't on it. Not covered yet:

- Only the brand name is compared. The rest of the label (e.g. "PEPTIDE
  INFUSED", the size) is read and stored in `label_read` but not checked
  against the reference photo.
- The reader can misread small text too, so a correct image can be flagged.
  There's no way to override a flag after a person has checked the image;
  today the fix is to regenerate.

## Phase 4: report creative test results by the right key

Assets in a creative test share `test_group_id` and carry `test_mode`:

- `fixed_format`: only the angle varies, so compare and report results by `angle`.
- `varied_format`: angle and format both vary, so report results as
  `angle + format`. Never credit a difference to the angle alone, because the
  format may have caused it.
