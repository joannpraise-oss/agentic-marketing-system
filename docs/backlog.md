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

`services/label_check.py` reads the label off the uploaded photo once (saved as
`labels/<photo>.txt` in the private bucket) and compares each generated image's
label with it. Three outcomes: an exact match passes; readable text that differs
in any word, number or punctuation is `label_mismatch`; text too small to read
passes as "unreadable". The brand name must always be readable and exact. A read
that disagrees is retried once; both reads go in `asset.label_read`. Not covered yet:

- The outcome (match / unreadable / mismatch) isn't stored: `label_ok` is True
  for both match and unreadable. Storing it needs a column (schema change: ask).
- The reader can still misread twice, on the generated image or on the
  reference photo itself, so a correct image can be flagged. There's no way to
  override a flag after a person has checked the image, or to correct a
  misread reference; today the fix is to regenerate, or upload a sharper photo.
- Which words mismatched is only logged, not stored on the asset.
- Reading the photo at upload (~$0.004) isn't counted against the spend caps.

## Premium option: 2k product images

At true real-world size, a hand-held 7 in bottle's small print (ingredient
badges, "8 fl oz (237 mL)") is only a few pixels tall at Marketing Studio's 1k,
so it garbles or can't be read. 2k would make it legible. Offer it later as a
premium setting (e.g. per request or per plan), not the default:

- Price `resolution: "2k"` with the free `/estimate` first and add it to
  `higgsfield.product_image_cost`, so the spend check reserves the real price.
- Label reads at 2k cost more too (more image tokens): re-measure `RESERVE_USD`.

## Phase 4: report creative test results by the right key

Assets in a creative test share `test_group_id` and carry `test_mode`:

- `fixed_format`: only the angle varies, so compare and report results by `angle`.
- `varied_format`: angle and format both vary, so report results as
  `angle + format`. Never credit a difference to the angle alone, because the
  format may have caused it.
