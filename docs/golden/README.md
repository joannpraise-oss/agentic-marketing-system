# Golden examples

Reference images for judging realism. When a prompt change is tested, compare its output with
these; a change that looks worse than them is a regression.

Both were made on 2026-11-02 (21:31 PST) at commit `bcecaa5`, for J’aduné's peptide-infused
hydrating hair mist, with the uploaded product photo. The realism wording they used is
`REALISM_PROMPT` in `services/creative.py` again:

> Real photographed human, not AI-perfect: visible pores and fine skin texture, natural
> unevenness, freckles/moles/spots where natural, healthy glow with real shadows and undertones,
> no airbrushing, natural camera grain.

## #25 — ugc (`25-ugc.png`)

What makes it good:

- The skin looks photographed, not rendered. You can see pores and texture on the nose and
  cheeks, a few small marks, uneven tone around the eyes and mouth, and fine hair at the hairline.
  The shine sits only on the high points.
- The hair is real: frizz, flyaways, strands that aren't symmetrical. The hand runs through it
  naturally.
- The light comes from a real window, with soft falloff across the face and a shadow under the
  jaw.
- It's a believable phone moment. She's mid-spray and the mist is visible, the framing is slightly
  off-centre, and the bathroom has ordinary detail (towels, a plant, a brass hook).
- The label copies the photo exactly: JB, J'ADUNÉ with the apostrophe, BEAUTY, PEPTIDE INFUSED,
  hydrating hair mist, 8 fl oz (237 mL). The label check passed.

Don't copy:

- The bottle is oversized for her hand. The old "fills at least 25% of the frame" rule did that,
  and true size now wins.
- She's in a tank top in a bathroom. Today's wardrobe and framing rule asks for everyday wear
  such as a linen shirt or knit top, to stay clear of the content filter.

## #26 — model_with_product (`26-model-with-product.png`)

What makes it good:

- The smile is natural, with real teeth (not uniform veneers), crinkled closed eyes and cheeks
  lifted by the smile.
- The skin has visible texture, with natural shine on the forehead and cheekbone rather than
  all-over gloss.
- Everyday styling: a cream knit cardigan, gold hoops and a pendant, nails done but not perfect.
- The bottle is held up beside her face, front-facing, and the hand doesn't cover the label.

Don't copy:

- The label check flagged it, and the flag is correct. The image prints "PEPTIDE-INFUSED" with a
  hyphen, but the real label says "PEPTIDE INFUSED".
- The bottle is still a little large next to her face (the same 25% rule).
