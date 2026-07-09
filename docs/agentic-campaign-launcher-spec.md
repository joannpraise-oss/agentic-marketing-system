# Feature Spec: Agentic Campaign Launcher
**Project:** Agentic Marketing Optimization System  
**Phase:** 5 (Months 9–12)  
**Status:** Specced — not yet in development  
**Last updated:** June 2026

---

## Problem Statement

Solo founders know their customer and their product, but launching a paid ad campaign requires context-switching across 4–6 tools, creative production skills, and dedicated time they don't have. The result: campaigns that never launch, or campaigns launched sloppily under pressure.

> *"I know the customer segment I want to target but I have to create the ad, log into Meta, set up the campaign and launch it. I've been unable to do it because of time."*
> — Joann, founder of Jadune (primary user persona)

---

## Vision

A solo founder should be able to go from campaign idea to live ad in under 15 minutes — using plain English, with no design skills, no platform expertise, and no agency required.

The system owns the strategy and orchestration layer. External creative APIs are interchangeable tools underneath it — not the product itself.

---

## User Story

**As a** solo founder,  
**I want to** describe a campaign in plain text and have the system generate, review, and launch it for me,  
**So that** I can run paid ads without context-switching, creative overhead, or platform expertise.

### Primary Flow

1. Founder types: *"I need a campaign for postpartum moms for Jadune"*
2. System asks: budget and creative format (Reels, Carousel, Static)
3. System generates: ad set + 2 creative variations via the Creative Orchestration Layer
4. Founder reviews: previews, edits copy/headline, regenerates if needed, sees platform preview
5. Founder approves: system launches directly via connected ad platform API
6. Done — total time under 15 minutes

---

## Target Users

**Primary:** Solo founders running DTC or service businesses who:
- Know their customer but lack time/skills to execute paid campaigns
- Have a modest ad budget ($500–$5,000/month)
- Are active on Meta and/or other social platforms

**Secondary (later):** Small marketing agencies managing campaigns for multiple clients

---

## Creative Orchestration Layer

This is the core architectural differentiator of the system. Rather than delegating all creative generation to a single platform (e.g. Meta's Advantage+ Creative), the system acts as an intelligent router — holding the strategy and brief, then selecting the best creative source for each format, brand context, and job.

### Why this matters

- **No single API is best at everything.** Firefly excels at brand-consistent static imagery. Meta Advantage+ is optimized for Meta placements. Video-native tools like Runway and Kling produce better motion. The router always uses the strongest tool for the job.
- **Vendor independence.** If any API changes pricing, deprecates a feature, or degrades in quality, the system routes around it without the founder noticing.
- **Own media as a creative lever.** Founders can upload their own footage, product photos, or UGC — the system remixes these into ad-ready formats. This preserves authenticity (real product, real founder, real customers) which consistently outperforms synthetic creative for DTC brands.
- **Data-driven routing over time.** Phase 4's analytics layer will tell the system which creative source actually converts best for which audience and format. Routing becomes intelligent, not just rule-based. This is the long-term moat.

### Creative Router Architecture

```
User brief
    ↓
Creative Strategy Agent (this system)
    ↓
Creative Router
    ├── Static image   → Adobe Firefly / DALL-E / Gemini Imagen / own media
    ├── Reels / video  → Runway / Kling / Sora / Meta Advantage+ / own media
    ├── Carousel       → Firefly multi-frame / own media remixed
    └── UGC-style      → Meta AI avatars / HeyGen / own media
    ↓
Brand overlay (logo, colors, safe zones applied universally)
    ↓
Human review interface
    ↓
Launch via platform API
```

### Creative Source Registry

Each creative source is treated as a pluggable adapter implementing a common interface: `brief_in → asset_out`. New tools can be added as they come online without changing the orchestration logic.

| Source | Type | Best for | Status |
|---|---|---|---|
| Own media (founder upload) | Upload | Authentic DTC, UGC remixing | v1 |
| Adobe Firefly | Static image | Brand-consistent imagery, safe licensing | v1 |
| DALL-E (OpenAI) | Static image | Fast generation, broad styles | v1 |
| Gemini Imagen | Static image | Google ecosystem, high resolution | v1 |
| Meta Advantage+ Creative | Video / static | Meta-optimized placements | v1 |
| Runway | Video | High quality short-form video | v2 |
| Kling | Video | Motion, product showcase | v2 |
| Sora (OpenAI) | Video | Cinematic, narrative video | v2 |
| HeyGen | UGC-style video | AI avatar presenter ads | v2 |
| [Future tools] | TBD | Added as market evolves | v3+ |

### Routing Logic (v1)

Initial routing is rule-based by format:

| Format requested | Primary source | Fallback |
|---|---|---|
| Static — own media available | Own media + Firefly overlay | Firefly solo |
| Static — no own media | Adobe Firefly | DALL-E |
| Reels — own media available | Own media remixed via Meta Advantage+ | Runway (v2) |
| Reels — no own media | Meta Advantage+ Creative | Kling (v2) |
| Carousel | Firefly multi-frame | Own media if available |
| UGC-style | Own media preferred | HeyGen avatar (v2) |

In Phase 4+, routing incorporates performance data: *"Firefly static + own media overlay has a 2.3x higher ROAS than pure Firefly for postpartum audience on Meta"* → system auto-selects that combination for similar briefs.

### AI Disclosure Compliance

As of March 2026, Meta requires disclosure on ads containing AI-generated or AI-modified content. The system automatically applies the correct disclosure flag at launch based on which creative sources were used in the final asset. This is non-negotiable and handled at the infrastructure level, not left to the founder.

---

## Scope

### In scope for v1
- Natural language campaign brief → structured campaign parameters
- Brand asset ingestion (upload or extract from URL)
- Creative Orchestration Layer with static image sources (Firefly, DALL-E, own media)
- Meta Advantage+ Creative for Reels (delegated, not built)
- Brand overlay applied universally (logo, colors, safe zones)
- Human-in-the-loop review: preview, edit copy, regenerate, platform preview
- Launch via Meta Ads API (Facebook + Instagram)
- AI disclosure flag applied automatically
- Campaign saved to database with full audit trail

### In scope for v2
- Video generation sources (Runway, Kling, Sora)
- UGC-style sources (HeyGen)
- TikTok launch integration
- Data-driven routing from Phase 4 analytics

### Explicitly out of scope for v1
- Google, Pinterest, LinkedIn, Snapchat integrations
- Automated post-launch optimization
- Agency multi-client management

### Channels (full roadmap)
| Priority | Channel | Notes |
|---|---|---|
| v1 | Meta (Facebook + Instagram) | Single API, highest DTC ROI |
| v2 | TikTok | Separate API, different creative norms |
| v2 | Google | Different format (search/display) |
| v3 | Pinterest, LinkedIn, Snapchat | Niche use cases |

---

## Feature Requirements

### 1. Campaign Brief Input
- Free-text input: *"I need a campaign for postpartum moms for Jadune"*
- System extracts: product/brand, audience segment, implied goal
- System asks 2 follow-up questions only: budget + format
- Formats: Reels (vertical), Carousel (3–5 cards), Static (single image)

### 2. Brand Asset Management (Onboarding)
- **Upload path:** Logo (PNG with transparency preferred), brand colors (hex), font (optional), own media (photos/video)
- **URL extraction path:** Paste website URL → system scrapes logo, dominant colors, font hints
- Assets stored per user account, reused across all campaigns
- Own media library: founders can build a growing asset library over time
- Editable at any time from account settings

### 3. Creative Generation
All four quality dimensions apply — speed, quality, brand consistency, and user control:
- **Copy:** GPT-4o generates headline + primary text + CTA for each variation
- **Images/video:** Routed via Creative Orchestration Layer based on format and available assets
- **Brand overlay:** Logo composited, colors applied, safe zones respected — applied after generation regardless of source
- **2 variations per submission** — may use different creative sources if router determines that's optimal
- Generation target: under 60 seconds for static; under 3 minutes for video (v2)

### 4. Review Interface
Four-layer review before launch:
1. **Static preview** — rendered creative with copy overlaid
2. **Editable copy** — headline, primary text, CTA editable inline
3. **Regenerate** — regenerate either variation independently (can specify different source if desired)
4. **Platform preview** — simulated view of how ad renders in Meta feed/Stories/Reels

Founder iterates until satisfied. System does not auto-launch.

### 5. Launch Integration
- **Meta Ads API** — requires Meta Business account connected via OAuth
- System creates: Campaign → Ad Set (targeting from brief) → Ad Creative → Ad
- AI disclosure flag applied automatically based on creative sources used
- Budget set during review step
- Post-launch: campaign ID saved, linked to brand profile and creative source audit trail

---

## Technical Architecture

### New components required

| Component | Description |
|---|---|
| `CampaignBriefParser` | LLM chain: natural language → structured campaign params |
| `BrandAssetService` | Upload storage + URL scraping + own media library |
| `CreativeOrchestrator` | Router: selects source per format, manages API calls, applies brand overlay |
| `CreativeSourceAdapters` | Pluggable adapters per API (Firefly, DALL-E, Runway, etc.) — common interface |
| `AdCopyGenerator` | GPT-4o structured output: headline, body, CTA × 2 variations |
| `BrandOverlayPipeline` | Pillow-based: logo composite, color application, safe zone enforcement |
| `ReviewWorkflowUI` | Frontend: preview, edit, regenerate, platform mock |
| `MetaAdsConnector` | OAuth + Campaign/AdSet/Creative/Ad API calls + disclosure flag |
| `CampaignAuditLog` | DB: every action, creative source used, approval, and launch timestamped |

### APIs and services

| Service | Purpose |
|---|---|
| OpenAI GPT-4o | Brief parsing, copy generation |
| Adobe Firefly API | Brand-safe static image generation |
| DALL-E (OpenAI) | Static image fallback |
| Gemini Imagen | Static image (Google ecosystem) |
| Meta Marketing API | Campaign creation, launch, Advantage+ Creative |
| Runway / Kling / Sora | Video generation (v2) |
| HeyGen | UGC-style avatar video (v2) |
| Unsplash / Pexels | Stock photo sourcing (own media fallback) |
| Pillow (Python) | Brand overlay composition |
| Supabase Storage | Brand assets + own media library |

### Meta API prerequisites
Meta App Review is the longest lead-time item — apply during Phase 3 or 4, not when Phase 5 begins:
1. Create Meta Developer App
2. Request `ads_management` and `ads_read` permissions
3. Submit for Meta App Review (1–4 weeks)
4. Implement OAuth for user connection flow

---

## Dependencies on Earlier Phases

| Phase | What it provides |
|---|---|
| Phase 1 (current) | Campaign data schema, analysis pipeline — defines what "good performance" looks like |
| Phase 3 | Auth system — required for per-user OAuth token storage (Meta, Adobe, etc.) |
| Phase 4 | Analytics layer — feeds performance data back into Creative Router for intelligent source selection |

**Do not start Phase 5 before Phase 3 (auth) is complete.**

---

## Open Questions

1. **Own media video remixing:** For Reels using founder-uploaded footage, does the system do basic trimming/assembly itself (via FFmpeg), or does it pass raw footage to Meta Advantage+ / Runway for assembly?
2. **Creative source selection transparency:** Does the founder see which API generated their creative, or is this abstracted away? (Transparency may build trust; abstraction keeps UX simpler.)
3. **Licensing:** Adobe Firefly is commercially safe by design. DALL-E and Gemini have their own terms. The system needs a per-source licensing flag so founders are always protected.
4. **Billing model:** Creative API costs vary significantly per call. Is this a usage-based add-on, a premium tier, or bundled into base pricing?
5. **Agency path:** When agencies are added, does each client get their own brand profile, own media library, and platform OAuth connections?

---

## Success Metrics

- Time from brief to launched ad: **≤ 15 minutes**
- Creative approval rate (approves without regenerating): **≥ 60%**
- Campaigns launched vs. campaigns started: **≥ 80% completion rate**
- Creative source routing accuracy (Phase 4+): system selects highest-converting source **≥ 70% of the time** vs. random selection baseline
- Founder sentiment: *"I would not have run this campaign without this tool"*

---

*This spec was written in June 2026 and reflects the product vision as of Phase 1 completion. The Creative Orchestration Layer is designed to be extensible — new creative APIs are added as adapters without changing the core orchestration logic. Details will be refined during Phase 4 as the analytics layer matures and Meta API access is established.*
