<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# RFC: a shared tier for the whole-file data cache (cold-start elimination)

Status: **exploration — no implementation**. Companion to the per-process
object-bytes cache (`iceberg::cache::ObjectBytesCache`, foyer disk tier) and
its statistics surface.

## Problem

The per-process whole-file data cache eliminates ranged-GET amplification
(measured on one small hot table: 536 data GETs per merge collapsed to one
GET per distinct file) and makes repeat reads free. But the cache is
per-process on an ephemeral volume: **every new worker pod starts cold**,
re-fetching the same immutable files its predecessors on the same node just
fetched. Observed: a pod 13 seconds old had already pulled 3.5 GB. With
short-lived pods (minutes–hours) on long-lived nodes (1–3 days) hosting 4–6
worker pods, **same-node duplicate fetches are the dominant waste**, plus
N× duplicate disk usage (16 GB × pods per node).

### Constraints (established)

- foyer's disk engine is **single-process** (no cross-process locking;
  fixed-name partition files, in-process allocation state). A naively
  shared directory is corruption, not sharing.
- No central in-memory store: RAM economics are wrong for ~8–256 MB
  immutable values served by range, and a central proxy is a SPOF and a
  bandwidth funnel (~700 Mbps class traffic).
- The REST catalog is the existing scaling wall — any design must add
  zero catalog-path load.
- Nodes span AZs; cross-AZ transfer is ~$0.01/GB **each direction**.
  Object storage (standard and Express One Zone) → compute in-region is
  free of transfer charges.
- Requests: S3 standard GET $0.0004/1k. Pre-cache the estate did ~75M
  data GETs/day (~$900/mo); post-cache, steady-state GETs are already
  down to first-touch + new files.

## Option 4 first: what does do-nothing actually cost? (the baseline to beat)

With whole-file caching live, a cold start costs **one GET per file in the
working set**, not 536 ranged GETs per merge:

- **Dollars**: ~100–500 GETs per pod start ≈ $0.00004–0.0002. Even 10k pod
  starts/day ≈ 2M GETs/day ≈ **$25/mo**. In-region S3→EC2 bytes are free.
  Cold starts are nearly free in request dollars *because of* the
  whole-file design.
- **Latency**: the real cost. First-touch of a file is an S3 whole-object
  fetch (~100–800 ms) vs ~1–10 ms from local disk, paid on the critical
  path of a pod's first slices. A 3.5–16 GB warm-up spread over early
  slices ≈ tens of seconds of added wall-clock per pod lifetime. For
  long-lived heavy pods this is single-digit percent; for short-lived
  light-lane pods (few slices) it can dominate the pod's useful work.
- **Disk duplication**: 16 GB × 4–6 pods ≈ 64–96 GB of identical bytes
  per 160 GB node root.

**Conclusion**: any shared tier is justified by *latency and duplication*,
not by S3 request spend. The per-slice `cache_stats` now shipping makes
this measurable precisely (see "Measure first").

## Option 1 — node-local cache daemon (centralized-per-node)

A small Rust daemon (DaemonSet) that **exclusively owns** a hostPath
directory with a foyer hybrid cache inside, serving same-node pods over a
Unix domain socket. Pods' `ObjectBytesCache` gains a remote-first client
implementation (the trait is already the seam: `get`/`set`/`stats` over
UDS); pods keep their small in-memory tier, drop their private disk tier.

- **Solves**: same-node cold starts (the dominant waste) + disk
  duplication (one 32–64 GB tier per node instead of 6 × 16 GB). Node
  lifetime 1–3 days means warmth survives pod churn almost entirely.
- **Explicitly avoids**: cross-AZ traffic (localhost only), SPOF (daemon
  down ⇒ client falls back to direct object storage; merges never fail on
  cache), catalog load (data plane only), foyer multi-process corruption
  (single owner per directory).
- **Latency delta vs in-process disk**: UDS round-trip ~50–150 µs +
  memcpy-bound streaming (multi-GB/s). A 32 MB file ≈ 10–20 ms via UDS vs
  ~8–12 ms in-process — effectively parity, and both are 20–50× better
  than an S3 first-touch. Singleflight in the daemon also collapses the
  N-pods-miss-the-same-file race that per-pod caches cannot.
- **Failure/lifecycle**: hostPath + foyer's recovery keep the cache warm
  across daemon restarts and upgrades; node loss loses only cache (re-warm).
- **Effort**: daemon crate (UDS server, foyer, singleflight, stats
  endpoint) ~3–5 days; client `ObjectBytesCache` impl ~1–2 days;
  DaemonSet/manifests/socket mount ~1 day; fallback paths, stats wiring,
  soak ~2–3 days. **≈ 2 calendar weeks.** No new managed dependency; ~$0
  marginal AWS cost.

## Option 2 — S3 Express One Zone directory bucket as a shared L2

Read-through second tier in the object store itself: pod miss → Express
bucket → miss → standard S3 → write back to Express + local. Prior art:
mountpoint-s3's shared-cache mode does exactly this (keyed by ETag). We
would wire it **directly at the `ObjectBytesCache` seam** (an S3-backed L2
implementation, ~300–500 lines), not via FUSE.

- **Availability**: Express One Zone **is available in ap-south-1
  (Mumbai)** (verify the target AZ overlaps the cluster's).
- **Pricing (post the April 2025 cuts; us-east-1 figures, Mumbai typically
  ~10–20 % higher — verify)**: GETs $0.00003/1k (−85 %; now ~13× cheaper
  than standard GETs), per-GB retrieval ~$0.0006/GB and upload
  ~$0.0032/GB (−60 %, now applied to all bytes), storage ~$0.11/GB-mo.
- **Worked estimate**: hot working set 2–5 TB ⇒ storage $250–600/mo;
  cache-fill uploads (new files + churn, ~100–300 GB/day) ≈ $10–30/mo;
  cold-start retrievals (say 1k pod starts/day × 2–4 GB actually pulled)
  ≈ $40–90/mo. **All-in ≈ $300–700/mo.**
- **One-zone risk**: acceptable by construction — it is a cache; AZ loss
  = re-warm from standard S3.
- **Cross-AZ trap check**: object-storage access in-region carries **no
  transfer charge** regardless of AZ (One Zone is a durability/latency
  placement, not a transfer-billing boundary) — pods in other AZs pay
  ~1 ms extra latency, not dollars. Verify this on the current pricing
  page before committing; if it ever billed as inter-AZ transfer the
  option dies (see the trap quantified below).
- **Gaps**: no LRU — size control is lifecycle expiry (day granularity) +
  admission discipline (e.g. write-on-second-miss); latency is single-digit
  ms (better than standard S3's 30–200 ms, ~5–10× worse than node-local
  NVMe); does nothing for duplicate *disk* usage per node unless pods also
  shrink their local tiers.
- **Effort**: ~3–5 days + bucket/lifecycle IaC. Zero infra to operate.
- **Unique strength vs option 1**: warmth survives **node** churn too —
  brand-new nodes (spot replacement) start warm.

## Option 3 — existing projects survey

| Project | Verdict |
|---|---|
| mountpoint-s3 (local + shared-cache modes) | Pattern donor for option 2 (Express shared cache, ETag-keyed). FUSE itself is a poor fit for churning spot pods, and our in-process whole-file cache already outperforms its local mode at our seam. |
| Dragonfly / nydus (P2P blob distribution) | Built for container images; scheduler + seed-peer infrastructure is heavy, cross-AZ-unaware peers reintroduce the transfer trap. Overkill. |
| JuiceFS | POSIX FS with a metadata service (Redis/TiKV) — reintroduces exactly the central stateful dependency this platform rejected. |
| Alluxio | JVM cluster cache; heavy, contrary to the JVM-free direction. |
| opendal cache layer | Removed upstream; nothing to reuse. |
| foyer distributed mode | Does not exist; foyer is explicitly a node-local hybrid cache. |

Nothing in the ecosystem beats a ~1k-line UDS daemon reusing foyer and the
existing trait seam.

### The cross-AZ trap, quantified (why a central cluster service stays rejected)

A single in-cluster cache service serving ~2–3 TB/day of warm reads with
~⅔ of pods in other AZs ⇒ 1.3–2 TB/day × $0.02/GB ≈ **$800–1,200/mo of
inter-AZ transfer** — comparable to or worse than the original pre-cache
S3 GET bill, plus a SPOF and a ~700 Mbps funnel. Any shared design must be
either node-local (option 1) or object-storage-resident (option 2).

## Comparison

| | Same-node cold start | Fresh-node cold start | Disk dup | Cross-AZ $ | SPOF | Marginal $ | Effort |
|---|---|---|---|---|---|---|---|
| Do nothing | cold | cold | 6× | — | — | ~$25/mo GETs | 0 |
| 1. Node daemon | **warm** | cold | **1×** | none | none (fallback) | ~$0 | ~2 wks |
| 2. Express L2 | warm | **warm** | 6× (unless local tiers shrink) | none¹ | none (fallback) | ~$300–700/mo | ~1 wk |
| Central service | warm | warm | 1× | **$800–1,200/mo** | yes | high | ~3 wks |

¹ verify current pricing-page wording before committing.

## Recommendation

1. **Measure first (days, zero risk)** — the stats surface shipped for
   exactly this. Segment per-slice `cache_stats` diffs by pod age:
   (a) fraction of misses/inserts occurring in a pod's first N minutes vs
   steady state; (b) duplicate-fetch share: Σ inserts across same-node
   pods vs distinct files touched per node per day; (c) added wall-clock
   attributable to first-touch on short-lived pods. Also check whether
   simply **lengthening light-lane pod lifetimes** (more slices per pod)
   erases most of the waste with zero cache work — it attacks the same
   number.
2. **Primary: option 1 (node-local daemon)** if the measurement confirms
   same-node duplication dominates (expected: node lifetime ≫ pod
   lifetime). It is the only option that fits every constraint
   structurally — no SPOF, no cross-AZ exposure, no new managed spend, no
   catalog load — and reuses foyer plus the existing trait seam. ~2 weeks.
3. **Alternative/complement: option 2 (Express One Zone L2)** if the
   measurement shows a material fresh-node share (spot churn), or if two
   weeks of engineering is worth trading for ~$300–700/mo and a week of
   wiring. The two compose cleanly later (daemon L2 → Express L3) behind
   the same trait.
