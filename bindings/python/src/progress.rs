// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Renders the compaction engine's progress events as text lines.
//!
//! The engine reports progress as `tracing` events and prints nothing
//! itself. A Python caller has no `tracing` subscriber of its own, so this
//! module installs one that writes each progress event to stderr as a single
//! `compact-phase key=value ...` line, fields in the order the engine
//! recorded them.
//!
//! The layer is interested in the progress target ONLY: every other
//! `tracing` callsite in the process stays disabled, exactly as it is with no
//! subscriber at all.

use std::fmt::Write as _;
use std::sync::OnceLock;

use iceberg_compaction::engine::PROGRESS_TARGET;
use tracing::field::{Field, Visit};
use tracing::subscriber::Interest;
use tracing::{Event, Metadata, Subscriber};
use tracing_subscriber::Registry;
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

/// The line every progress event starts with.
const LINE_PREFIX: &str = "compact-phase";

/// A layer that hands each progress event, rendered as one line, to `sink`.
pub(crate) struct ProgressLines<W> {
    sink: W,
}

impl<W> ProgressLines<W> {
    pub(crate) fn new(sink: W) -> Self {
        Self { sink }
    }
}

struct LineVisitor(String);

impl Visit for LineVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        let _ = write!(self.0, " {}={}", field.name(), value);
    }

    // Numbers, booleans and `%display` values all arrive here; their `Debug`
    // form is the plain value (no quotes).
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        let _ = write!(self.0, " {}={:?}", field.name(), value);
    }
}

impl<S, W> Layer<S> for ProgressLines<W>
where
    S: Subscriber,
    W: Fn(&str) + Send + Sync + 'static,
{
    fn register_callsite(&self, metadata: &'static Metadata<'static>) -> Interest {
        if metadata.target() == PROGRESS_TARGET {
            Interest::always()
        } else {
            Interest::never()
        }
    }

    fn enabled(&self, metadata: &Metadata<'_>, _ctx: Context<'_, S>) -> bool {
        metadata.target() == PROGRESS_TARGET
    }

    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        if event.metadata().target() != PROGRESS_TARGET {
            return;
        }
        let mut line = LineVisitor(String::from(LINE_PREFIX));
        event.record(&mut line);
        (self.sink)(&line.0);
    }
}

/// Install the stderr renderer once per process. If the embedding process
/// already installed a global subscriber, that one keeps the events and this
/// call changes nothing.
pub(crate) fn install() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let subscriber =
            Registry::default().with(ProgressLines::new(|line: &str| eprintln!("{line}")));
        let _ = tracing::subscriber::set_global_default(subscriber);
    });
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    fn capture() -> (Arc<Mutex<Vec<String>>>, impl Subscriber + Send + Sync) {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&lines);
        let subscriber = Registry::default().with(ProgressLines::new(move |line: &str| {
            sink.lock().unwrap().push(line.to_string())
        }));
        (lines, subscriber)
    }

    #[test]
    fn a_progress_event_is_one_key_value_line_in_field_order() {
        let (lines, subscriber) = capture();
        tracing::subscriber::with_default(subscriber, || {
            let (gi, n) = (3, 839);
            tracing::info!(
                target: PROGRESS_TARGET,
                table = %"db.t",
                phase = "group-done",
                group = %format_args!("{gi}/{n}"),
                files = 72usize,
                wall_s = 47u64,
                in_flight = 1usize,
                elapsed_s = 140u64
            );
        });
        assert_eq!(*lines.lock().unwrap(), vec![
            "compact-phase table=db.t phase=group-done group=3/839 files=72 wall_s=47 \
             in_flight=1 elapsed_s=140"
                .to_string()
        ]);
    }

    #[test]
    fn every_other_target_stays_disabled() {
        let (lines, subscriber) = capture();
        tracing::subscriber::with_default(subscriber, || {
            tracing::info!(target: "some::other::crate", answer = 42);
            tracing::error!("an untargeted event");
            assert!(!tracing::enabled!(target: "some::other::crate", tracing::Level::ERROR));
            assert!(tracing::enabled!(target: PROGRESS_TARGET, tracing::Level::INFO));
        });
        assert!(lines.lock().unwrap().is_empty());
    }
}
