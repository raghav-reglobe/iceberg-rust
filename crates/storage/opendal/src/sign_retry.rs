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

//! Reclassifies request-SIGNING failures as temporary so `RetryLayer`
//! retries them.
//!
//! Why this exists: reqsign's `Signer` caches the loaded credential and
//! re-loads it lazily at sign time once it expires. The credential
//! provider chain swallows per-provider errors (a transient STS /
//! network failure inside e.g. the web-identity provider logs a warning
//! and falls through to `Ok(None)`), which the signer surfaces as
//! `failed to load signing credential`. OpenDAL wraps that as
//! `ErrorKind::Unexpected` with operation `reqsign::Sign` and does NOT
//! mark it temporary — so `RetryLayer` (which retries only
//! `is_temporary()` errors) lets a single credential-refresh blip kill
//! a multi-hour operation the moment the cached credential expires.
//!
//! Signing happens strictly BEFORE the request is sent, so retrying a
//! sign-phase failure can never duplicate a side effect: each retry
//! re-enters `Signer::sign`, which re-attempts the credential load.
//! This layer therefore marks any error whose operation is
//! `reqsign::Sign` as temporary; the retry budget and backoff stay
//! owned by the `RetryLayer` stacked above it.

use opendal::raw::{
    Access, Layer, LayeredAccess, OpCopier, OpCopy, OpCreateDir, OpDelete, OpList, OpPresign,
    OpRead, OpRename, OpStat, OpWrite, RpCopy, RpCreateDir, RpDelete, RpList, RpPresign, RpRead,
    RpRename, RpStat, RpWrite, oio,
};
use opendal::{Buffer, Metadata, Result};

/// Marks request-signing failures (`reqsign::Sign` operation errors) as
/// temporary. Must sit BENEATH `RetryLayer` (applied to the operator
/// first) so the retry layer observes the reclassified error.
#[derive(Clone, Debug, Default)]
pub struct SignErrorRetryLayer;

/// Reclassify a sign-phase failure as temporary; leave every other
/// error untouched. The operation tag is only reachable through the
/// Display form — opendal's `Error` exposes no `operation()` getter.
fn reclassify(err: opendal::Error) -> opendal::Error {
    if err.kind() == opendal::ErrorKind::Unexpected
        && !err.is_temporary()
        && err.to_string().contains("reqsign::Sign")
    {
        err.set_temporary()
    } else {
        err
    }
}

impl<A: Access> Layer<A> for SignErrorRetryLayer {
    type LayeredAccess = SignErrorRetryAccessor<A>;

    fn layer(&self, inner: A) -> Self::LayeredAccess {
        SignErrorRetryAccessor { inner }
    }
}

#[doc(hidden)]
#[derive(Debug)]
pub struct SignErrorRetryAccessor<A: Access> {
    inner: A,
}

impl<A: Access> LayeredAccess for SignErrorRetryAccessor<A> {
    type Inner = A;
    type Reader = SignErrorRetryWrapper<A::Reader>;
    type Writer = SignErrorRetryWrapper<A::Writer>;
    type Lister = SignErrorRetryWrapper<A::Lister>;
    type Deleter = SignErrorRetryWrapper<A::Deleter>;
    type Copier = SignErrorRetryWrapper<A::Copier>;

    fn inner(&self) -> &Self::Inner {
        &self.inner
    }

    async fn create_dir(&self, path: &str, args: OpCreateDir) -> Result<RpCreateDir> {
        self.inner.create_dir(path, args).await.map_err(reclassify)
    }

    async fn read(&self, path: &str, args: OpRead) -> Result<(RpRead, Self::Reader)> {
        self.inner
            .read(path, args)
            .await
            .map(|(rp, r)| (rp, SignErrorRetryWrapper::new(r)))
            .map_err(reclassify)
    }

    async fn write(&self, path: &str, args: OpWrite) -> Result<(RpWrite, Self::Writer)> {
        self.inner
            .write(path, args)
            .await
            .map(|(rp, w)| (rp, SignErrorRetryWrapper::new(w)))
            .map_err(reclassify)
    }

    async fn copy(
        &self,
        from: &str,
        to: &str,
        args: OpCopy,
        opts: OpCopier,
    ) -> Result<(RpCopy, Self::Copier)> {
        self.inner
            .copy(from, to, args, opts)
            .await
            .map(|(rp, c)| (rp, SignErrorRetryWrapper::new(c)))
            .map_err(reclassify)
    }

    async fn rename(&self, from: &str, to: &str, args: OpRename) -> Result<RpRename> {
        self.inner.rename(from, to, args).await.map_err(reclassify)
    }

    async fn stat(&self, path: &str, args: OpStat) -> Result<RpStat> {
        self.inner.stat(path, args).await.map_err(reclassify)
    }

    async fn delete(&self) -> Result<(RpDelete, Self::Deleter)> {
        self.inner
            .delete()
            .await
            .map(|(rp, d)| (rp, SignErrorRetryWrapper::new(d)))
            .map_err(reclassify)
    }

    async fn list(&self, path: &str, args: OpList) -> Result<(RpList, Self::Lister)> {
        self.inner
            .list(path, args)
            .await
            .map(|(rp, l)| (rp, SignErrorRetryWrapper::new(l)))
            .map_err(reclassify)
    }

    async fn presign(&self, path: &str, args: OpPresign) -> Result<RpPresign> {
        self.inner.presign(path, args).await.map_err(reclassify)
    }
}

/// Streaming ops sign fresh HTTP requests too (ranged GET per read
/// chunk, one PUT per multipart part, continuation-page LIST, batch
/// DELETE, complete-multipart on close) — so the reclassification must
/// also cover the Reader/Writer/Lister/Deleter/Copier surfaces.
#[doc(hidden)]
pub struct SignErrorRetryWrapper<T> {
    inner: T,
}

impl<T> SignErrorRetryWrapper<T> {
    fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<R: oio::Read> oio::Read for SignErrorRetryWrapper<R> {
    async fn read(&mut self) -> Result<Buffer> {
        self.inner.read().await.map_err(reclassify)
    }
}

impl<W: oio::Write> oio::Write for SignErrorRetryWrapper<W> {
    async fn write(&mut self, bs: Buffer) -> Result<()> {
        self.inner.write(bs).await.map_err(reclassify)
    }

    async fn close(&mut self) -> Result<Metadata> {
        self.inner.close().await.map_err(reclassify)
    }

    async fn abort(&mut self) -> Result<()> {
        self.inner.abort().await.map_err(reclassify)
    }
}

impl<L: oio::List> oio::List for SignErrorRetryWrapper<L> {
    async fn next(&mut self) -> Result<Option<oio::Entry>> {
        self.inner.next().await.map_err(reclassify)
    }
}

impl<D: oio::Delete> oio::Delete for SignErrorRetryWrapper<D> {
    async fn delete(&mut self, path: &str, args: OpDelete) -> Result<()> {
        self.inner.delete(path, args).await.map_err(reclassify)
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await.map_err(reclassify)
    }
}

impl<C: oio::Copy> oio::Copy for SignErrorRetryWrapper<C> {
    async fn next(&mut self) -> Result<Option<usize>> {
        self.inner.next().await.map_err(reclassify)
    }

    async fn close(&mut self) -> Result<Metadata> {
        self.inner.close().await.map_err(reclassify)
    }

    async fn abort(&mut self) -> Result<()> {
        self.inner.abort().await.map_err(reclassify)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use opendal::layers::RetryLayer;
    use opendal::raw::AccessorInfo;
    use opendal::{Capability, EntryMode, ErrorKind, Operator};

    use super::*;

    /// The exact error shape opendal-core's `new_request_sign_error`
    /// produces for a reqsign credential-load failure.
    fn sign_error() -> opendal::Error {
        opendal::Error::new(ErrorKind::Unexpected, "signing http request")
            .with_operation("reqsign::Sign")
            .set_source(anyhow::anyhow!("failed to load signing credential"))
    }

    /// Fails `stat` with the sign error `fail_times` times, then succeeds.
    #[derive(Debug, Clone)]
    struct FlakySignService {
        fail_times: usize,
        calls: Arc<AtomicUsize>,
    }

    impl Access for FlakySignService {
        type Reader = oio::Reader;
        type Writer = oio::Writer;
        type Lister = oio::Lister;
        type Deleter = oio::Deleter;
        type Copier = oio::Copier;

        fn info(&self) -> Arc<AccessorInfo> {
            let am = AccessorInfo::default();
            am.set_native_capability(Capability {
                stat: true,
                ..Default::default()
            });
            am.into()
        }

        async fn stat(&self, _: &str, _: OpStat) -> Result<RpStat> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_times {
                Err(sign_error())
            } else {
                Ok(RpStat::new(Metadata::new(EntryMode::FILE)))
            }
        }
    }

    #[test]
    fn test_reclassify_marks_sign_error_temporary() {
        let err = reclassify(sign_error());
        assert!(err.is_temporary(), "sign error must become temporary");
        assert_eq!(err.kind(), ErrorKind::Unexpected);
    }

    #[test]
    fn test_reclassify_leaves_other_errors_untouched() {
        let err = reclassify(
            opendal::Error::new(ErrorKind::Unexpected, "send http request")
                .with_operation("http_util::Client::send"),
        );
        assert!(!err.is_temporary(), "non-sign errors must stay permanent");

        let err = reclassify(opendal::Error::new(ErrorKind::NotFound, "not found"));
        assert!(!err.is_temporary());
        assert_eq!(err.kind(), ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn test_retry_layer_retries_reclassified_sign_error() {
        let calls = Arc::new(AtomicUsize::new(0));
        let svc = FlakySignService {
            fail_times: 2,
            calls: Arc::clone(&calls),
        };

        // Same stacking order as production: sign-retry beneath RetryLayer.
        let op = Operator::from_inner(Arc::new(svc))
            .layer(SignErrorRetryLayer)
            .layer(RetryLayer::new().with_min_delay(std::time::Duration::from_millis(1)));

        let meta = op
            .stat("somefile")
            .await
            .expect("stat must succeed once RetryLayer retries the reclassified sign error");
        assert_eq!(meta.mode(), EntryMode::FILE);
        assert_eq!(calls.load(Ordering::SeqCst), 3, "2 failures + 1 success");
    }

    #[tokio::test]
    async fn test_without_layer_sign_error_is_fatal() {
        let calls = Arc::new(AtomicUsize::new(0));
        let svc = FlakySignService {
            fail_times: 1,
            calls: Arc::clone(&calls),
        };

        // RetryLayer alone (no reclassification) must NOT retry — this
        // pins the underlying opendal behavior the layer exists to fix;
        // if this test ever fails, upstream made sign errors retryable
        // and SignErrorRetryLayer can be dropped.
        let op = Operator::from_inner(Arc::new(svc))
            .layer(RetryLayer::new().with_min_delay(std::time::Duration::from_millis(1)));

        let err = op.stat("somefile").await.expect_err("must fail");
        assert!(err.to_string().contains("reqsign::Sign"));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "no retry without the layer"
        );
    }
}
