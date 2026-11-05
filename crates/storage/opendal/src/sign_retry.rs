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

use std::fmt::Formatter;
use std::sync::Arc;

use opendal::raw::*;
use opendal::{Buffer, Capability, Metadata, Result};

/// Marks request-signing failures (`reqsign::Sign` operation errors) as
/// temporary. Must sit BENEATH `RetryLayer` (applied to the operator
/// first) so the retry layer observes the reclassified error.
#[derive(Clone, Debug, Default)]
pub struct SignErrorRetryLayer;

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

impl Layer for SignErrorRetryLayer {
    fn apply_service(&self, inner: Servicer) -> Servicer {
        Arc::new(SignErrorRetryService { inner })
    }
}

/// The service wrapper: every operation and every I/O body forwards to the
/// inner service and reclassifies the error it returns.
#[doc(hidden)]
pub struct SignErrorRetryService {
    inner: Servicer,
}

impl std::fmt::Debug for SignErrorRetryService {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignErrorRetryService")
            .field("inner", &self.inner)
            .finish()
    }
}

impl Service for SignErrorRetryService {
    type Reader = SignErrorRetryWrapper<oio::Reader>;
    type Writer = SignErrorRetryWrapper<oio::Writer>;
    type Lister = SignErrorRetryWrapper<oio::Lister>;
    type Deleter = SignErrorRetryWrapper<oio::Deleter>;
    type Copier = SignErrorRetryWrapper<oio::Copier>;

    fn info(&self) -> ServiceInfo {
        self.inner.info()
    }

    fn capability(&self) -> Capability {
        self.inner.capability()
    }

    async fn create_dir(
        &self,
        ctx: &OperationContext,
        path: &str,
        args: OpCreateDir,
    ) -> Result<RpCreateDir> {
        self.inner
            .create_dir(ctx, path, args)
            .await
            .map_err(reclassify)
    }

    fn read(&self, ctx: &OperationContext, path: &str, args: OpRead) -> Result<Self::Reader> {
        self.inner
            .read(ctx, path, args)
            .map(SignErrorRetryWrapper::new)
            .map_err(reclassify)
    }

    fn write(&self, ctx: &OperationContext, path: &str, args: OpWrite) -> Result<Self::Writer> {
        self.inner
            .write(ctx, path, args)
            .map(SignErrorRetryWrapper::new)
            .map_err(reclassify)
    }

    fn copy(
        &self,
        ctx: &OperationContext,
        from: &str,
        to: &str,
        args: OpCopy,
        opts: OpCopier,
    ) -> Result<Self::Copier> {
        self.inner
            .copy(ctx, from, to, args, opts)
            .map(SignErrorRetryWrapper::new)
            .map_err(reclassify)
    }

    async fn rename(
        &self,
        ctx: &OperationContext,
        from: &str,
        to: &str,
        args: OpRename,
    ) -> Result<RpRename> {
        self.inner
            .rename(ctx, from, to, args)
            .await
            .map_err(reclassify)
    }

    async fn stat(&self, ctx: &OperationContext, path: &str, args: OpStat) -> Result<RpStat> {
        self.inner.stat(ctx, path, args).await.map_err(reclassify)
    }

    fn delete(&self, ctx: &OperationContext) -> Result<Self::Deleter> {
        self.inner
            .delete(ctx)
            .map(SignErrorRetryWrapper::new)
            .map_err(reclassify)
    }

    fn list(&self, ctx: &OperationContext, path: &str, args: OpList) -> Result<Self::Lister> {
        self.inner
            .list(ctx, path, args)
            .map(SignErrorRetryWrapper::new)
            .map_err(reclassify)
    }

    async fn presign(
        &self,
        ctx: &OperationContext,
        path: &str,
        args: OpPresign,
    ) -> Result<RpPresign> {
        self.inner
            .presign(ctx, path, args)
            .await
            .map_err(reclassify)
    }
}

/// Wraps an I/O body so every error it yields is reclassified too: a
/// credential refresh can fail on any ranged read or multipart part, not
/// only on the operation that opened the body.
#[doc(hidden)]
pub struct SignErrorRetryWrapper<T> {
    inner: T,
}

impl<T> SignErrorRetryWrapper<T> {
    fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T: oio::Read> oio::Read for SignErrorRetryWrapper<T> {
    async fn open(&self, range: BytesRange) -> Result<(RpRead, Box<dyn oio::ReadStreamDyn>)> {
        self.inner
            .open(range)
            .await
            .map(|(rp, stream)| {
                (
                    rp,
                    Box::new(SignErrorRetryWrapper::new(stream)) as Box<dyn oio::ReadStreamDyn>,
                )
            })
            .map_err(reclassify)
    }

    async fn read(&self, range: BytesRange) -> Result<(RpRead, Buffer)> {
        self.inner.read(range).await.map_err(reclassify)
    }
}

impl<T: oio::ReadStream> oio::ReadStream for SignErrorRetryWrapper<T> {
    async fn read(&mut self) -> Result<Buffer> {
        self.inner.read().await.map_err(reclassify)
    }
}

impl<T: oio::Write> oio::Write for SignErrorRetryWrapper<T> {
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

impl<T: oio::List> oio::List for SignErrorRetryWrapper<T> {
    async fn next(&mut self) -> Result<Option<oio::Entry>> {
        self.inner.next().await.map_err(reclassify)
    }
}

impl<T: oio::Delete> oio::Delete for SignErrorRetryWrapper<T> {
    async fn delete(&mut self, path: &str, args: OpDelete) -> Result<()> {
        self.inner.delete(path, args).await.map_err(reclassify)
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await.map_err(reclassify)
    }
}

impl<T: oio::Copy> oio::Copy for SignErrorRetryWrapper<T> {
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
    use opendal::ErrorKind;

    use super::*;

    /// The exact error shape the signer path produces for a credential-load
    /// failure.
    fn sign_error() -> opendal::Error {
        opendal::Error::new(ErrorKind::Unexpected, "signing http request")
            .with_operation("reqsign::Sign")
            .set_source(anyhow::anyhow!("failed to load signing credential"))
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
}
