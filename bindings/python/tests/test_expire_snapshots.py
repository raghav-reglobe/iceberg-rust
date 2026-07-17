# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""End-to-end `compaction.expire_snapshots` against a real REST catalog.

Spins up the Iceberg REST fixture (pinned image — `:latest` rejects the
namespace encoding this test relies on) on a shared local-filesystem
warehouse, writes snapshots with pyiceberg, expires them through the Rust
binding, and verifies both the metadata AND the physical files: expired
snapshots gone, exclusive files deleted from storage, retained files intact,
table still scanning with exact row counts.

Requires a running docker daemon; skipped otherwise.
"""

import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import pyarrow as pa
import pytest

REST_FIXTURE_IMAGE = "apache/iceberg-rest-fixture:1.10.1"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "info"], check=True, capture_output=True, timeout=30
        )
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _docker_available(), reason="docker daemon not available"
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _now_ms() -> int:
    return int(time.time() * 1000)


@pytest.fixture(scope="module")
def rest_fixture():
    """REST fixture container on a file:// warehouse bind-mounted at the SAME
    path inside and outside the container, so server-written metadata and
    client-written data resolve identically for pyiceberg AND the Rust side."""
    warehouse = tempfile.mkdtemp(prefix="expire-e2e-", dir="/tmp")
    port = _free_port()
    container = subprocess.run(
        [
            "docker", "run", "-d", "--rm",
            "-p", f"{port}:8181",
            "-v", f"{warehouse}:{warehouse}",
            "-e", f"CATALOG_WAREHOUSE=file://{warehouse}",
            REST_FIXTURE_IMAGE,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    uri = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 90
        while True:
            try:
                with urllib.request.urlopen(f"{uri}/v1/config", timeout=2):
                    break
            except Exception:
                if time.time() > deadline:
                    logs = subprocess.run(
                        ["docker", "logs", container],
                        capture_output=True,
                        text=True,
                    )
                    raise RuntimeError(
                        f"REST fixture not healthy in 90s:\n{logs.stderr[-2000:]}"
                    )
                time.sleep(1)
        yield uri, warehouse
    finally:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True)
        shutil.rmtree(warehouse, ignore_errors=True)


def _local_path(uri: str) -> Path:
    return Path(uri.removeprefix("file://"))


def test_expire_snapshots_end_to_end(rest_fixture):
    from pyiceberg.catalog.rest import RestCatalog

    from pyiceberg_core import compaction

    uri, _warehouse = rest_fixture
    catalog = RestCatalog(name="rest", uri=uri)
    catalog.create_namespace("db")

    batch = pa.table({"id": pa.array([1, 2, 3, 4], type=pa.int32())})
    tbl = catalog.create_table("db.t", schema=batch.schema)
    for offset in (0, 4, 8):
        shifted = pa.table(
            {"id": pa.array([offset + v for v in (1, 2, 3, 4)], type=pa.int32())}
        )
        tbl.append(shifted)

    tbl = catalog.load_table("db.t")
    snapshots = tbl.metadata.snapshots
    assert len(snapshots) == 3
    current_id = tbl.metadata.current_snapshot_id
    expired_lists = [
        s.manifest_list for s in snapshots if s.snapshot_id != current_id
    ]
    retained_list = next(
        s.manifest_list for s in snapshots if s.snapshot_id == current_id
    )
    props = {"uri": uri}
    cutoff = _now_ms() + 60_000

    # Dry run: full report, zero changes.
    report = compaction.expire_snapshots(
        props, "rest.db.t", older_than_ms=cutoff, retain_last=1, dry_run=True
    )
    assert report["dry_run"] is True
    assert report["removed_snapshots"]["count"] == 2
    assert report["manifest_lists"] == 2
    assert report["deleted_files"] == 0
    tbl = catalog.load_table("db.t")
    assert len(tbl.metadata.snapshots) == 3, "dry run must not commit"
    for manifest_list in expired_lists + [retained_list]:
        assert _local_path(manifest_list).exists(), "dry run must not delete"

    # Real run: metadata commit first, then exclusive-file cleanup.
    report = compaction.expire_snapshots(
        props, "rest.db.t", older_than_ms=cutoff, retain_last=1
    )
    assert report["dry_run"] is False
    assert report["removed_snapshots"]["count"] == 2
    assert sorted(report["removed_snapshots"]["ids"]) == sorted(
        s.snapshot_id for s in snapshots if s.snapshot_id != current_id
    )
    assert report["removed_refs"] == []
    # Fast appends share manifests + data files with the retained head:
    # only the two old manifest lists are exclusive.
    assert report["manifest_lists"] == 2
    assert report["manifests"] == 0
    assert report["data_files"] == 0
    assert report["delete_files"] == 0
    assert report["stats_files"] == 0
    assert report["deleted_files"] == 2
    assert report["failed_deletes"] == 0

    # Metadata: expired snapshots gone, current retained.
    tbl = catalog.load_table("db.t")
    assert [s.snapshot_id for s in tbl.metadata.snapshots] == [current_id]

    # Storage: exclusive manifest lists physically deleted, retained intact.
    for manifest_list in expired_lists:
        assert not _local_path(manifest_list).exists()
    assert _local_path(retained_list).exists()

    # The table still scans every row.
    result = tbl.scan().to_arrow()
    assert result.num_rows == 12
    assert sorted(result["id"].to_pylist()) == list(range(1, 13))

    # Idempotency: a second call is a clean no-op (None, like rewrite_manifests).
    assert (
        compaction.expire_snapshots(
            props, "rest.db.t", older_than_ms=_now_ms() + 60_000, retain_last=1
        )
        is None
    )
    tbl = catalog.load_table("db.t")
    assert len(tbl.metadata.snapshots) == 1
    assert tbl.scan().to_arrow().num_rows == 12


def test_expire_snapshots_argument_validation(rest_fixture):
    from pyiceberg_core import compaction

    uri, _warehouse = rest_fixture
    with pytest.raises(ValueError, match="older_than_ms"):
        compaction.expire_snapshots({"uri": uri}, "rest.db.t", 0, 1)
    with pytest.raises(ValueError, match="retain_last"):
        compaction.expire_snapshots({"uri": uri}, "rest.db.t", _now_ms(), 0)
