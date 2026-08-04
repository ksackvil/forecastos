from contextlib import contextmanager
from pathlib import Path
from typing import Generator
import requests

TIMEOUT_SEC = 30
CHUNK_BYTES = 1 << 20


class FileDownloader:
    """Download URLs into `data_dir`, skipping files already present.

    Bytes land in a sibling `.part` file that is renamed only once the transfer
    completes, so an interrupted run never leaves a truncated file behind for the
    next run to mistake for a finished download.
    """

    def __init__(self, data_dir: str | Path, cleanup: bool):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup = cleanup

    def download(self, url: str, filename: str | None = None) -> Path:
        """Return the local path for `url`, downloading it only if missing."""
        out_path = self.data_dir / (filename or url.split("/")[-1])

        if out_path.exists():
            print(f"already downloaded: {out_path}")
            return out_path

        part = out_path.with_name(out_path.name + ".part")
        print(f"downloading: {url}\nsaving to:   {out_path}")

        try:
            with requests.get(url, stream=True, timeout=TIMEOUT_SEC) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                done = 0

                with open(part, "wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            print(
                                f"\r  {_fmt(done)} / {_fmt(total)}"
                                f" ({done / total:4.0%})",
                                end="",
                            )

            part.replace(out_path)
        except BaseException:
            # Also catches KeyboardInterrupt: a half-written capture is worse than
            # no capture, since `out_path.exists()` would later skip re-downloading.
            part.unlink(missing_ok=True)
            raise

        print(f"\nsaved {out_path} ({_fmt(out_path.stat().st_size)})")
        return out_path

    @contextmanager
    def fetch(
        self,
        url: str,
        filename: str | None = None
    ) -> Generator[Path, None, None]:
        """Yield the local path for `url`, removing it afterwards if `cleanup`."""
        path = self.download(url, filename)
        try:
            yield path
        finally:
            if self.cleanup:
                print(f"removing {path}")
                path.unlink(missing_ok=True)


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
