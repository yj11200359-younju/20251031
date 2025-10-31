#!/usr/bin/env python3
"""간단한 정적 웹 서버.

`site/` 디렉터리에 있는 정적 자산을 제공한다.
"""

from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
from pathlib import Path
import argparse
import os
import sys

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_DIR = Path("site")


def run(directory: Path, host: str, port: int) -> None:
    directory = directory.resolve()
    if not directory.exists():
        raise FileNotFoundError(f"제공할 디렉터리를 찾을 수 없습니다: {directory}")

    os.chdir(directory)
    handler = SimpleHTTPRequestHandler

    with TCPServer((host, port), handler) as httpd:
        print(f"정적 파일을 제공하는 중입니다: http://{host}:{port}/")
        print(f"서비스 경로: {directory}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n서버를 종료합니다.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="간단한 정적 웹 서버 실행")
    parser.add_argument("--host", default=DEFAULT_HOST, help="바인드할 호스트 (기본값: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="포트 번호 (기본값: 8000)")
    parser.add_argument(
        "--directory",
        type=Path,
        default=DEFAULT_DIR,
        help="서비스할 디렉터리 경로 (기본값: site)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run(args.directory, args.host, args.port)


if __name__ == "__main__":
    main()
