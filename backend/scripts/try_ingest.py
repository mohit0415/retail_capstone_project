import argparse
import json
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BACKEND_ROOT))

DEFAULT_URL = "http://localhost:8000"

DEFAULT_FILE = BACKEND_ROOT / "data" / "policies" / "Information_Security_Access_Control_Policy.pdf"

CRLF = b"\r\n"


def call(request: urllib.request.Request) -> tuple[int, dict | str]:
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")

        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, raw
    except urllib.error.URLError as error:
        print(f"could not reach the API: {error.reason}")
        print("start it first with 'uv run python main.py' from the backend folder")
        sys.exit(1)


def mint_token(base_url: str, user_id: str, role: str, departments: list[str]) -> str:
    payload = json.dumps({"user_id": user_id, "role": role, "departments": departments}).encode()

    request = urllib.request.Request(
        f"{base_url}/auth/token",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    status, body = call(request)

    if status != 200:
        print(f"could not mint a token ({status}): {body}")
        sys.exit(1)

    return body["access_token"]


def upload(base_url: str, token: str, path: Path, force: bool) -> tuple[int, dict | str]:
    boundary = uuid.uuid4().hex

    body = b"".join(
        [
            f"--{boundary}".encode(),
            CRLF,
            f'Content-Disposition: form-data; name="file"; filename="{path.name}"'.encode(),
            CRLF,
            b"Content-Type: application/pdf",
            CRLF,
            CRLF,
            path.read_bytes(),
            CRLF,
            f"--{boundary}--".encode(),
            CRLF,
        ]
    )

    request = urllib.request.Request(
        f"{base_url}/ingest?force={'true' if force else 'false'}",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )

    return call(request)


def show_status(base_url: str, token: str) -> None:
    request = urllib.request.Request(
        f"{base_url}/ingest/status",
        headers={"Authorization": f"Bearer {token}"},
        method="GET",
    )

    status, body = call(request)

    if status != 200 or not isinstance(body, dict):
        print(f"\ncorpus status unavailable ({status}): {body}")
        return

    print(f"\nembedding model   {body.get('embed_model')}")
    print(f"corpus directory  {body.get('corpus_dir')}")

    documents = body.get("documents") or []

    if not documents:
        print("no documents indexed yet")
        return

    print(f"\n{'document':46} {'type':20} {'ver':6} {'parser':11} nodes")
    print("-" * 100)

    for document in documents:
        print(
            f"{str(document.get('document_title'))[:44]:46} "
            f"{str(document.get('doc_type')):20} "
            f"{str(document.get('version')):6} "
            f"{str(document.get('parsed_with')):11} "
            f"{document.get('node_count')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Mint an admin token and push one document through POST /ingest")
    parser.add_argument("--file", default=str(DEFAULT_FILE), help="document to upload")
    parser.add_argument("--url", default=DEFAULT_URL, help="base URL of the running API")
    parser.add_argument("--user", default="mohit", help="user_id to put in the token")
    parser.add_argument("--role", default="admin", help="role to put in the token")
    parser.add_argument("--department", action="append", default=[], help="department claim, repeatable")
    parser.add_argument("--keep", action="store_true", help="do not force, skip if the file hash is known")
    parser.add_argument("--status-only", action="store_true", help="only print what is already indexed")
    args = parser.parse_args()

    path = Path(args.file)

    token = mint_token(args.url, args.user, args.role, args.department)

    print(f"token minted for {args.user} as {args.role}")

    if args.status_only:
        show_status(args.url, token)
        return

    if not path.exists():
        print(f"no such file: {path}")
        sys.exit(1)

    print(f"uploading {path.name} ({path.stat().st_size} bytes), this waits for the whole parse")

    status, body = upload(args.url, token, path, force=not args.keep)

    if status != 200 or not isinstance(body, dict):
        print(f"\ningest failed with {status}:")
        print(json.dumps(body, indent=2) if isinstance(body, dict) else body)
        sys.exit(1)

    if body.get("status") == "skipped":
        print(f"\nskipped: {body.get('reason')}")
        show_status(args.url, token)
        return

    print(
        f"\nindexed {body['file_name']} as {body['doc_type']} v{body['version']} via {body['parsed_with']}"
    )
    print(f"  text nodes   {body['text_nodes']}")
    print(f"  table nodes  {body['table_nodes']}")
    print(f"  image nodes  {body['image_nodes']}")
    print(f"  superseded   {body['superseded_nodes']}")

    if body["image_nodes"] == 0:
        print("\nno image node was built. Run scripts/diagnose_ingestion.py --path <file> to see whether")
        print("the router found a drawn figure and whether the vision deployment answered.")

    show_status(args.url, token)


if __name__ == "__main__":
    main()
