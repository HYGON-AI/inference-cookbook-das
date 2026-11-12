"""Deploy the unchanged managed proxy with frontend keepalive above client TTL."""
import json
import os
from pathlib import Path
import uvicorn
import managed_proxy_ka1 as managed

managed.proxy.global_args = managed.proxy.parse_args()
args = managed.proxy.global_args
config = uvicorn.Config(managed.proxy.app, host=args.host, port=args.port,
                        access_log=False, timeout_keep_alive=60)
(Path(os.environ.get('HY4_PROXY_RUN_DIR',str(Path(__file__).resolve().parents[1]/'current/run')))/'frontend-transport.json').write_text(json.dumps({
    'pid': os.getpid(), 'timeout_keep_alive': config.timeout_keep_alive,
    'upstream_keepalive_expiry': 1, 'upstream_retries': 0,
    'model_request_retries_added': False}, indent=2))
uvicorn.Server(config).run()
