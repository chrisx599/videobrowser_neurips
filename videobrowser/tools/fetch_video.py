from langchain_community.document_loaders import YoutubeLoader, YoutubeAudioLoader, GoogleApiYoutubeLoader, BiliBiliLoader
from langchain_core.documents import Document
import sys
import os
import glob
import random
import requests
import threading
import time
import yt_dlp
import shutil
from pytubefix import YouTube
from openai import OpenAI
from videobrowser.config import get_config
from videobrowser.utils.parser import clean_vtt_text, extract_youtube_id
from videobrowser.utils.cache import cache_manager


# ---------------------------------------------------------------------------
# Download statistics tracker
# ---------------------------------------------------------------------------
class _DownloadStats:
    """Thread-safe counters for video/audio/transcript download outcomes."""

    def __init__(self):
        self._lock = threading.Lock()
        self.video_cache_hit = 0
        self.video_download_ok = 0
        self.video_download_fail = 0
        self.audio_cache_hit = 0
        self.audio_download_ok = 0
        self.audio_download_fail = 0
        self.transcript_cache_hit = 0
        self.transcript_fetch_ok = 0
        self.transcript_fetch_fail = 0

    def record(self, category: str, outcome: str):
        with self._lock:
            attr = f"{category}_{outcome}"
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr) + 1)

    def summary(self) -> str:
        with self._lock:
            v_total = self.video_cache_hit + self.video_download_ok + self.video_download_fail
            a_total = self.audio_cache_hit + self.audio_download_ok + self.audio_download_fail
            t_total = self.transcript_cache_hit + self.transcript_fetch_ok + self.transcript_fetch_fail
            lines = [
                "📊 Download Statistics:",
                f"  Video     — total: {v_total}, cache_hit: {self.video_cache_hit}, downloaded: {self.video_download_ok}, failed: {self.video_download_fail}",
                f"  Audio     — total: {a_total}, cache_hit: {self.audio_cache_hit}, downloaded: {self.audio_download_ok}, failed: {self.audio_download_fail}",
                f"  Transcript— total: {t_total}, cache_hit: {self.transcript_cache_hit}, fetched: {self.transcript_fetch_ok}, failed: {self.transcript_fetch_fail}",
            ]
            return "\n".join(lines)

    def reset(self):
        with self._lock:
            for attr in list(vars(self)):
                if attr != "_lock":
                    setattr(self, attr, 0)


download_stats = _DownloadStats()


# ---------------------------------------------------------------------------
# Proxy pool — loaded from data/ip_pools.txt
# Format per line: host:port\nuser:pass  (parsed from Cliproxy dashboard export)
# All connections go through port 443 (the only port allowed by our firewall).
# ---------------------------------------------------------------------------
_PROXY_POOL: list[str] = []  # list of "socks5h://user:pass@host:443"
_PROXY_POOL_SOURCE: str | None = None


def _is_proxy_enabled() -> bool:
    proxy_cfg = getattr(get_config(), "proxy", None)
    return bool(getattr(proxy_cfg, "enabled", True))


def _get_proxy_pool_path() -> str:
    proxy_cfg = getattr(get_config(), "proxy", None)
    return str(getattr(proxy_cfg, "pool_path", "data/ip_pools.txt"))


def _get_validated_pool_path() -> str:
    proxy_cfg = getattr(get_config(), "proxy", None)
    return str(getattr(proxy_cfg, "validated_pool_path", "data/ip_pools_alive.txt"))


def _get_proxy_protocol() -> str:
    proxy_cfg = getattr(get_config(), "proxy", None)
    return str(getattr(proxy_cfg, "protocol", "socks5h"))


def _get_proxy_credentials() -> tuple[str | None, str | None]:
    proxy_cfg = getattr(get_config(), "proxy", None)
    user = getattr(proxy_cfg, "username", None) or os.getenv("PROXY_USERNAME")
    passwd = getattr(proxy_cfg, "password", None) or os.getenv("PROXY_PASSWORD")
    return user, passwd


def _load_proxy_pool(path: str | None = None) -> list[str]:
    """Parse proxy list into proxy URLs.

    Supports two formats:
    - TXT: one ``host:port:user:pass`` per line
    - JSON: list of ``{"ip": ..., "port": ..., "entryPoint": ...}`` objects
      (Oxylabs dashboard export).  Credentials come from config or env.

    When the entryPoint goes through an SSH tunnel, the host is replaced
    with ``127.0.0.1`` so the request goes through the local tunnel port.
    """
    if path is None:
        path = _get_proxy_pool_path()
    if not os.path.exists(path):
        return []
    protocol = _get_proxy_protocol()
    proxies = []

    if path.endswith(".json"):
        import json as _json
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
        user, passwd = _get_proxy_credentials()
        if not user or not passwd:
            print("⚠️ Proxy credentials missing for JSON pool. "
                  "Set proxy.username/password in config or PROXY_USERNAME/PROXY_PASSWORD env.")
            return []
        proxy_cfg = getattr(get_config(), "proxy", None)
        use_tunnel = getattr(proxy_cfg, "ssh_tunnel", False)
        for entry in data:
            port = entry.get("port", 8001)
            host = "127.0.0.1" if use_tunnel else entry.get("entryPoint", "127.0.0.1")
            proxies.append(f"{protocol}://{user}:{passwd}@{host}:{port}")
    else:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) == 4:
                host, port, user, passwd = parts
                proxies.append(f"{protocol}://{user}:{passwd}@{host}:{port}")
    return proxies


def _test_proxy(proxy_url: str, timeout: float = 10) -> bool:
    """Test if a SOCKS5 proxy is reachable by connecting to httpbin."""
    try:
        r = requests.get(
            "https://httpbin.org/ip",
            proxies={"http": proxy_url, "https": proxy_url},
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False


def validate_proxy_pool(path: str | None = None, timeout: float = 10) -> list[str]:
    """Load proxies, test connectivity, write alive list to ip_pools_alive.txt.

    If ``proxy.enabled`` is false in the loaded config, skip loading and
    testing entirely — cached videos don't need network access, and direct
    connections are fine for uncached ones.
    """
    global _PROXY_POOL, _PROXY_POOL_SOURCE
    proxy_cfg = getattr(get_config(), "proxy", None)
    if proxy_cfg is not None and not getattr(proxy_cfg, "enabled", True):
        print("🔌 Proxy disabled in config — skipping proxy pool validation.")
        _PROXY_POOL = []
        _PROXY_POOL_SOURCE = None
        return []
    if path is None:
        path = _get_proxy_pool_path()
    raw = _load_proxy_pool(path)
    if not raw:
        print("⚠️ No proxies found in", path)
        return []

    print(f"🔍 Testing {len(raw)} proxies (timeout={timeout}s)...")
    alive: list[str] = []
    for proxy in raw:
        label = proxy.split("@")[0].split("//")[1].split(":")[0][:8]
        ok = _test_proxy(proxy, timeout=timeout)
        status = "✓" if ok else "✗"
        print(f"  {status} {label}... {'alive' if ok else 'dead'}")
        if ok:
            alive.append(proxy)

    print(f"📊 Proxy pool: {len(alive)}/{len(raw)} alive")

    # Write alive proxies so child processes can skip re-validation
    validated_pool_path = _get_validated_pool_path()
    with open(validated_pool_path, "w", encoding="utf-8") as f:
        for p in alive:
            f.write(p + "\n")

    _PROXY_POOL = alive
    _PROXY_POOL_SOURCE = validated_pool_path
    return alive


def get_random_proxy() -> str | None:
    """Return a random proxy URL from the validated pool."""
    global _PROXY_POOL, _PROXY_POOL_SOURCE
    if not _is_proxy_enabled():
        return None

    validated_pool_path = _get_validated_pool_path()
    if _PROXY_POOL_SOURCE != validated_pool_path:
        _PROXY_POOL = []
        _PROXY_POOL_SOURCE = validated_pool_path

    if not _PROXY_POOL:
        # Try pre-validated file first (written by validate_proxy_pool in main process)
        if os.path.exists(validated_pool_path):
            _PROXY_POOL = [
                line.strip()
                for line in open(validated_pool_path, encoding="utf-8")
                if line.strip()
            ]
        if not _PROXY_POOL:
            # No validated file — load raw pool without re-testing.
            # validate_proxy_pool() should be called once in the main process
            # before spawning workers; avoid noisy re-validation here.
            _PROXY_POOL = _load_proxy_pool()
    if not _PROXY_POOL:
        return None
    return random.choice(_PROXY_POOL)


def _remove_dead_proxy(proxy_url: str) -> None:
    """Remove a dead proxy from the in-memory pool."""
    global _PROXY_POOL
    if proxy_url in _PROXY_POOL:
        _PROXY_POOL.remove(proxy_url)
        print(f"🗑️ Removed dead proxy ({len(_PROXY_POOL)} remaining)")


# Global rate limiter for YouTube downloads — max 1 concurrent download,
# with a cooldown between requests to avoid rate limiting.
_download_lock = threading.Lock()
_last_download_time = 0.0
_DOWNLOAD_COOLDOWN = 2.0  # seconds between YouTube requests


def _throttle_download():
    """Acquire download lock and enforce cooldown between YouTube requests."""
    global _last_download_time
    _download_lock.acquire()
    elapsed = time.time() - _last_download_time
    if elapsed < _DOWNLOAD_COOLDOWN:
        time.sleep(_DOWNLOAD_COOLDOWN - elapsed)


def _release_download():
    """Release download lock and record timestamp."""
    global _last_download_time
    _last_download_time = time.time()
    _download_lock.release()

# Patch pytube if needed (though usually importing pytubefix directly is better)
try:
    import pytubefix
    sys.modules["pytube"] = pytubefix
except ImportError:
    pass


def get_valid_yt_dlp_cookiefile(cookie_path: str = "data/cookies.txt") -> str | None:
    if not os.path.exists(cookie_path):
        return None

    try:
        with open(cookie_path, "r", encoding="utf-8") as f:
            first_nonempty = next((line.strip() for line in f if line.strip()), "")
    except OSError as e:
        print(f"⚠️ Could not read cookie file {cookie_path}: {e}")
        return None

    if first_nonempty.startswith("# Netscape HTTP Cookie File"):
        return cookie_path

    print(f"⚠️ Ignoring invalid yt-dlp cookie file: {cookie_path}")
    return None


def _generate_po_token() -> tuple[str, str]:
    """Generate a PO token using the Node.js youtube-po-token-generator package."""
    import json
    import subprocess

    node_path = shutil.which("node")
    if node_path is None:
        raise RuntimeError("Node.js not found")

    npm_path = shutil.which("npm")
    if npm_path is None:
        raise RuntimeError("npm not found")
    global_root = subprocess.check_output(
        [npm_path, "root", "-g"],
        timeout=5,
    ).decode().strip()

    script = (
        "const{generate}=require('youtube-po-token-generator');"
        "generate().then(r=>process.stdout.write(JSON.stringify(r)))"
        ".catch(e=>{process.stderr.write(String(e));process.exit(1)});"
    )
    result = subprocess.check_output(
        [node_path, "-e", script],
        env={**os.environ, "NODE_PATH": global_root},
        timeout=30,
    )
    data = json.loads(result)
    return data["visitorData"], data["poToken"]



def get_ytdlp_po_opts() -> dict:
    """Return yt-dlp options dict with JS runtime, EJS solver, proxy, and optional PO token.

    PO token is controlled by the YT_USE_PO_TOKEN env var (default: false).
    With residential proxy IPs, PO token is typically unnecessary.
    """
    opts: dict = {
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:github"],
        "sleep_interval_requests": 1,    # seconds between HTTP requests
        "retry_sleep_functions": {"http": lambda n: 3 * (2 ** n)},  # exponential backoff
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 15,  # seconds — fail fast on dead proxies
    }
    # Prefer `tv` client (no JS cipher, no PO token required, fewer bot checks);
    # fall back to web_safari and web when tv can't serve the format.
    youtube_args: dict = {"player_client": ["tv", "web_safari", "web"]}
    po_data = _get_or_generate_po_token()
    if po_data:
        youtube_args["po_token"] = [f"web+{po_data['poToken']}"]
        youtube_args["visitor_data"] = [po_data["visitorData"]]
    opts["extractor_args"] = {"youtube": youtube_args}
    proxy_cfg = getattr(get_config(), "proxy", None)
    use_cookies = getattr(proxy_cfg, "use_cookies", True)
    if use_cookies:
        cookie_file = get_valid_yt_dlp_cookiefile()
        if cookie_file:
            opts["cookiefile"] = cookie_file
    if _is_proxy_enabled():
        proxy = get_random_proxy()
        if proxy:
            opts["proxy"] = proxy
    return opts


def _is_usable_media_file(path: str | None) -> bool:
    return bool(path and os.path.exists(path) and os.path.getsize(path) > 0)


def _drop_invalid_cache_file(path: str | None) -> None:
    if not path or not os.path.exists(path):
        return
    try:
        os.remove(path)
        print(f"🗑️ Removed invalid cached media: {path}")
    except OSError as exc:
        print(f"⚠️ Failed to remove invalid cached media {path}: {exc}")


def ytdlp_download_with_throttle(ydl_opts: dict, video_url: str, download: bool = True) -> dict:
    """Run yt-dlp extract_info with global throttle and retry on rate limit.

    Handles proxy failures by swapping to a different proxy on retry.
    Returns the info dict from extract_info.
    """
    max_retries = 3
    for attempt in range(max_retries):
        _throttle_download()
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=download)
            return info
        except Exception as e:
            err_str = str(e)
            is_proxy_error = any(s in err_str for s in (
                "SOCKS server failure", "ProxyError", "proxy",
                "Connection refused", "Connection reset",
            ))
            is_rate_limit = any(s in err_str for s in ("rate-limited", "429", "403"))
            is_bot_challenge = any(s in err_str for s in (
                # YouTube returns this on the IP-level bot challenge; only a
                # different exit IP reliably clears it.
                "Sign in to confirm", "not a bot", "cookies-from-browser",
            ))

            if (is_proxy_error or is_bot_challenge) and attempt < max_retries - 1:
                dead_proxy = ydl_opts.get("proxy")
                if is_proxy_error and dead_proxy:
                    _remove_dead_proxy(dead_proxy)
                new_proxy = get_random_proxy()
                if new_proxy:
                    ydl_opts["proxy"] = new_proxy
                    label = "bot-challenge" if is_bot_challenge else "proxy"
                    print(f"🔄 {label} hit, switching proxy (attempt {attempt + 1}/{max_retries})")
                    if is_bot_challenge:
                        time.sleep(3 * (attempt + 1))
                    continue
                else:
                    ydl_opts.pop("proxy", None)
                    print("⚠️ No proxies left, trying direct connection...")
                    continue
            elif is_rate_limit and attempt < max_retries - 1:
                wait = 10 * (2 ** attempt)
                print(f"⏳ Rate limited, waiting {wait}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(wait)
                continue
            raise
        finally:
            if _download_lock.locked():
                _release_download()
    raise RuntimeError(f"Failed after {max_retries} retries for {video_url}")


_po_token_cache: dict | None = None
_PO_TOKEN_PATH = "data/po_token.json"


def _get_or_generate_po_token() -> dict | None:
    """Return PO token dict, auto-generating if missing or expired."""
    global _po_token_cache
    if _po_token_cache is not None:
        return _po_token_cache

    # 1. Try loading from file
    if os.path.exists(_PO_TOKEN_PATH):
        try:
            import json as _json
            with open(_PO_TOKEN_PATH, encoding="utf-8") as f:
                _po_token_cache = _json.load(f)
            if _po_token_cache.get("visitorData") and _po_token_cache.get("poToken"):
                return _po_token_cache
        except Exception:
            pass

    # 2. Auto-generate
    try:
        print("🔑 Generating PO token...")
        visitor_data, po_token = _generate_po_token()
        _po_token_cache = {"visitorData": visitor_data, "poToken": po_token}
        # Save to file for reuse across processes
        import json as _json
        with open(_PO_TOKEN_PATH, "w", encoding="utf-8") as f:
            _json.dump(_po_token_cache, f, indent=2)
        print("✅ PO token generated and saved.")
        return _po_token_cache
    except Exception as e:
        print(f"⚠️ PO token generation failed: {e}")
        return None


def invalidate_po_token():
    """Clear cached PO token so next call re-generates."""
    global _po_token_cache
    _po_token_cache = None
    if os.path.exists(_PO_TOKEN_PATH):
        os.remove(_PO_TOKEN_PATH)


def create_pytubefix_youtube(video_url: str) -> YouTube:
    kwargs: dict = {
        "client": "WEB",
        "use_oauth": False,
        "allow_oauth_cache": False,
    }
    # Use PO token — auto-generated if needed
    po_data = _get_or_generate_po_token()
    if po_data:
        kwargs["use_po_token"] = True
        kwargs["po_token_verifier"] = lambda: (
            po_data["visitorData"],
            po_data["poToken"],
        )
    if _is_proxy_enabled():
        proxy = get_random_proxy()
        if proxy:
            kwargs["proxies"] = {"http": proxy, "https": proxy}
    return YouTube(video_url, **kwargs)

def fetch_with_oxylabs(video_url: str) -> list[Document]:
    """
    Fetches the transcript using Oxylabs Realtime API.
    """
    config = get_config()
    username = config.transcript.oxylabs_username or os.getenv("OXYLABS_USERNAME")
    password = config.transcript.oxylabs_password or os.getenv("OXYLABS_PASSWORD")

    if not username or not password:
        print("⚠️ Oxylabs credentials missing. Please set OXYLABS_USERNAME and OXYLABS_PASSWORD.")
        return []
    
    video_id = extract_youtube_id(video_url)
    payload = {
        'source': 'youtube_transcript',
        'query': video_id,
        'context': [
            {'key': 'language_code', 'value': 'en'},
            {'key': 'transcript_origin', 'value': 'auto_generated'}
        ]
    }

    try:
        response = requests.post(
            'https://realtime.oxylabs.io/v1/queries',
            auth=(username, password),
            json=payload,
            timeout=60 # Add timeout for safety
        )
        response.raise_for_status()
        data = response.json()
        
        results = data.get('results', [])
        if not results:
             print(f"⚠️ Oxylabs returned no results for {video_url}")
             return []

        # Parse nested structure
        # results -> [0] -> content -> [list of segments]
        # segment -> transcriptSegmentRenderer -> snippet -> runs -> [0] -> text
        
        content = results[0].get('content', [])
        transcript_parts = []
        
        if isinstance(content, list):
            for item in content:
                # Handle the specific structure provided by user
                renderer = item.get('transcriptSegmentRenderer')
                if renderer:
                    snippet = renderer.get('snippet', None)
                    if snippet:
                        runs = snippet.get('runs', [])
                        for run in runs:
                            text = run.get('text')
                            if text:
                                transcript_parts.append(text)
        
        transcript_text = " ".join(transcript_parts)
        
        if not transcript_text:
             print(f"⚠️ Could not parse transcript text from Oxylabs response for {video_id}")
             transcript_text = "" # Explicitly set to empty to trigger subtitle fallback
        else:
             return [Document(
                page_content=transcript_text,
                metadata={"source": video_url, "provider": "oxylabs"}
            )]

    except Exception as e:
        print(f"⚠️ Oxylabs fetch failed for {video_url}: {e}")
        transcript_text = "" # Ensure transcript_text is empty to trigger subtitle fallback
    
    # If transcript is empty, try to get subtitles
    if not transcript_text:
        print(f"🔄 No transcript found, attempting to fetch subtitles for {video_id}...")
        subtitle_payload = {
            'source': 'youtube_subtitles',
            'query': video_id,
            'context': [
                {'key': 'language_code', 'value': 'en'},
                {'key': 'subtitle_origin', 'value': 'auto_generated'}
            ]
        }
        try:
            subtitle_response = requests.post(
                'https://realtime.oxylabs.io/v1/queries',
                auth=(username, password),
                json=subtitle_payload,
                timeout=60
            )
            subtitle_response.raise_for_status()
            subtitle_data = subtitle_response.json()

            subtitle_results = subtitle_data.get('results', [])
            if not subtitle_results:
                print(f"⚠️ Oxylabs returned no subtitle results for {video_url}")
                return []
            
            # Parse subtitle structure
            # results -> [0] -> content -> auto_generated -> <language_code> -> events
            # event -> segs -> [0] -> utf8
            
            content_block = subtitle_results[0].get('content', {})
            auto_generated = content_block.get('auto_generated', {})
            english_subtitles = auto_generated.get('en', {}) # Assuming 'en' for now
            events = english_subtitles.get('events', [])
            
            subtitle_parts = []
            for event in events:
                segs = event.get('segs', [])
                for seg in segs:
                    text = seg.get('utf8')
                    if text:
                        subtitle_parts.append(text)
            
            transcript_text = " ".join(subtitle_parts)

            if not transcript_text:
                print(f"⚠️ Could not parse subtitle text from Oxylabs response for {video_id}")
                # print(f"DEBUG subtitle content sample: {subtitle_results}") # Debugging aid
                return []
            
            print(f"✅ Successfully fetched subtitles for {video_id}")

        except Exception as e:
            print(f"⚠️ Oxylabs subtitle fetch failed for {video_url}: {e}")
            return []

    if transcript_text:
        return [Document(
            page_content=transcript_text,
            metadata={"source": video_url, "provider": "oxylabs"}
        )]
    else:
        return []

def fetch_with_ytdlp(video_url: str) -> list[Document]:
    """
    Fetches transcript using yt-dlp (downloading subtitles).
    """
    import tempfile
    
    # Create a safe temp directory for this download
    video_id = extract_youtube_id(video_url)
    temp_dir = f"data/temp/subs/{video_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'outtmpl': f'{temp_dir}/%(id)s',
        'quiet': True,
        'no_warnings': True,
        **get_ytdlp_po_opts(),
    }

    cookie_file = get_valid_yt_dlp_cookiefile()
    if cookie_file:
        ydl_opts['cookiefile'] = cookie_file
        print("🍪 Using cookies.txt for yt-dlp...")

    try:
        print(f"🔍 Fetching transcript for {video_url} using yt-dlp...")
        info = ytdlp_download_with_throttle(ydl_opts, video_url, download=False)
        if info.get('is_live'):
            print(f"⚠️ Skipping live video: {video_url}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return []

        ytdlp_download_with_throttle(ydl_opts, video_url, download=True)
        
        # Find the vtt file
        vtt_files = glob.glob(f"{temp_dir}/*.vtt")
        if not vtt_files:
            print(f"⚠️ yt-dlp: No subtitles found for {video_url}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return []
            
        vtt_path = vtt_files[0]
        
        # Read raw content
        with open(vtt_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
            
        # Clean using utility function
        transcript_text = clean_vtt_text(raw_content)
        
        # Cleanup temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if not transcript_text:
             return []

        return [Document(
            page_content=transcript_text,
            metadata={"source": video_url, "provider": "ytdlp"}
        )]

    except Exception as e:
        print(f"⚠️ yt-dlp fetch failed for {video_url}: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return []

def fetch_with_whisper(video_url: str) -> list[Document]:
    """
    Fetches transcript using yt-dlp to download audio and OpenAI Whisper to transcribe.
    """
    config = get_config()
    cache_manager.refresh()
    
    # 1. Check Audio/Video Cache
    is_video_source = False
    audio_path = None
    if config.cache.enabled and cache_manager.has_audio(video_url):
        audio_path = cache_manager.get_audio_path(video_url)
        if _is_usable_media_file(audio_path):
            print(f"📦 Cache hit for audio file: {audio_path}, using for transcription...")
            download_stats.record("audio", "cache_hit")
            use_temp = False
        else:
            _drop_invalid_cache_file(audio_path)
            audio_path = None
    elif config.cache.enabled and cache_manager.has_video(video_url):
        audio_path = cache_manager.get_video_path(video_url)
        if _is_usable_media_file(audio_path):
            print(f"📦 Cache hit for video file: {audio_path}, using for transcription...")
            download_stats.record("audio", "cache_hit")
            is_video_source = True
            use_temp = False
        else:
            _drop_invalid_cache_file(audio_path)
            audio_path = None
        
    # 2. Download Audio
    if audio_path is None:
        video_id = extract_youtube_id(video_url)
        downloader = config.watcher.video_downloader
        
        # Paths
        if config.cache.enabled:
            audio_dir = str(cache_manager.audio_dir)
            use_temp = False
        else:
            audio_dir = f"data/temp/audio/{video_id}"
            os.makedirs(audio_dir, exist_ok=True)
            use_temp = True
            
        audio_path = os.path.join(audio_dir, f"{video_id}.mp3")
        downloaded_via_pytube = False

        # Attempt Pytubefix if configured
        if downloader == "pytubefix":
            try:
                print(f"🔍 Fetching audio for {video_url} using pytubefix...")
                yt = create_pytubefix_youtube(video_url)
                max_dur = config.watcher.max_duration_seconds
                if max_dur and getattr(yt, "length", 0) and yt.length > max_dur:
                    print(f"⚠️ Skipping long video ({yt.length}s > {max_dur}s) for Whisper: {video_url}")
                    if use_temp: shutil.rmtree(audio_dir, ignore_errors=True)
                    return []
                # Filter for audio
                stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
                if stream:
                    # Download raw
                    raw_name = f"{video_id}_raw"
                    # pytube appends extension automatically, but we can't easily predict it (mp4/webm)
                    # so we let it download and find it
                    out_file = stream.download(output_path=audio_dir, filename=raw_name)
                    
                    # Convert to MP3 16k mono for Whisper
                    import subprocess
                    cmd = [
                        "ffmpeg", "-i", out_file, 
                        "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", 
                        "-y", audio_path
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if _is_usable_media_file(audio_path):
                        downloaded_via_pytube = True
                        # Clean raw file
                        if os.path.exists(out_file) and out_file != audio_path:
                            os.remove(out_file)
                            
            except Exception as e:
                print(f"⚠️ pytubefix audio fetch failed: {e}. Falling back to yt-dlp...")

        # Fallback to yt-dlp if pytube failed or wasn't used
        if not downloaded_via_pytube and not os.path.exists(audio_path):
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '32',
                }],
                'outtmpl': f'{audio_dir}/{video_id}.%(ext)s' if config.cache.enabled else f'{audio_dir}/%(id)s.%(ext)s',
                'quiet': True,
                'no_warnings': True,
                **get_ytdlp_po_opts(),
            }
            
            try:
                print(f"🔍 Fetching audio for {video_url} using yt-dlp...")
                info = ytdlp_download_with_throttle(ydl_opts, video_url, download=False)
                if info.get('is_live'):
                    print(f"⚠️ Skipping live video for Whisper: {video_url}")
                    if use_temp: shutil.rmtree(audio_dir, ignore_errors=True)
                    return []
                max_dur = config.watcher.max_duration_seconds
                if max_dur and info.get('duration', 0) > max_dur:
                    print(f"⚠️ Skipping long video ({int(info['duration'])}s > {max_dur}s) for Whisper: {video_url}")
                    if use_temp: shutil.rmtree(audio_dir, ignore_errors=True)
                    return []
                ytdlp_download_with_throttle(ydl_opts, video_url, download=True)
            except Exception as e:
                print(f"⚠️ Whisper download failed (yt-dlp) for {video_url}: {e}")
                download_stats.record("audio", "download_fail")
                if use_temp: shutil.rmtree(audio_dir, ignore_errors=True)
                return []
                
        # Final Verification
        if not config.cache.enabled and not _is_usable_media_file(audio_path):
             # Find it if name differed (yt-dlp dynamic ext)
             found = glob.glob(f"{audio_dir}/*.mp3")
             if found:
                 audio_path = found[0]
             else:
                 print(f"⚠️ No audio file found for {video_url}")
                 download_stats.record("audio", "download_fail")
                 if use_temp: shutil.rmtree(audio_dir, ignore_errors=True)
                 return []

        if not _is_usable_media_file(audio_path):
             print(f"⚠️ Audio path valid but file missing: {audio_path}")
             download_stats.record("audio", "download_fail")
             return []

        download_stats.record("audio", "download_ok")

    try:
        # Transcribe with Whisper
        api_key = config.transcript.api_key or config.llm.default.api_key
        model = config.transcript.model or "whisper-1"
        base_url = config.transcript.base_url
        language = config.transcript.language
        
        if not api_key and not base_url:
             print("⚠️ ASR client configuration missing. Set transcript.api_key or transcript.base_url.")
             return []

        client_kwargs = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        client = OpenAI(**client_kwargs)
        
        processing_path = audio_path
        is_temp_conversion = False
        
        # Always extract audio if source is a video file
        if is_video_source:
            # Determine target path for extracted audio
            if config.cache.enabled:
                target_audio_path = cache_manager.get_audio_storage_path(video_url, ext="mp3")
                
                # Check if already exists
                if _is_usable_media_file(target_audio_path):
                    print(f"📦 Cache hit for extracted audio: {target_audio_path}")
                    processing_path = target_audio_path
                    is_temp_conversion = False
                else:
                    print(f"🔄 Extracting audio from cached video to: {target_audio_path}...")
                    try:
                        import subprocess
                        # Use ffmpeg to extract
                        cmd = [
                            "ffmpeg", "-i", audio_path, 
                            "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", 
                            "-y", target_audio_path
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        if _is_usable_media_file(target_audio_path):
                            processing_path = target_audio_path
                            print(f"✅ Audio extracted and cached.")
                        else:
                            print("⚠️ FFmpeg failed to create output file.")
                            _drop_invalid_cache_file(audio_path)
                            _drop_invalid_cache_file(target_audio_path)
                            return []
                    except Exception as fe:
                        print(f"⚠️ FFmpeg processing failed: {fe}")
                        _drop_invalid_cache_file(audio_path)
                        _drop_invalid_cache_file(target_audio_path)
                        return []
            else:
                # Use temp path
                import uuid
                import subprocess
                
                temp_extract_dir = "data/temp/whisper_extract"
                os.makedirs(temp_extract_dir, exist_ok=True)
                target_audio_path = os.path.join(temp_extract_dir, f"{uuid.uuid4()}.mp3")
                
                print(f"🔄 Extracting audio from video to temp: {target_audio_path}...")
                try:
                    cmd = [
                        "ffmpeg", "-i", audio_path, 
                        "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", 
                        "-y", target_audio_path
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    if _is_usable_media_file(target_audio_path):
                        processing_path = target_audio_path
                        is_temp_conversion = True
                        print(f"✅ Audio extracted to temp.")
                    else:
                        print("⚠️ FFmpeg failed to create output file.")
                        _drop_invalid_cache_file(target_audio_path)
                        return []
                except Exception as fe:
                    print(f"⚠️ FFmpeg processing failed: {fe}")
                    _drop_invalid_cache_file(target_audio_path)
                    return []

        with open(processing_path, "rb") as audio_file:
            transcription_kwargs = {
                "model": model,
                "file": audio_file,
                "response_format": "verbose_json",
            }
            if language:
                transcription_kwargs["language"] = language

            response = client.audio.transcriptions.create(**transcription_kwargs)
            
        # Extract segments with timestamps
        segments = []
        duration = 0.0
        
        # The response object has a 'segments' attribute which is a list of objects
        if hasattr(response, 'segments') and getattr(response, "segments", None):
            duration = getattr(response, "duration", 0.0)
            for seg in response.segments:
                # Access attributes directly as per OpenAI python lib
                segments.append({
                    "start": getattr(seg, "start", 0.0),
                    "end": getattr(seg, "end", 0.0),
                    "text": getattr(seg, "text", "").strip()
                })
        # If accessing as dict (just in case it changes or is different version)
        elif isinstance(response, dict) and "segments" in response:
             duration = response.get("duration", 0.0)
             for seg in response["segments"]:
                segments.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text", "").strip()
                })
        
        transcript_text = response.text
        
        # Cleanup temp conversion file
        if is_temp_conversion and os.path.exists(processing_path):
            os.remove(processing_path)
            # Try to remove dir if empty
            try:
                os.rmdir(os.path.dirname(processing_path))
            except:
                pass
        
        # Cleanup if temp download (video file)
        if use_temp:
            shutil.rmtree(os.path.dirname(audio_path), ignore_errors=True)
        
        if not transcript_text:
             return []
             
        return [Document(
            page_content=transcript_text,
            metadata={
                "source": video_url, 
                "provider": "whisper",
                "audio_duration": duration,
                "segments": segments
            }
        )]
        
    except Exception as e:
        print(f"⚠️ Whisper transcription failed for {video_url}: {e}")
        return []

def fetch_transcript_with_timestamps(video_url: str) -> list[dict]:
    """
    Fetches transcript with segment-level timestamps.
    Returns: list[dict] where each dict is {"start": float, "end": float, "text": str}
    """
    config = get_config()
    
    # 1. Try Cache
    if config.cache.enabled:
        cached_segments = cache_manager.get_transcript_with_timestamps(video_url)
        if cached_segments:
            print(f"📦 Cache hit for transcript with timestamps: {video_url}")
            download_stats.record("transcript", "cache_hit")
            return cached_segments
    
    docs = []
    if config.transcript.provider == "whisper":
        docs = fetch_with_whisper(video_url)
    
    # Add other providers if they support timestamps later
    
    if docs and "segments" in docs[0].metadata:
        segments = docs[0].metadata["segments"]
        
        # 2. Save to Cache
        if config.cache.enabled:
            cache_manager.save_transcript_with_timestamps(video_url, segments)
            
        return segments
    
    return []

def fetch_youtube_video_transcript(video_url: str) -> list:
    """
    Fetches the transcript (subtitles) of a YouTube video.
    Returns a list of LangChain Document objects.
    """
    config = get_config()
    
    # 1. Try Cache
    if config.cache.enabled:
        cached_text = cache_manager.get_transcript(video_url)
        if cached_text:
            print(f"📦 Cache hit for transcript: {video_url}")
            download_stats.record("transcript", "cache_hit")
            return [Document(page_content=cached_text, metadata={"source": video_url, "provider": "cache"})]

    # 2. Fetch from Provider
    docs = []
    if config.transcript.provider == "oxylabs":
        print(f"🔍 Fetching transcript for {video_url} using Oxylabs...")
        docs = fetch_with_oxylabs(video_url)
    elif config.transcript.provider == "ytdlp":
        docs = fetch_with_ytdlp(video_url)
    elif config.transcript.provider == "whisper":
        docs = fetch_with_whisper(video_url)
    else:
        # Default to local
        try:
            loader = YoutubeLoader.from_youtube_url(
                video_url, 
                add_video_info=True,
            )
            docs = loader.load()
        except Exception as e:
            print(f"⚠️ Transcript fetch failed for {video_url}: {e}")
            docs = []

    # 3. Save to Cache & record stats
    if docs:
        download_stats.record("transcript", "fetch_ok")
        if config.cache.enabled:
            cache_manager.save_transcript(video_url, docs[0].page_content)
    else:
        download_stats.record("transcript", "fetch_fail")

    return docs


def _try_download(method: str, video_url: str, video_id: str, output_dir: str, config) -> str | None:
    """Attempt download with a single method. Returns path, 'skip' for live, or None on failure."""
    if method == "ytdlp":
        out_tmpl = os.path.join(output_dir, f"{video_id}.%(ext)s")
        ydl_opts = {
            'format': 'best[height<=360][ext=mp4]/best[height<=360]/best',
            'outtmpl': out_tmpl,
            'quiet': True,
            'no_warnings': True,
            **get_ytdlp_po_opts(),
        }
        try:
            print(f"⬇️ Downloading video '{video_id}' using yt-dlp...")
            info = ytdlp_download_with_throttle(ydl_opts, video_url, download=False)
            if info.get('is_live'):
                print(f"⚠️ Skipping live video for download: {video_url}")
                return "skip"
            max_dur = config.watcher.max_duration_seconds
            if max_dur and info.get('duration', 0) > max_dur:
                print(f"⚠️ Skipping long video ({int(info['duration'])}s > {max_dur}s) for download: {video_url}")
                return "skip"
            info = ytdlp_download_with_throttle(ydl_opts, video_url, download=True)
            file_path = yt_dlp.YoutubeDL(ydl_opts).prepare_filename(info)
            if os.path.exists(file_path):
                print(f"✅ Video downloaded to: {file_path}")
                return file_path
        except Exception as e:
            print(f"❌ yt-dlp download failed: {e}")
    else:
        for attempt in range(2):
            try:
                yt = create_pytubefix_youtube(video_url)
                if yt.vid_info.get('playabilityStatus', {}).get('liveStreamability'):
                    print(f"⚠️ Skipping live video for download (pytubefix): {video_url}")
                    return "skip"
                max_dur = config.watcher.max_duration_seconds
                if max_dur and getattr(yt, "length", 0) and yt.length > max_dur:
                    print(f"⚠️ Skipping long video ({yt.length}s > {max_dur}s) for download (pytubefix): {video_url}")
                    return "skip"
                stream = yt.streams.filter(res="360p", file_extension="mp4", progressive=True).first()
                if not stream:
                    stream = yt.streams.filter(file_extension="mp4", progressive=True).order_by("resolution").asc().first()
                if not stream:
                    raise ValueError("No suitable MP4 stream found.")
                print(f"⬇️ Downloading '{yt.title}' ({stream.resolution}) using pytubefix...")
                if config.cache.enabled:
                    filename = f"{video_id}.mp4"
                else:
                    filename = stream.default_filename
                    filename = "".join([c for c in filename if c.isalpha() or c.isdigit() or c == ' ' or c == '.']).rstrip()
                file_path = stream.download(output_path=output_dir, filename=filename)
                return file_path
            except Exception as e:
                is_bot = "bot" in str(e).lower() or "BotDetection" in type(e).__name__
                if is_bot and attempt == 0:
                    print(f"🔑 Bot detected — regenerating PO token and retrying...")
                    invalidate_po_token()
                    continue
                print(f"❌ pytubefix download failed: {e}")
    return None


def download_video_file(video_url: str) -> str:
    """
    Downloads the video file (low resolution for efficiency) using either yt-dlp or pytubefix.
    Returns the absolute path to the downloaded file.
    """
    config = get_config()
    cache_manager.refresh()
    
    # 1. Check Cache
    if config.cache.enabled and cache_manager.has_video(video_url):
        path = cache_manager.get_video_path(video_url)
        if _is_usable_media_file(path):
            print(f"📦 Cache hit for video file: {path}")
            download_stats.record("video", "cache_hit")
            return path
        _drop_invalid_cache_file(path)

    # 2. Determine output directory
    if config.cache.enabled:
        output_dir = str(cache_manager.video_dir)
    else:
        output_dir = "data/temp/video"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    video_id = extract_youtube_id(video_url)

    # 3. Download — try configured downloader first, fall back to the other
    downloader = config.watcher.video_downloader
    order = ["pytubefix", "ytdlp"] if downloader == "pytubefix" else ["ytdlp", "pytubefix"]

    for method in order:
        result = _try_download(method, video_url, video_id, output_dir, config)
        if result == "skip":
            return None
        if result:
            download_stats.record("video", "download_ok")
            return result
        # first method failed, try fallback
        if method == order[0] and len(order) > 1:
            print(f"🔄 {method} failed, falling back to {order[1]}...")

    download_stats.record("video", "download_fail")
    return None

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=zd_57PFJkNM" # Example

    # Test Transcript
    # docs = fetch_with_ytdlp(video_url)
    docs = fetch_youtube_video_transcript(video_url)
    if docs:
        print(f"✅ Transcript found ({len(docs[0].page_content)} chars)")
        print(docs[0].page_content[:500] + "...\n")  # Print first 500 chars
    else:
        print("❌ No transcript found.")
    
    # Test Download
    # path = download_video_file(video_url)
    # if path:
    #     print(f"✅ Video downloaded to: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-proxies", action="store_true",
                        help="Test proxy pool connectivity and write alive list")
    parser.add_argument("--proxy-file", default="data/ip_pools.txt")
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()

    if args.validate_proxies:
        validate_proxy_pool(args.proxy_file, timeout=args.timeout)
