"""Machine-local Claude automation token storage, separate from interactive login."""
from __future__ import annotations

import getpass
import os
from pathlib import Path
import stat
import tempfile


def token_path() -> Path:
    return Path.home() / '.config/neuro-autoresearch/claude-oauth-token'


def claude_token() -> str:
    """Prefer an explicit environment token; otherwise read the private saved token."""
    token = os.environ.get('CLAUDE_CODE_OAUTH_TOKEN', '').strip()
    if token:
        return token
    path = token_path()
    try:
        with path.open() as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise ValueError(f'Claude token file must be owned by you with mode 600: {path}')
            token = stream.read().strip()
    except FileNotFoundError:
        token = ''
    if not token:
        raise ValueError('Claude automation requires a token. Run claude setup-token, then '
                         'python -m search.auth to save it, or export CLAUDE_CODE_OAUTH_TOKEN.')
    return token


def save_token(token: str) -> Path:
    token = token.strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError('Paste a single nonempty token, without embedded whitespace.')
    path = token_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    fd, temporary = tempfile.mkstemp(prefix='.claude-token-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(token + '\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def main() -> None:
    import sys
    if not sys.stdin.isatty():
        raise SystemExit('Run python -m search.auth in an interactive terminal for a hidden token prompt.')
    try:
        path = save_token(getpass.getpass('Paste Claude automation token (hidden): '))
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from None
    print(f'Saved Claude automation token to {path} (owner-only access).')
    print('Claude search launches will load it automatically; no export is needed.')


if __name__ == '__main__':
    main()
