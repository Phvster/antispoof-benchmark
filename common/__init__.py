import sys as _sys

# Log/--help có tiếng Việt: console Windows (cp1252) sẽ lỗi UnicodeEncodeError -> ép UTF-8.
for _s in (_sys.stdout, _sys.stderr):
    if _s is not None and hasattr(_s, "reconfigure") and (_s.encoding or "").lower() != "utf-8":
        try:
            _s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
