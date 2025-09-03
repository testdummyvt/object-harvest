from object_harvest.utils.paths import safe_stem


def test_safe_stem_local_path():
    assert safe_stem("/tmp/foo/bar/baz.jpg") == "baz"


def test_safe_stem_url_with_path():
    assert safe_stem("https://example.com/images/photo.png") == "photo.png"


def test_safe_stem_url_no_path():
    # When URL path empty, falls back to netloc
    assert safe_stem("https://example.com") == "example.com"


def test_safe_stem_windows_like():
    # Backslashes replaced
    assert safe_stem("C:\\images\\cat.png").endswith("cat")


def test_safe_stem_separators():
    # Slashes/backslashes turned into underscores; basename without extension preserved
    assert safe_stem("a/b\\c.jpg") == "b_c"
