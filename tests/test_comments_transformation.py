from src.transformations import CommentsTransformation

def test_transformation_name():
    assert CommentsTransformation().name == "comments"

def test_removes_single_line_comment():
    source = "int x = 1; // comment\nint y = 2;\n"

    result = CommentsTransformation().transform(source)

    assert "// comment" not in result.transformed_text
    assert "int x = 1;" in result.transformed_text
    assert "int y = 2;" in result.transformed_text

def test_removes_block_comment():
    source = "int x /* comment */ = 1;\n"

    result = CommentsTransformation().transform(source)

    assert "comment" not in result.transformed_text
    assert "int x" in result.transformed_text
    assert "= 1;" in result.transformed_text


def test_preserves_newlines_inside_block_comment():
    source = (
        "int x = 1; /* first line\n"
        "second line\n"
        "third line */ int y = 2;\n"
    )

    result = CommentsTransformation().transform(source)

    assert result.transformed_text.count("\n") == source.count("\n")


def test_preserves_string_containing_comment_markers():
    source = (
        'const char* a = "// not comment";\n'
        'const char* b = "/* not comment */";\n'
    )

    result = CommentsTransformation().transform(source)

    assert result.transformed_text == source


def test_preserves_character_literals():
    source = (
        "char a = '/';\n"
        "char b = '\\'';\n"
        "char c = '\\\\';\n"
    )

    result = CommentsTransformation().transform(source)

    assert result.transformed_text == source


def test_preserves_raw_string_literal():
    source = (
        'const char* text = R"tag(\n'
        '// not a comment\n'
        '/* not a comment */\n'
        ')tag";\n'
    )

    result = CommentsTransformation().transform(source)

    assert result.transformed_text == source


def test_comment_removal_does_not_merge_tokens():
    source = "int/**/value = 1;\n"

    result = CommentsTransformation().transform(source)

    assert "int    value" in result.transformed_text


def test_removes_comment_at_end_of_file():
    source = "int x = 1; // trailing comment"

    result = CommentsTransformation().transform(source)

    assert "// trailing comment" not in result.transformed_text
    assert result.transformed_text.startswith("int x = 1;")


def test_unterminated_block_comment_fails_closed():
    source = "int x = 1; /* unfinished"

    result = CommentsTransformation().transform(source)

    assert result.transformed_text == source
    assert result.changed is False
    assert result.config["skipped_reason"] == (
        "unterminated_block_comment"
    )


def test_line_splicing_fails_closed():
    source = (
        "int x = 1; \\\n"
        "// continuation comment\n"
    )

    result = CommentsTransformation().transform(source)

    assert result.transformed_text == source
    assert result.changed is False
    assert result.config["skipped_reason"] == (
        "line_splicing_detected"
    )


def test_no_comments_returns_unchanged_source():
    source = "int main() {\n    return 0;\n}\n"

    result = CommentsTransformation().transform(source)

    assert result.transformed_text == source
    assert result.changed is False


def test_deterministic_transformation():
    source = "int x = 1; // comment\n"

    transformation = CommentsTransformation()

    first = transformation.transform(source, seed=42)
    second = transformation.transform(source, seed=42)

    assert first.transformed_text == second.transformed_text


def test_metadata():
    source = "int x = 1; // comment\n"

    result = CommentsTransformation().transform(source, seed=42)

    assert result.transformation_type == "comments"
    assert result.config["strategy"] == "lexical_comment_removal"
    assert result.config["preserve_newlines"] is True
    assert result.config["seed"] == 42

def test_malformed_prefixed_raw_string_fails_closed():
    source = 'auto s = u8R"invalid delimiter(/* comment */)invalid delimiter";'

    result = CommentsTransformation().transform(source)

    assert result.transformed_text == source
    assert result.changed is False
    assert result.config["skipped_reason"] == (
        "malformed_raw_string_literal"
    )


def test_prefixed_raw_string_without_open_paren_fails_closed():
    source = 'auto s = u8R"delimiter_without_paren /* comment */;'

    result = CommentsTransformation().transform(source)

    assert result.transformed_text == source
    assert result.changed is False
    assert result.config["skipped_reason"] == (
        "malformed_raw_string_literal"
    )