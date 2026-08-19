"""Extra tests for bounding-box quality (co-location, punctuation, no giant boxes).

These guard the fix for a real defect found by running the app: a name whose parts
appear far apart on the page was merged into a box spanning the whole page.
"""

from unittest.mock import patch


def _run(names, word_boxes):
    with (
        patch("app.services.bbox_service.extract_names", return_value=names),
        patch(
            "app.services.bbox_service.get_word_bounding_boxes",
            return_value=word_boxes,
        ),
    ):
        from app.services.bbox_service import find_name_bounding_boxes

        return find_name_bounding_boxes("test.pdf", "dummy")


def test_matches_despite_ocr_trailing_punctuation():
    """OCR yields 'Thompson,' with a comma; it should still match 'Thompson'."""
    boxes = [
        {"word": "Margaret", "page": 0, "x": 100, "y": 50, "width": 70, "height": 12},
        {"word": "Thompson,", "page": 0, "x": 175, "y": 50, "width": 80, "height": 12},
    ]
    result = _run(["Margaret Thompson"], boxes)
    assert len(result) == 1
    assert result[0]["name"] == "Margaret Thompson"
    assert result[0]["height"] <= 24  # about one line, not page-spanning


def test_parts_on_different_lines_do_not_merge():
    """A 'Margaret' at the top and a 'Thompson' at the bottom must NOT form a box."""
    boxes = [
        {"word": "Margaret", "page": 0, "x": 100, "y": 50, "width": 70, "height": 12},
        {"word": "Thompson", "page": 0, "x": 100, "y": 520, "width": 80, "height": 12},
    ]
    result = _run(["Margaret Thompson"], boxes)
    assert result == [], "Parts on different lines should not merge into a giant box"


def test_prefers_colocated_occurrence_over_distant_one():
    """With both a co-located pair and a stray distant word, pick the co-located one."""
    boxes = [
        {"word": "Margaret", "page": 0, "x": 100, "y": 50, "width": 70, "height": 12},
        {"word": "Thompson", "page": 0, "x": 175, "y": 50, "width": 80, "height": 12},
        {"word": "Thompson", "page": 0, "x": 100, "y": 700, "width": 80, "height": 12},
    ]
    result = _run(["Margaret Thompson"], boxes)
    assert len(result) == 1
    # Height stays one line: it merged the top pair, not top-Margaret with bottom-Thompson.
    assert result[0]["height"] <= 24
    assert result[0]["y"] == 50


def test_duplicate_colocated_names_map_to_successive_occurrences():
    """Two real occurrences on different lines yield two separate, tight boxes."""
    boxes = [
        {"word": "John", "page": 0, "x": 100, "y": 50, "width": 40, "height": 12},
        {"word": "Smith", "page": 0, "x": 145, "y": 50, "width": 50, "height": 12},
        {"word": "John", "page": 0, "x": 100, "y": 200, "width": 40, "height": 12},
        {"word": "Smith", "page": 0, "x": 145, "y": 200, "width": 50, "height": 12},
    ]
    result = _run(["John Smith", "John Smith"], boxes)
    assert len(result) == 2
    ys = sorted(b["y"] for b in result)
    assert ys == [50, 200]
    assert all(b["height"] <= 24 for b in result)
