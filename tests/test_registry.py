"""Sổ đăng ký: tên trong config -> lớp thực thi.

Đây là cơ chế khiến scripts/train.py không cần biết YOLO tồn tại, nên nó phải
thất bại rõ ràng khi tra hụt — im lặng trả về None sẽ đẩy lỗi xuống tận lúc
huấn luyện.
"""

import pytest

from canopyseg import registry


@pytest.fixture
def kind(request):
    """Một "kind" RIÊNG cho mỗi test.

    Sổ đăng ký là trạng thái toàn cục ở cấp module, nên dùng chung một tên sẽ
    khiến test này thấy đăng ký của test kia và thứ tự chạy quyết định kết quả.
    Lấy tên test làm namespace là cách rẻ nhất để cô lập.
    """
    return f"thunghiem_{request.node.name}"


def test_register_then_resolve(kind):
    @registry.register(kind, "alpha")
    class Alpha:
        pass

    assert registry.resolve(kind, "alpha") is Alpha


def test_available_is_sorted(kind):
    for n in ("zeta", "alpha", "mu"):
        registry.register(kind, n)(type(n, (), {}))
    assert registry.available(kind) == ["alpha", "mu", "zeta"]


def test_registering_the_same_name_twice_raises(kind):
    registry.register(kind, "dup")(type("A", (), {}))
    with pytest.raises(ValueError, match="đã được đăng ký"):
        registry.register(kind, "dup")(type("B", (), {}))


def test_re_registering_the_identical_class_is_allowed(kind):
    """Import lại cùng một module không được biến thành lỗi."""
    cls = type("Same", (), {})
    registry.register(kind, "same")(cls)
    registry.register(kind, "same")(cls)
    assert registry.resolve(kind, "same") is cls


def test_unknown_name_lists_what_exists(kind):
    registry.register(kind, "co_that")(type("A", (), {}))
    with pytest.raises(KeyError) as e:
        registry.resolve(kind, "khong-co")
    assert "co_that" in str(e.value)


def test_unknown_kind_hints_at_the_import(kind):
    with pytest.raises(KeyError, match="chưa nạp"):
        registry.resolve("kind_chua_ton_tai", "x")


def test_available_of_unknown_kind_is_empty():
    assert registry.available("kind_chua_ton_tai") == []


def test_decorator_returns_the_class_unchanged(kind):
    class Orig:
        attr = 1

    assert registry.register(kind, "orig")(Orig) is Orig
    assert registry.resolve(kind, "orig").attr == 1


# ------------------------------------------------- các đăng ký thật của dự án
def test_project_trainers_and_models_are_registered():
    import canopyseg.models  # noqa: F401
    import canopyseg.training  # noqa: F401

    # Bốn model của bảng benchmark nằm trong benchmark/<model>_<người>/cofseg/,
    # mỗi thư mục một sổ đăng ký riêng. Ở gốc chỉ còn đường chạy nhanh và
    # nhánh promptable.
    assert {"yolo", "maskrcnn"} <= set(registry.available("trainer"))
    assert {"yolo_seg", "maskrcnn", "coco_predictions", "two_stage", "refine",
            "sam2_auto", "sam2_oracle"} <= set(registry.available("model"))
