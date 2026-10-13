# app/flower_rounds.py — Chọn tán cho một vòng gán nhãn hoa.
#
# Luật (configs/flower_rounds.yaml, khối `pick` của từng vòng):
#   crowns_per_field   số tán mỗi ruộng
#   images_per_field   số ẢNH mỗi ruộng; có khoá này thì bỏ qua crowns_per_field
#                      (dùng với crowns_per_image: all để gán kín từng ảnh)
#   crowns_per_image   tối đa bao nhiêu tán trong một ảnh ("all" = mọi tán)
#   skip_border        bỏ tán chạm mép ảnh (tán bị cắt dở)
#   buffer_frames      ảnh cách ảnh của vòng phía bên kia ≤ ngần ấy khung trong
#                      cùng đường bay thì không lấy (khung liền kề chồng ~45%,
#                      tức chụp trùng cây: bộ chấm và bộ học không được chung cây)
#   fields             chỉ lấy các ruộng này
#   seed               cùng seed thì ra đúng danh sách cũ
#
# Một ảnh chỉ thuộc một vòng. Ảnh trong mỗi ruộng được lấy rải đều các đường
# bay (xen kẽ từng đường), trong mỗi ảnh lấy ngẫu nhiên các tán đủ điều kiện.
import random
import re

DJI = re.compile(r"DJI_\d{14}_(\d{4})_")


def frame_no(name, fallback=0):
    """Số thứ tự khung trong tên ảnh DJI; ảnh không theo mẫu dùng `fallback`."""
    m = DJI.search(name)
    return int(m.group(1)) if m else fallback


def touches_border(poly, size, margin=1):
    h, w = size
    xs, ys = poly[0::2], poly[1::2]
    return min(xs) <= margin or min(ys) <= margin or max(xs) >= w - 1 - margin or max(ys) >= h - 1 - margin


def near_any(img, others, k):
    """Ảnh `img` cách một ảnh trong `others` (cùng đường bay) ≤ k khung."""
    return any(abs(img["seq"] - s) <= k for s in others.get(img["flight"], ()))


def build(rcfg, images, taken, block, buffer):
    """Chọn tán cho vòng `rcfg`.

    images: [{name, flat, field, flight, seq, size=(h, w), crowns=[poly, ...]}]
    taken:  ảnh (flat) đã thuộc một vòng nào đó — không lấy lại
    block:  ảnh của phía bên kia (bộ vàng nếu đang dựng vòng học, và ngược lại)
    buffer: số khung đệm quanh các ảnh `block`
    """
    pick = dict(rcfg.get("pick") or {})
    per_field = int(pick.get("crowns_per_field", 5))
    img_field = pick.get("images_per_field")
    img_field = None if img_field is None else int(img_field)
    per_img = pick.get("crowns_per_image", 3)
    per_img = None if str(per_img).lower() == "all" else int(per_img)
    skip_border = bool(pick.get("skip_border", False))
    seed = pick.get("seed", 0)
    fields = set(pick.get("fields") or [])
    by_flight = {}
    for im in images:
        if im["flat"] in block:
            by_flight.setdefault(im["flight"], []).append(im["seq"])
    cands = [im for im in images
             if im["flat"] not in taken and im["crowns"]
             and (not fields or im["field"] in fields)
             and not (buffer and near_any(im, by_flight, buffer))]
    items = []
    for field in sorted({im["field"] for im in cands}):
        rng = random.Random("%s:%s" % (seed, field))
        flights = {}
        for im in sorted((c for c in cands if c["field"] == field), key=lambda c: c["name"]):
            flights.setdefault(im["flight"], []).append(im)
        names = sorted(flights)
        rng.shuffle(names)
        for f in names:
            rng.shuffle(flights[f])
        order, i = [], 0
        while any(flights[f] for f in names):          # xen kẽ từng đường bay
            f = names[i % len(names)]
            if flights[f]:
                order.append(flights[f].pop(0))
            i += 1
        got = n_img = 0
        for im in order:
            if (n_img >= img_field) if img_field is not None else (got >= per_field):
                break
            ks = [k for k, p in enumerate(im["crowns"])
                  if not (skip_border and touches_border(p, im["size"]))]
            if not ks:
                continue
            rng.shuffle(ks)
            if per_img is None:
                take = ks
            elif img_field is not None:
                take = ks[:per_img]
            else:
                take = ks[:min(per_img, per_field - got)]
            for k in sorted(take):
                xs, ys = im["crowns"][k][0::2], im["crowns"][k][1::2]
                items.append(dict(image=im["flat"], name=im["name"], crown=k, field=field,
                                  flight=im["flight"], seq=im["seq"],
                                  bbox=[int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1]))
            got += len(take)
            n_img += 1
    return items


def double_subset(items, n):
    """n tán cho người gán thứ hai: lấy xen kẽ theo ruộng để ruộng nào cũng có."""
    by_field = {}
    for it in items:
        by_field.setdefault(it["field"], []).append(it)
    out, i = [], 0
    fields = sorted(by_field)
    while len(out) < n and any(by_field[f] for f in fields):
        f = fields[i % len(fields)]
        if by_field[f]:
            out.append(by_field[f].pop(0))
        i += 1
    return out


def agreement(pairs):
    """IoU và F1 của lớp hoa, cộng dồn pixel qua các cặp (a, b) uint8 0/1/255.
    Pixel là 255 ở bất kỳ bên nào thì bỏ qua."""
    tp = fp = fn = 0
    for a, b in pairs:
        ok = (a != 255) & (b != 255)
        A, B = (a == 1) & ok, (b == 1) & ok
        tp += int((A & B).sum())
        fp += int((B & ~A).sum())
        fn += int((A & ~B).sum())
    den = tp + fp + fn
    return dict(iou=(tp / den) if den else None, f1=(2 * tp / (2 * tp + fp + fn)) if den else None,
                tp=tp, fp=fp, fn=fn)


def review_queue(items, unc, fix_top=0.30, check_random=0.10, seed=0):
    """Hàng đợi của vòng duyệt: trong TỪNG ruộng, `fix_top` phần tán kém chắc
    nhất (unc cao nhất; tán thiếu số đo coi như kém chắc nhất) rồi `check_random`
    phần tán ngẫu nhiên trong số còn lại. Xếp theo ruộng xen kẽ, phần kém chắc
    trước phần ngẫu nhiên. Trả [(item, lý do, unc)]; tán không vào hàng đợi sẽ
    được nhận tự động lúc chốt vòng.

    Xếp hạng trong từng ruộng chứ không chung cả vòng: thứ dễ nhầm khác nhau
    theo ruộng, xếp chung thì ruộng "khó" nuốt hết lượt duyệt."""
    by = {}
    for it in items:
        by.setdefault(it["field"], []).append(it)
    fixes, checks = {}, {}
    for f, lst in by.items():
        key = lambda it: unc.get((it["image"], int(it["crown"])), float("inf"))  # noqa: E731
        lst = sorted(lst, key=lambda it: (-key(it), it["image"], int(it["crown"])))
        n_fix = int(round(len(lst) * fix_top))
        n_chk = int(round(len(lst) * check_random))
        rest = lst[n_fix:]
        rng = random.Random("%s:%s" % (seed, f))
        fixes[f] = [(it, "uncertain", key(it)) for it in lst[:n_fix]]
        checks[f] = [(it, "random", key(it)) for it in rng.sample(rest, min(n_chk, len(rest)))]

    def weave(groups):
        out, i, fs_ = [], 0, sorted(groups)
        while any(groups[f] for f in fs_):
            f = fs_[i % len(fs_)]
            if groups[f]:
                out.append(groups[f].pop(0))
            i += 1
        return out
    return weave(fixes) + weave(checks)
