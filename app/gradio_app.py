# Giao diện Phase 1 (Gradio) - chạy từ thư mục gốc repo:
#     python -m app.gradio_app        # mở http://127.0.0.1:7860
#
# Tab 1 "Sửa seed": chọn tile -> click vào cây -> SAM đề xuất mask ->
#                    Chấp nhận / Hoàn tác -> Lưu COCO (bước sửa tay của con người).
# Tab 2 "Pipeline": bấm nút chạy 01-04 và xem log/metric.
import glob
import os
import subprocess

import cv2
import gradio as gr
import numpy as np
import yaml

from app.annotator import SamAnnotator

CFG = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
CKPT = "weights/sam_finetuned_p1.pt"  # dùng nếu đã finetune, không có cũng chạy
ann = SamAnnotator(CFG["sam"], checkpoint=CKPT)
STATE = {"tile": None, "pending": None}


def list_tiles():
    return sorted(glob.glob(os.path.join(CFG["data"]["tiles_dir"], "*.png")))


def load_tile(path):
    if not path:
        return None, "Chưa chọn tile."
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    ann.set_image(img)
    STATE["tile"], STATE["pending"] = path, None
    return img, "Đã nạp ảnh. Click vào 1 cây để SAM đề xuất mask."


def on_click(mode, evt: gr.SelectData):
    x, y = evt.index  # toạ độ pixel nơi bạn click
    positive = mode.startswith("Thêm")
    STATE["pending"] = ann.predict_point(x, y, positive=positive)
    return ann.overlay(pending=STATE["pending"])


def accept():
    if STATE["pending"] is not None:
        ann.add_mask(STATE["pending"])
        STATE["pending"] = None
    return ann.overlay(), f"Đã có {len(ann.masks)} cây."


def undo():
    ann.remove_last()
    STATE["pending"] = None
    return ann.overlay(), f"Đã có {len(ann.masks)} cây."


def save():
    if not STATE["tile"]:
        return "Chưa nạp ảnh."
    name = os.path.basename(STATE["tile"])
    out = os.path.join(
        CFG["data"]["masks_dir"], "corrected", name.rsplit(".", 1)[0] + ".json"
    )
    n = ann.export_coco(name, ann.image.shape[0], ann.image.shape[1], out)
    return f"Đã lưu {n} cây -> {out}"


def run_step(script):
    p = subprocess.run(
        ["python", f"scripts/{script}"], capture_output=True, text=True
    )
    return (p.stdout + "\n" + p.stderr)[-4000:]


with gr.Blocks(title="coffee-seg | Phase 1") as demo:
    gr.Markdown("# coffee-seg - Phase 1 GUI")
    with gr.Tab("Sửa seed (interactive SAM)"):
        with gr.Row():
            tile_dd = gr.Dropdown(list_tiles(), label="Chọn tile")
            mode = gr.Radio(
                ["Thêm (foreground)", "Loại (background)"],
                value="Thêm (foreground)",
                label="Chế độ click",
            )
        img = gr.Image(label="Click vào cây", interactive=True)
        status = gr.Textbox(label="Trạng thái")
        with gr.Row():
            gr.Button("✔ Chấp nhận mask").click(accept, None, [img, status])
            gr.Button("↩ Hoàn tác").click(undo, None, [img, status])
            gr.Button("💾 Lưu COCO").click(save, None, status)
        tile_dd.change(load_tile, tile_dd, [img, status])
        img.select(on_click, mode, img)
    with gr.Tab("Chạy pipeline & kết quả"):
        log = gr.Textbox(label="Log", lines=18)
        with gr.Row():
            gr.Button("01 Cắt tile").click(
                lambda: run_step("01_make_tiles.py"), None, log
            )
            gr.Button("02 Sinh seed").click(
                lambda: run_step("02_gen_seed.py"), None, log
            )
            gr.Button("03 Finetune").click(
                lambda: run_step("03_finetune_sam.py"), None, log
            )
            gr.Button("04 Infer + Eval").click(
                lambda: run_step("04_infer_eval.py"), None, log
            )

if __name__ == "__main__":
    demo.launch()