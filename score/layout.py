"""谱面排版参数 —— 单一事实源。

本文件的数值来自对标准样张（双击练习3）的逐像素实测，单位 mm / pt：
    页面 210×297，左右边距各 10mm → 版心 190mm
    每页 6 行，行距（中线到中线）39.5mm，行高（长竖线）20.1mm，短竖线 10mm
    每小节 2 个 → 每行 2 小节，每小节 12 格（格宽 7.92mm）
    竖线交替：偶数格为长线（深），奇数格为短线（浅）
    音符居中在格内；橙字在中线上方，蓝字在中线下方
"""
from __future__ import annotations

PAGE = {
    "w": 210.0, "h": 297.0,
    "ml": 10.0, "mr": 10.0, "mt": 16.0, "mb": 12.0,
}

SYSTEM = {
    "per_page": 6,
    "height": 20.1,        # 长竖线长度
    "short_height": 10.0,  # 短竖线长度
    "pitch": 39.5,         # 行距（中线到中线）
    "first_mid_y": 59.8,   # 第一行中线距页顶
    "bars": 2,
    "cells": 12,           # 每小节格数（可由 score_json.grid.cells 覆盖）
}

NOTE = {
    "upper_size": 11.5,
    "lower_size": 10.5,
    "gap_mm": 3.2,         # 音符与中线的间距（上/中/下三档，拉开才分得清）
    "tap_size": 9.0,
}

LINE = {
    # 三级竖线（用户定义）：粗=大节（每 16 格），长=小节（每 4 格），短=格
    "bar":   {"lw": 1.4, "color": "#0A0A0A", "h": 20.1},   # 大节
    "long":  {"lw": 0.7, "color": "#8C8C8C", "h": 16.5},   # 小节
    "short": {"lw": 0.5, "color": "#B0B0B0", "h": 10.0},   # 格
    "mid":   {"lw": 0.5, "color": "#B8B8B8", "h": 0.0},
}

CELLS_PER_SUBBAR = 4      # 每小节的格数（4/4：4 个 16 分音符 = 1 拍）

COLOR = {
    "blue": "#34589E",     # 音位 1、2
    "orange": "#E8912D",   # 音位 3–9
    "tap": "#9AA5B4",      # 无音高敲击 ×
    "title": "#111111",
    "meta": "#6B6B6B",
    "index": "#5A5A5A",
    "approx": "#B9C5D6",
}

TEXT = {
    "title_size": 18.0,
    "tempo_size": 11.0,
    "bar_number_size": 7.0,
    "footer_size": 6.0,
}
