"""旧库 -> 新库 数据迁移脚本（12/01 配额重置后执行）。

用法：
  python scripts/migrate_old_to_new.py --old "<旧库连接串>" --new "<新库连接串>"

为什么需要它：
  - 旧项目 handpan(winter-boat-44978878) 的出网额度在 2026-10 用尽，
    Neon 直接把该项目的 postgres 实例 SUSPEND，任何连接都被拒：
    "Your account or project has exceeded the quota."
  - 计费额度是「按项目」的，新项目 purple-boat 有独立额度，因此生产已切到新库。
  - 但旧库里存着历史 scores（含 1MB 级的 midi_data_json），必须补迁。
  - 额度按月重置（下一个周期自 2026-12-01 起），重置后旧库即可读，本脚本即可跑通。

安全：
  - 只读旧库；只写新库。
  - 默认只在新库无数据时写入（幂等保护），--force 可强制。
  - 逐行分批，避免一次性拉取撑爆内存/出网。
"""
from __future__ import annotations

import argparse
import sys
from urllib.parse import urlsplit, urlunsplit

import psycopg

TABLE = "scores"


def normalize(url: str) -> str:
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    p = urlsplit(url)
    # 去掉 channel_binding（部分客户端不支持）
    q = "&".join(x for x in p.query.split("&") if not x.startswith("channel_binding="))
    return urlunsplit(("postgresql", p.netloc, p.path, q, ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="旧库连接串")
    ap.add_argument("--new", required=True, help="新库连接串")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    old_u, new_u = normalize(args.old), normalize(args.new)

    with psycopg.connect(new_u, connect_timeout=20) as nc:
        with nc.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {TABLE}")
            have = cur.fetchone()[0]
    if have and not args.force:
        print(f"新库已有 {have} 行，跳过。加 --force 可继续。")
        return 0

    total = 0
    with psycopg.connect(old_u, connect_timeout=20) as oc:
        with oc.cursor() as ocur:
            ocur.execute(f"SELECT count(*) FROM {TABLE}")
            total = ocur.fetchone()[0]
            print("旧库总行数:", total)

            with psycopg.connect(new_u, connect_timeout=20) as nc:
                with nc.cursor() as ncur:
                    offset = 0
                    while True:
                        ocur.execute(
                            f"SELECT id, user_id, original_file_url, midi_data_json, pdf_url,"
                            f" status, error_message, title, source_type, created_at, updated_at"
                            f" FROM {TABLE} ORDER BY id LIMIT %s OFFSET %s",
                            (args.batch, offset),
                        )
                        rows = ocur.fetchall()
                        if not rows:
                            break
                        ncur.executemany(
                            f"INSERT INTO {TABLE} (id, user_id, original_file_url, midi_data_json,"
                            f" pdf_url, status, error_message, title, source_type, created_at, updated_at)"
                            f" VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                            f" ON CONFLICT (id) DO NOTHING",
                            rows,
                        )
                        nc.commit()
                        offset += len(rows)
                        print(f"  已迁 {offset}/{total}")
                        if len(rows) < args.batch:
                            break
    print("迁移完成，共", offset, "行")
    return 0


if __name__ == "__main__":
    sys.exit(main())
