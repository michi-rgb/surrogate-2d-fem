"""同一寸法格子でFEMとサロゲートの実時間を比較し、日本語Markdownを作る。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.base import clone

from beam_surrogate import Settings, feasible_intervals, save_json, solve_fem, write_csv


def extract_ranges(lengths, heights, mask):
    """両方式で同じ範囲抽出処理を使う。戻り値は長さごとの高さ区間。"""
    return [[float(length), lo, hi, count]
            for length, row in zip(lengths, mask)
            for lo, hi, count in feasible_intervals(heights, row)]


def select_candidates(x, mask, count):
    """配布プログラムと同じ規則で、最小体積候補と下限高さ付近を選ぶ。"""
    if not mask.any() or count == 0:
        return []
    grid = mask.shape[1]
    accepted = np.flatnonzero(mask.ravel())
    best = int(accepted[np.argmin(x[accepted, 0]*x[accepted, 1])])
    selected = [best]
    boundary = [i*grid + np.flatnonzero(row)[0] for i, row in enumerate(mask) if row.any()]
    for pos in np.linspace(0, len(boundary)-1, min(count, len(boundary)), dtype=int):
        if boundary[pos] not in selected and len(selected) < count:
            selected.append(int(boundary[pos]))
    return selected


def stats(values):
    return {"median": float(np.median(values)), "min": float(min(values)), "max": float(max(values))}


def write_report(path, result):
    """手入力の時間を混ぜず、保存した計測結果だけからレポートを生成する。"""
    r = result
    timings = r["timings"]
    med = lambda key: timings[key]["median"]
    fem = med("fem_search_s")
    surrogate = med("surrogate_search_s")
    checked = med("surrogate_with_verification_s")
    initial = med("initialization_s")
    loaded = med("load_and_search_s")
    startup = initial + loaded
    startup_checked = initial + med("load_s") + checked
    recovery = max(1, math.ceil(initial/(fem-loaded))) if fem > loaded else None
    recovery_checked = max(1, math.ceil(initial/(fem-med("load_s")-checked))) if fem > med("load_s")+checked else None
    timing_rows = []
    labels = {
        "data_generation_s": "教師データ800点をFEMで再生成",
        "fit_s": "Extra Treesの学習（640点）",
        "save_s": "モデル保存",
        "initialization_s": "初期構築合計（上記3工程）",
        "load_s": "保存済みモデル読込",
        "fem_search_s": "全格子をFEM解析＋合否判定＋範囲抽出",
        "surrogate_search_s": "サロゲート一括予測＋合否判定＋範囲抽出",
        "load_and_search_s": "モデル読込＋サロゲート探索",
        "verification_s": "候補選定＋候補のFEM再確認",
        "surrogate_with_verification_s": "サロゲート探索＋候補のFEM再確認（読込除外）",
        "cached_fem_search_s": "既計算FEMを荷重比例で再利用＋判定＋範囲抽出",
    }
    for key, label in labels.items():
        s = timings[key]
        timing_rows.append(f"| {label} | {s['median']:.6f} | {s['min']:.6f} | {s['max']:.6f} |")
    accuracy_rows = []
    for a in r["accuracy"]:
        accuracy_rows.append(f"| {a['factor']:.2f} | {a['predicted_feasible']} | {a['false_accept']} | "
                             f"{a['false_reject']} | {a['agreement_percent']:.3f}% | "
                             f"{a['false_accept_fraction_of_predictions_percent']:.3f}% |")
    comparisons = []
    for count in (1, 5, 10, 100):
        comparisons.append(f"| {count} | {count*fem:.3f} | {initial+count*loaded:.3f} | "
                           f"{initial+count*(med('load_s')+checked):.3f} |")
    s = r["settings"]
    env = r["environment"]
    text = f"""# FEMとサロゲートによるはり寸法範囲探索の時間比較

計測日時：{r['measured_at']}。既存の配布コードを使い、同一PC・同じ{r['grid']}×{r['grid']}格子（{r['points']:,}通り）を実計算した結果です。

## 結果

学習済みサロゲートの探索は **{surrogate:.4f}秒**、全格子のFEM探索は **{fem:.3f}秒**で、計算部分は **約{fem/surrogate:,.0f}倍**の速度差でした。モデル読込を含めるとサロゲートは{loaded:.4f}秒、FEMに対して約{fem/loaded:,.0f}倍です。

候補{r['verification_count']}点をFEMで再確認する運用では、探索と再確認の合計は **{checked:.4f}秒**（モデル読込除外）、全格子FEMに対して約{fem/checked:,.0f}倍でした。ただし確認対象は一部だけであり、全格子FEMと同じ精度・保証範囲ではありません。

教師データ生成・学習・モデル保存の初期構築は **{initial:.3f}秒**です。初期構築と読込を含めた初回探索は{startup:.3f}秒、FEM再確認を含めると{startup_checked:.3f}秒になります。今回の細かい格子では、全点FEMを毎回やり直す比較において、初期構築費用は探索 **{recovery}回目**で回収する計算です（再確認付きでも{recovery_checked}回目）。

一方、この問題は線形弾性なので、**同じ寸法格子のFEM値を保存すれば、以後は荷重比例と上限判定だけで再探索できます**。この再利用方式は{med('cached_fem_search_s'):.6f}秒でした。「毎回全点をFEMで再計算する場合」の速度差を、そのままあらゆるFEM運用に一般化することはできません。

## 比較条件

| 項目 | 条件 |
|---|---|
| 荷重 | {r['force_N']:g} N、上面中央の下向き集中荷重 |
| たわみ上限 | {r['limit_mm']:g} mm、中央断面・高さ中央の鉛直変位 |
| 寸法 | L={s['length_min']:g}～{s['length_max']:g} mm、H={s['height_min']:g}～{s['height_max']:g} mm |
| 格子 | 各{r['grid']}点、L刻み{r['length_step_mm']:g} mm、H刻み{r['height_step_mm']:g} mm |
| 材料・面外厚さ | E={s['young']:g} MPa、ν={s['poisson']:g}、t={s['thickness']:g} mm |
| FEM | 2次元平面応力Q4、{s['nx']}×{s['ny']}要素、{2*(s['nx']+1)*(s['ny']+1):,}自由度（拘束前） |
| 支持 | 左下ピン、右下ローラー |
| サロゲート | ExtraTreesRegressor、300本、n_jobs=-1、乱数シード{s['seed']} |
| 教師データ | {r['samples']}点を独立FEM、640点学習・160点テスト。既存CSVと同一条件を再計算 |
| 余裕係数 | 主な時間比較は{r['factor']:g}。予測たわみに係数を掛けて合否判定 |
| 反復 | 各工程{r['repeats']}回。代表値は中央値、最小・最大も表示 |

## 時間の実測値

単位はすべて秒です。各行は対応する処理ブロックの実測です。合計時間も反復ごとに合算してから中央値を取るため、各行の中央値の和と完全には一致しない場合があります。

| 処理 | 中央値 [s] | 最小 [s] | 最大 [s] |
|---|---:|---:|---:|
{chr(10).join(timing_rows)}

FEM探索は各寸法でメッシュ生成、剛性組立、境界条件設定、疎行列求解を実施し、たわみを取得します。サロゲート探索は同じ全格子を一括予測します。両方式とも、たわみから合否を判定し、長さごとの高さ区間を抽出する時間まで含めました。候補のFEM再確認は既存プログラムと同じ選定規則を使い、既計算の全点FEM値を流用せず改めて解いて計時しました。

Python起動・ライブラリーimport・依存ライブラリーのインストール・グリッド入力配列作成・結果CSV/画像の保存・レポート生成は計時外です。モデル保存と読込は別途実測しています。したがって、これはコマンドを起動して終了するまでの総待ち時間ではありません。初期構築も計算とモデル保存の合計で、評価図等の作成は含みません。

## 速さと引き換えになる判定誤差

ここでの正解は「同じメッシュのFEM」です。実物やメッシュ収束後の厳密解を正解にした評価ではありません。

全格子のFEM合格点は **{r['fem_feasible']:,}点 / {r['points']:,}点**でした。サロゲートのたわみ予測は全格子に対してMAPE **{r['grid_mape_percent']:.3f}%**、最大相対誤差 **{r['grid_max_relative_error_percent']:.3f}%**です。格子評価は学習に使っていませんが、今回の荷重1条件の評価であり、荷重全域の独立試験を表すものではありません。

| 予測の余裕係数 | 予測合格点 | 見逃し：予測合格・FEM不合格 | 過剰棄却：予測不合格・FEM合格 | 全点の合否一致率 | 見逃し数 / 予測合格点 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(accuracy_rows)}

係数1.00の行は純粋な合否予測の比較です。係数{r['factor']:g}の行は教材の運用条件であり、余裕を取ることによる棄却も含みます。多数の点が合格する問題では一致率が高く見えやすいため、見逃し件数を併せて判断してください。

候補{r['verification_count']}点の再確認では、**{r['verification_failures']}点がFEMで上限を超過**しました。確認した一部の点が合格しても、予測した高さ区間全体の合格を保証しません。詳細は [全点比較CSV](benchmark_output/grid_comparison.csv) と [再確認候補CSV](benchmark_output/verified_candidates.csv) に保存しています。

## 初期構築コストと繰り返し利用

以下は各工程の実測中央値を使った**累積時間の試算**です。1、5、10、100回の処理全体をそれぞれ実行した測定ではありません。各回で全点FEMをやり直す方式と、初期構築を1回だけ行い毎回モデルを読込む方式を比較しています。

| 探索回数 | 毎回全点FEM [s] | 初期構築＋読込＋予測 [s] | 初期構築＋読込＋予測＋候補FEM確認 [s] |
|---|---:|---:|---:|
{chr(10).join(comparisons)}

回収回数は `ceil(初期構築時間 / (全点FEM時間 − 1回のサロゲート運用時間))` で算出しました。寸法格子、要素数、教師点数、モデルが変われば回収回数も変わります。今回、教師データ{r['samples']}点より探索格子{r['points']:,}点が多いことも、初回から時間を短縮できる理由です。

ただし、この表はFEM結果の保存・再利用を使わない場合の試算です。同じ格子なら、たわみ上限だけの変更は保存値の比較で済み、荷重変更も線形弾性の範囲では `新しいたわみ = 保存たわみ × 新荷重 / 保存時荷重` で求められます。再利用FEMの初回費用は全点FEM探索時間、2回目以降の計算は表の再利用時間程度です。今回はこの荷重比例の成り立ちも、別荷重で独立に解いたFEMとの比較で確認しました（最大相対差 {r['linearity_max_relative_error']:.3e}）。

## 講座での説明と使い分け

- サロゲートは、高価な解析を多数回呼ぶ探索で、学習後の計算を短縮する手段です。初期構築を含めて比較すると費用対効果を説明できます。
- 今回のような線形問題で、同じ格子と荷重・上限だけを変更するなら、FEM値の保存・荷重比例も有力です。サロゲートだけが高速化の方法ではありません。
- 新しい寸法の組合せを多数調べる用途では、サロゲートは再計算を減らせます。ただし境界付近の見逃しがあるため、採用する設計候補はFEMで確認します。
- 本結果は既存の教育用FEMを逐次実行した比較です。FEM側の並列化、求解器の変更、単調性を使った境界探索などとの最良性能比較ではありません。サロゲート側は300本の木を `n_jobs=-1` で並列処理しています。

## 再現方法と保存した証拠

```powershell
python -X utf8 benchmark_surrogate.py --repeats {r['repeats']} --grid {r['grid']} --force {r['force_N']:g} --limit {r['limit_mm']:g}
```

既存の `output/training_data.csv` と、自分で作成した `output/surrogate.joblib` を入力にします。既存の教師点のFEMを再実行して時間を測り、同じ学習/テスト分割でモデルを再学習します。モデルファイルの出所を確認してください。

| 保存物 | 内容 |
|---|---|
| [計測スクリプト](benchmark_surrogate.py) | 日本語コメント付きの再実行コード |
| [測定結果JSON](benchmark_output/benchmark_results.json) | 環境、条件、工程別集計、精度、ソースのSHA-256 |
| [反復別の生時間CSV](benchmark_output/timings.csv) | 各反復の実測秒数 |
| [全点比較CSV](benchmark_output/grid_comparison.csv) | 各寸法のFEM値・予測値・合否 |
| [寸法範囲CSV](benchmark_output/dimension_ranges_comparison.csv) | FEM、予測、係数適用後の長さごとの高さ範囲 |

計測環境は {env['os']}、Python {env['python']}、NumPy {env['numpy']}、SciPy {env['scipy']}、scikit-learn {env['sklearn']}。CPU識別情報は `{env['processor']}`、OSが認識する論理CPU数は{env['logical_cpus']}です。ハードウェア、電源設定、他アプリの負荷で時間は変動します。各反復で同一PC上のFEMとサロゲートを順に計測し、同時実行はしていません。FEMとサロゲートを各1回ウォームアップしてから測定しました。モデル読込はOSキャッシュを消去していないため、ディスクの完全なコールド起動時間ではありません。
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("output"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_output"))
    parser.add_argument("--report", type=Path, default=Path("FEM_surrogate_time_comparison.md"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--grid", type=int, default=81)
    parser.add_argument("--force", type=float, default=1000)
    parser.add_argument("--limit", type=float, default=0.5)
    parser.add_argument("--factor", type=float, default=1.15)
    parser.add_argument("--verify", type=int, default=12)
    args = parser.parse_args()
    if args.repeats < 1 or args.grid < 2 or args.verify < 0:
        parser.error("repeats>=1、grid>=2、verify>=0 を指定してください。")
    if not np.isfinite([args.force, args.limit, args.factor]).all() or args.limit <= 0 or args.factor < 1:
        parser.error("上限は正の有限値、係数は1以上の有限値にしてください。")
    bundle = joblib.load(args.source / "surrogate.joblib")
    model = bundle["model"]
    s = Settings(**bundle["settings"])
    s.validate()
    if not s.force_min <= args.force <= s.force_max:
        parser.error("荷重は学習範囲内を指定してください。")
    if type(model).__name__ != "ExtraTreesRegressor" or model.n_estimators != 300:
        parser.error("このレポートは教材既定のExtra Trees 300本を対象にしています。")
    data = np.genfromtxt(args.source / "training_data.csv", delimiter=",", names=True,
                         dtype=None, encoding="utf-8-sig")
    x_data = np.column_stack([data[name] for name in ("length_mm", "height_mm", "force_N")])
    train_mask = data["split"] == "train"
    if len(data) != 800 or train_mask.sum() != 640:
        parser.error("このレポートは教材既定の800点（学習640点）を対象にしています。")
    args.output.mkdir(parents=True, exist_ok=True)
    lengths = np.linspace(s.length_min, s.length_max, args.grid)
    heights = np.linspace(s.height_min, s.height_max, args.grid)
    ll, hh = np.meshgrid(lengths, heights, indexing="ij")
    x = np.column_stack((ll.ravel(), hh.ravel(), np.full(ll.size, args.force)))
    shape = ll.shape
    # import・最初の求解・並列ワーカー初期化の影響を減らす
    solve_fem(800, 80, args.force, s)
    model.predict(np.log(x))
    raw = []
    for repeat in range(args.repeats):
        record = {"repeat": repeat+1}
        start = time.perf_counter()
        targets = np.array([solve_fem(*row, s).deflection for row in x_data])
        record["data_generation_s"] = time.perf_counter()-start
        np.testing.assert_allclose(targets, data["fem_deflection_mm"], rtol=1e-8, atol=1e-10)
        fitted = clone(model)
        start = time.perf_counter()
        fitted.fit(np.log(x_data[train_mask]), np.log(targets[train_mask]))
        record["fit_s"] = time.perf_counter()-start
        start = time.perf_counter()
        joblib.dump({"model": fitted, "settings": bundle["settings"]}, args.output / "benchmark_model.joblib")
        record["save_s"] = time.perf_counter()-start
        record["initialization_s"] = sum(record[k] for k in ("data_generation_s", "fit_s", "save_s"))
        start = time.perf_counter()
        loaded = joblib.load(args.output / "benchmark_model.joblib")["model"]
        record["load_s"] = time.perf_counter()-start
        print(f"反復{repeat+1}/{args.repeats}: 初期構築完了、全{len(x)}点FEMを計測中", flush=True)
        start = time.perf_counter()
        truth = np.empty(len(x))
        for i, row in enumerate(x):
            truth[i] = solve_fem(*row, s).deflection
            if (i+1) % 1500 == 0:
                print(f"  FEM {i+1}/{len(x)}点", flush=True)
        truth_mask = (truth <= args.limit).reshape(shape)
        fem_ranges = extract_ranges(lengths, heights, truth_mask)
        record["fem_search_s"] = time.perf_counter()-start
        start = time.perf_counter()
        predicted = np.exp(loaded.predict(np.log(x)))
        mask = (predicted*args.factor <= args.limit).reshape(shape)
        predicted_ranges = extract_ranges(lengths, heights, mask)
        record["surrogate_search_s"] = time.perf_counter()-start
        record["load_and_search_s"] = record["load_s"] + record["surrogate_search_s"]
        start = time.perf_counter()
        selected = select_candidates(x, mask, args.verify)
        verified = [(index, solve_fem(*x[index], s).deflection) for index in selected]
        record["verification_s"] = time.perf_counter()-start
        record["surrogate_with_verification_s"] = record["surrogate_search_s"] + record["verification_s"]
        # FEMキャッシュを再利用する場合の、荷重1.5倍での探索を実測する
        start = time.perf_counter()
        cached = truth * 1.5
        extract_ranges(lengths, heights, (cached <= args.limit).reshape(shape))
        record["cached_fem_search_s"] = time.perf_counter()-start
        raw.append(record)
        print(f"反復{repeat+1}: FEM={record['fem_search_s']:.3f}秒、予測={record['surrogate_search_s']:.4f}秒", flush=True)
        write_csv(args.output / "timings.csv", list(record), [list(row.values()) for row in raw])
    # 保存値の比例則を、異なる寸法・別荷重の独立求解で確認する
    relative_errors = []
    for index in np.linspace(0, len(x)-1, 7, dtype=int):
        actual = solve_fem(x[index, 0], x[index, 1], args.force*1.5, s).deflection
        relative_errors.append(abs(actual/(truth[index]*1.5)-1))
    accuracy = []
    for factor in dict.fromkeys((1.0, args.factor)):
        accepted = predicted*factor <= args.limit
        false_accept = int(np.sum(accepted & (truth > args.limit)))
        accuracy.append({"factor": factor, "predicted_feasible": int(accepted.sum()),
                         "false_accept": false_accept,
                         "false_reject": int(np.sum(~accepted & (truth <= args.limit))),
                         "agreement_percent": float(np.mean(accepted == (truth <= args.limit))*100),
                         "false_accept_fraction_of_predictions_percent": false_accept/max(1, int(accepted.sum()))*100})
    result = {
        "measured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "grid": args.grid, "points": len(x), "force_N": args.force, "limit_mm": args.limit,
        "factor": args.factor, "repeats": args.repeats, "samples": len(data),
        "length_step_mm": float(lengths[1]-lengths[0]), "height_step_mm": float(heights[1]-heights[0]),
        "settings": bundle["settings"], "timings": {key: stats([row[key] for row in raw]) for key in raw[0] if key != "repeat"},
        "accuracy": accuracy, "fem_feasible": int(truth_mask.sum()),
        "grid_mape_percent": float(np.mean(abs(predicted/truth-1))*100),
        "grid_max_relative_error_percent": float(np.max(abs(predicted/truth-1))*100),
        "verification_count": len(verified), "verification_failures": sum(value > args.limit for _, value in verified),
        "linearity_max_relative_error": float(max(relative_errors)),
        "environment": {"os": platform.platform(), "processor": platform.processor(),
                        "logical_cpus": os.cpu_count(), "python": sys.version.split()[0],
                        "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), Path("beam_surrogate.py"),
                                       args.source / "training_data.csv", args.source / "surrogate.joblib")},
    }
    save_json(args.output / "benchmark_results.json", result)
    write_csv(args.output / "grid_comparison.csv",
              ["length_mm", "height_mm", "force_N", "fem_mm", "predicted_mm", "factored_mm", "fem_pass", "surrogate_pass"],
              [list(row)+[truth[i], predicted[i], args.factor*predicted[i], bool(truth[i] <= args.limit),
                          bool(predicted[i]*args.factor <= args.limit)] for i, row in enumerate(x)])
    write_csv(args.output / "verified_candidates.csv",
              ["length_mm", "height_mm", "force_N", "predicted_mm", "fem_mm", "fem_pass"],
              [list(x[index])+[predicted[index], value, bool(value <= args.limit)] for index, value in verified])
    raw_ranges = extract_ranges(lengths, heights, (predicted <= args.limit).reshape(shape))
    write_csv(args.output / "dimension_ranges_comparison.csv",
              ["method", "length_mm", "height_min_mm", "height_max_mm", "grid_points"],
              [[method]+row for method, ranges in (("FEM", fem_ranges), ("surrogate_raw", raw_ranges),
                                                   ("surrogate_factored", predicted_ranges)) for row in ranges])
    write_report(args.report, result)
    print(f"レポート保存: {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
