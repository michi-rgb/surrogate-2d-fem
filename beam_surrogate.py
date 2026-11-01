"""2次元FEM → 教師データ → 決定木系サロゲート → 寸法探索の講座用教材。

単位は N, mm, MPa (= N/mm^2)。実行例は README.md を参照。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # GUIのないPCでも画像ファイルを保存できるようにする
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Settings:
    """講座で最初に確認する解析条件。寸法範囲は学習と探索で共用する。"""

    young: float = 210_000.0  # ヤング率 [MPa]：鋼を想定
    poisson: float = 0.3  # ポアソン比 [-]
    thickness: float = 20.0  # 面外の厚さ [mm]：今回は固定
    nx: int = 120  # 長さ方向の要素数（中央に節点を置くため偶数）
    ny: int = 12  # 高さ方向の要素数（同じく偶数）
    length_min: float = 400.0
    length_max: float = 1200.0
    height_min: float = 40.0
    height_max: float = 120.0
    force_min: float = 100.0
    force_max: float = 2000.0
    seed: int = 42

    def validate(self):
        values = np.array(list(asdict(self).values()), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("解析条件には有限の数を指定してください。")
        if self.young <= 0 or self.thickness <= 0 or not -1 < self.poisson < 0.5:
            raise ValueError("材料定数または厚さの指定が不適切です。")
        if any(n < 2 or n % 2 or int(n) != n for n in (self.nx, self.ny)):
            raise ValueError("nx, ny は2以上の偶数にしてください。")
        for low, high in ((self.length_min, self.length_max),
                          (self.height_min, self.height_max),
                          (self.force_min, self.force_max)):
            if not 0 < low < high:
                raise ValueError("学習範囲は 0 < 下限 < 上限 にしてください。")


@dataclass
class FEMResult:
    deflection: float  # 中央断面の高さ中央にある節点の下向き変位 [mm]
    nodes: np.ndarray
    displacement: np.ndarray
    reactions: np.ndarray  # 左支点の水平、左支点の鉛直、右支点の鉛直反力 [N]
    relative_residual: float


def element_stiffness(dx: float, dy: float, s: Settings) -> np.ndarray:
    """4節点四角形（Q4）平面応力要素。2×2ガウス積分で剛性を計算する。"""
    # 構成式: [σx, σy, τxy] = D @ [εx, εy, γxy]
    d = s.young / (1 - s.poisson**2) * np.array(
        [[1, s.poisson, 0], [s.poisson, 1, 0], [0, 0, (1-s.poisson)/2]]
    )
    ke = np.zeros((8, 8))
    # 要素内の節点順序: 左下 → 右下 → 右上 → 左上
    for xi in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
        for eta in (-1 / np.sqrt(3), 1 / np.sqrt(3)):
            natural = 0.25 * np.array([
                [-(1-eta), 1-eta, 1+eta, -(1+eta)],
                [-(1-xi), -(1+xi), 1+xi, 1-xi],
            ])
            # 長方形要素なのでヤコビアンは diag(dx/2, dy/2)
            grad = natural * np.array([[2/dx], [2/dy]])
            b = np.zeros((3, 8))
            b[0, 0::2] = grad[0]
            b[1, 1::2] = grad[1]
            b[2, 0::2] = grad[1]
            b[2, 1::2] = grad[0]
            ke += b.T @ d @ b * dx * dy / 4 * s.thickness
    return ke


def solve_fem(length: float, height: float, force: float,
              s: Settings = Settings()) -> FEMResult:
    """両端単純支持はりを、はり要素ではなく2次元連続体要素で解く。"""
    s.validate()
    if not np.isfinite([length, height, force]).all() or min(length, height, force) <= 0:
        raise ValueError("長さ・高さ・荷重は正の有限値にしてください。")
    nx, ny = s.nx, s.ny
    x, y = np.meshgrid(np.linspace(0, length, nx+1), np.linspace(0, height, ny+1))
    nodes = np.column_stack((x.ravel(), y.ravel()))
    lower_left = (np.arange(ny)[:, None] * (nx+1) + np.arange(nx)).ravel()
    elements = np.column_stack((lower_left, lower_left+1,
                                lower_left+nx+2, lower_left+nx+1))
    # 各節点に水平・鉛直の2自由度を割り当て、疎行列へ一括組立する
    dofs = (2 * elements[:, :, None] + np.arange(2)).reshape(-1, 8)
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, (1, 8)).ravel()
    ke = element_stiffness(length/nx, height/ny, s)
    ndof = 2 * len(nodes)
    k = coo_matrix((np.tile(ke.ravel(), len(elements)), (rows, cols)),
                   shape=(ndof, ndof)).tocsr()
    f = np.zeros(ndof)
    top_center = ny * (nx+1) + nx//2
    f[2*top_center+1] = -force  # 上面中央の1節点に下向き集中荷重
    # 左下: ux=uy=0（ピン）、右下: uy=0（ローラー）。右端のuxは自由
    fixed = np.array([0, 1, 2*nx+1])
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof)
    u[free] = spsolve(k[free][:, free], f[free])
    residual = k @ u - f
    center = (ny//2) * (nx+1) + nx//2
    deflection = -float(u[2*center+1])
    if not np.isfinite(u).all() or deflection <= 0:
        raise RuntimeError("FEM解が不正です。メッシュと境界条件を確認してください。")
    return FEMResult(deflection, nodes, u.reshape(-1, 2), residual[fixed],
                     float(np.linalg.norm(residual[free])/force))


def beam_theory(length, height, force, s=Settings()):
    """検算専用のEuler–Bernoulliはり理論。教師データの生成には使用しない。"""
    inertia = s.thickness * np.asarray(height)**3 / 12
    return force * np.asarray(length)**3 / (48 * s.young * inertia)


def write_csv(path, fields, rows):
    """Excelでも開きやすいBOM付きUTF-8で保存する。"""
    with Path(path).open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        writer.writerows(rows)


def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def dataset_id(samples: int, s: Settings) -> str:
    """データ生成条件から再現可能な識別子を作る。"""
    definition = {"schema_version": 1, "sampler": "latin-hypercube-v1",
                  "samples": samples, "settings": asdict(s)}
    canonical = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def generate_dataset(cache: Path, samples: int, s: Settings) -> str:
    """FEM教師データを生成して条件別キャッシュに保存する。"""
    s.validate()
    if samples < 50:
        raise ValueError("学習には50点以上を指定してください（推奨800点以上）。")
    cache.mkdir(parents=True, exist_ok=True)
    identifier = dataset_id(samples, s)
    folder = cache / identifier
    data_path = folder / "training_data.npz"
    manifest_path = folder / "metadata.json"
    if data_path.is_file() and manifest_path.is_file():
        print(f"既存キャッシュを使用: {identifier} ({folder})")
        return identifier
    start = time.perf_counter()
    # Latin Hypercubeで長さ・高さ・荷重の3変数を範囲内にまんべんなく配置
    low = [s.length_min, s.height_min, s.force_min]
    high = [s.length_max, s.height_max, s.force_max]
    x = qmc.scale(qmc.LatinHypercube(d=3, seed=s.seed).random(samples), low, high)
    y = np.empty(samples)
    for i, (length, height, force) in enumerate(x):
        y[i] = solve_fem(length, height, force, s).deflection
        if (i+1) % 100 == 0 or i+1 == samples:
            print(f"教師データ: {i+1}/{samples} 点", flush=True)
    fem_seconds = time.perf_counter() - start
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, x=x, y=y)
    save_json(manifest_path, {
        "dataset_id": identifier, "schema_version": 1,
        "sampler": "latin-hypercube-v1", "samples": samples,
        "settings": asdict(s), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fem_data_seconds": fem_seconds,
        "data_file": data_path.name,
    })
    print(f"教師データを保存: {identifier} ({folder})")
    return identifier


def train(output: Path, samples: int, s: Settings, model_name: str,
          cache: Path = Path("data_cache"), identifier: str | None = None):
    """条件に一致するキャッシュ済み教師データを使ってモデルを学習する。"""
    s.validate()
    if samples < 50:
        raise ValueError("学習には50点以上を指定してください（推奨800点以上）。")
    use_selected_conditions = identifier is not None
    identifier = identifier or dataset_id(samples, s)
    folder = cache / identifier
    data_path, manifest_path = folder / "training_data.npz", folder / "metadata.json"
    if not data_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"教師データのキャッシュがありません: {identifier}。先に generate-data を実行してください。")
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if use_selected_conditions:
        # IDを明示した場合は、そのキャッシュに記録された条件を採用する。
        samples = metadata["samples"]
        s = Settings(**metadata["settings"])
    expected = dataset_id(metadata["samples"], Settings(**metadata["settings"]))
    if metadata.get("dataset_id") != identifier or expected != identifier:
        raise ValueError(f"キャッシュ条件とIDが一致しません: {identifier}")
    if metadata["samples"] != samples or metadata["settings"] != asdict(s):
        raise ValueError("指定した学習条件とキャッシュ条件が一致しません。samples、解析設定、dataset-idを確認してください。")
    with np.load(data_path) as dataset:
        x, y = dataset["x"], dataset["y"]
    if x.shape != (samples, 3) or y.shape != (samples,) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"教師データキャッシュの内容が不正です: {data_path}")
    output.mkdir(parents=True, exist_ok=True)
    fem_seconds = metadata["fem_data_seconds"]
    # 同一の試験データで学習やパラメーター調整をしない
    train_idx, test_idx = train_test_split(np.arange(samples), test_size=0.2,
                                          random_state=s.seed)
    model_class = ExtraTreesRegressor if model_name == "extra-trees" else RandomForestRegressor
    model = model_class(n_estimators=300, min_samples_leaf=1,
                        random_state=s.seed, n_jobs=-1)
    # 桁の違うたわみを公平に扱うため、入力と出力を対数変換する
    model.fit(np.log(x[train_idx]), np.log(y[train_idx]))
    predicted = np.exp(model.predict(np.log(x[test_idx])))
    relative = np.abs(predicted/y[test_idx] - 1)
    metrics = {
        "model": model_name, "samples": samples,
        "dataset_id": identifier,
        "train_samples": len(train_idx), "test_samples": len(test_idx),
        "mae_mm": float(mean_absolute_error(y[test_idx], predicted)),
        "mape_percent": float(mean_absolute_percentage_error(y[test_idx], predicted)*100),
        "r2": float(r2_score(y[test_idx], predicted)),
        "max_relative_error_percent": float(relative.max()*100),
        "fem_data_seconds": fem_seconds,
        "feature_importance_L_H_F": model.feature_importances_.tolist(),
    }
    split = np.full(samples, "train", dtype=object)
    split[test_idx] = "test"
    write_csv(output / "training_data.csv",
              ["length_mm", "height_mm", "force_N", "fem_deflection_mm", "split"],
              [list(x[i]) + [y[i], split[i]] for i in range(samples)])
    write_csv(output / "test_predictions.csv",
              ["length_mm", "height_mm", "force_N", "fem_deflection_mm", "predicted_mm"],
              [list(x[j]) + [y[j], predicted[i]] for i, j in enumerate(test_idx)])
    save_json(output / "settings.json", asdict(s))
    save_json(output / "dataset_metadata.json", metadata)
    save_json(output / "metrics.json", metrics)
    # joblibはPythonコードを含み得るため、自分で生成したファイルだけを読み込む
    joblib.dump({"model": model, "settings": asdict(s)}, output / "surrogate.joblib")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y[test_idx], predicted, s=16, alpha=0.65)
    limits = [min(y[test_idx].min(), predicted.min()), max(y[test_idx].max(), predicted.max())]
    ax.plot(limits, limits, "k--")
    ax.set(xscale="log", yscale="log", xlabel="FEM deflection [mm]",
           ylabel="Surrogate deflection [mm]", title=f"Held-out test: MAPE = {metrics['mape_percent']:.1f}%")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "test_accuracy.png", dpi=160)
    plt.close(fig)
    print(f"テスト誤差: MAPE={metrics['mape_percent']:.2f}%, R2={metrics['r2']:.4f}")
    return metrics


def feasible_intervals(heights, mask):
    """合格が飛び地になる場合にも対応し、連続するグリッド点の区間を返す。"""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1)
    return [(float(heights[g[0]]), float(heights[g[-1]]), len(g)) for g in groups]


def predict(output: Path, force: float, limit: float, grid: int,
            safety_factor: float, verify_count: int):
    """順方向サロゲートを格子探索し、制約を満たす寸法の組合せを列挙する。"""
    if not np.isfinite([force, limit, safety_factor]).all() or force <= 0 or limit <= 0:
        raise ValueError("荷重とたわみ上限には正の有限値を指定してください。")
    if grid < 2 or safety_factor < 1 or verify_count < 0:
        raise ValueError("grid>=2、safety-factor>=1、verify>=0 にしてください。")
    bundle = joblib.load(output / "surrogate.joblib")
    model, s = bundle["model"], Settings(**bundle["settings"])
    s.validate()
    if not s.force_min <= force <= s.force_max:
        raise ValueError(f"荷重は学習範囲 {s.force_min}～{s.force_max} N 内に指定してください。")
    lengths = np.linspace(s.length_min, s.length_max, grid)
    heights = np.linspace(s.height_min, s.height_max, grid)
    ll, hh = np.meshgrid(lengths, heights, indexing="ij")
    x = np.column_stack((ll.ravel(), hh.ravel(), np.full(ll.size, force)))
    start = time.perf_counter()
    predicted = np.exp(model.predict(np.log(x))).reshape(ll.shape)
    prediction_seconds = time.perf_counter() - start
    conservative = safety_factor * predicted
    feasible = conservative <= limit
    # 行ごとの高さ範囲なので、LとHを独立な上下限にまとめる誤解を避けられる
    intervals = []
    for i, length in enumerate(lengths):
        for low, high, count in feasible_intervals(heights, feasible[i]):
            intervals.append([length, low, high, count])
    write_csv(output / "dimension_ranges.csv",
              ["length_mm", "height_min_mm", "height_max_mm", "grid_points_in_interval"], intervals)
    write_csv(output / "design_grid.csv",
              ["length_mm", "height_mm", "force_N", "predicted_mm", "factored_mm", "predicted_feasible"],
              (list(x[i]) + [predicted.ravel()[i], conservative.ravel()[i], bool(feasible.ravel()[i])]
               for i in range(len(x))))
    # 長さ全域から境界付近の候補を選び、FEMで再確認する
    # 最小体積の予測候補も必ず含める（高さが最小とは限らないため全合格点から選ぶ）
    boundary = [i*grid + np.flatnonzero(row)[0] for i, row in enumerate(feasible) if row.any()]
    selected = []
    if boundary and verify_count:
        accepted = np.flatnonzero(feasible.ravel())
        best = int(accepted[np.argmin(x[accepted, 0]*x[accepted, 1])])
        selected = [best]
        for pos in np.linspace(0, len(boundary)-1, min(verify_count, len(boundary)), dtype=int):
            if boundary[pos] not in selected and len(selected) < verify_count:
                selected.append(boundary[pos])
    verified = []
    for index in selected:
        length, height, _ = x[index]
        actual = solve_fem(length, height, force, s).deflection
        verified.append([length, height, force, predicted.ravel()[index], actual,
                         bool(actual <= limit), length*height*s.thickness])
    write_csv(output / "fem_verified_candidates.csv",
              ["length_mm", "height_mm", "force_N", "predicted_mm", "fem_mm", "fem_pass", "volume_mm3"],
              verified)
    summary = {
        "force_N": force, "deflection_limit_mm": limit, "safety_factor": safety_factor,
        "grid": grid, "length_step_mm": float(lengths[1]-lengths[0]),
        "height_step_mm": float(heights[1]-heights[0]),
        "predicted_feasible_grid_points": int(feasible.sum()),
        "surrogate_prediction_seconds": prediction_seconds,
        "fem_verified_count": len(verified),
        "fem_failed_count": sum(not row[5] for row in verified),
        "note": "寸法範囲は離散格子上の予測。区間内部と未検証点の合格は保証しない。",
    }
    save_json(output / "search_summary.json", summary)
    fig, ax = plt.subplots(figsize=(8, 5))
    mesh = ax.pcolormesh(lengths, heights, conservative.T, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="Factored predicted deflection [mm]")
    # 合格・不合格の両方があるときだけ境界線を描く
    if feasible.any() and not feasible.all():
        ax.contour(lengths, heights, conservative.T, levels=[limit], colors="red", linewidths=2)
    for length, height, _, _, _, passed, _ in verified:
        ax.scatter(length, height, marker="o" if passed else "x",
                   color="white" if passed else "red", edgecolors="black" if passed else None, s=50)
    ax.set(xlabel="Span L [mm]", ylabel="Height H [mm]",
           title=f"F={force:g} N, limit={limit:g} mm, factor={safety_factor:g}\n"
                 "Red: predicted boundary | Circles: FEM passed | Crosses: FEM failed")
    fig.tight_layout()
    fig.savefig(output / "feasible_region.png", dpi=160)
    plt.close(fig)
    print(f"予測合格: {feasible.sum()}/{ll.size} 点（余裕係数 {safety_factor:g}）")
    print(f"FEM再確認: {len(verified)} 点、うち上限超過 {summary['fem_failed_count']} 点")
    if intervals:
        print("長さごとの高さ範囲 [mm]（表示は先頭5区間、全結果は dimension_ranges.csv）:")
        for length, low, high, _ in intervals[:5]:
            print(f"  L={length:.1f}: H={low:.1f}～{high:.1f}")
    else:
        print("探索した格子点に予測合格寸法はありません。寸法範囲の再設定と再学習を検討してください。")
    passed = [row for row in verified if row[5]]
    if passed:
        best = min(passed, key=lambda row: row[6])
        print(f"FEM確認済み候補中の最小体積: L={best[0]:.1f}, H={best[1]:.1f} mm, "
              f"たわみ={best[4]:.4f} mm（探索全体の最適解を保証するものではありません）")
    return summary


def plot_fem(result: FEMResult, length, height, output):
    """変形を拡大表示する。局所応力ではなく変位を観察するための図。"""
    scale = 0.12 * height / np.max(np.abs(result.displacement))
    deformed = result.nodes + scale*result.displacement
    fig, ax = plt.subplots(figsize=(10, 3))
    mesh = ax.scatter(deformed[:, 0], deformed[:, 1], c=-result.displacement[:, 1], s=3)
    ax.plot([0, length, length, 0, 0], [0, 0, height, height, 0], "k--", lw=1)
    ax.scatter([0, length], [0, 0], marker="^", c="red")
    ax.annotate("F", xy=(length/2, height), xytext=(length/2, height*1.5),
                ha="center", arrowprops={"arrowstyle": "->"})
    fig.colorbar(mesh, ax=ax, label="Downward displacement [mm]")
    ax.set(xlabel="x [mm]", ylabel="y [mm]", title=f"2D plane stress FEM, displacement scale x{scale:.1f}")
    ax.set_aspect("equal")
    ax.set_xlim(-0.05*length, 1.05*length)
    ax.set_ylim(-0.4*height, 1.8*height)  # 荷重矢印とタイトルの重なりを防ぐ
    fig.tight_layout()
    fig.savefig(output / "fem_deformation.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["demo", "generate-data", "train", "predict", "fem"], default="demo")
    parser.add_argument("--output", type=Path, default=Path("output"), help="出力フォルダー")
    parser.add_argument("--samples", type=int, default=800, help="教師データ点数")
    parser.add_argument("--data-cache", type=Path, default=Path("data_cache"), help="条件別教師データのキャッシュ先")
    parser.add_argument("--dataset-id", help="使用する教師データID（省略時は点数と解析条件から算出）")
    parser.add_argument("--model", choices=["extra-trees", "random-forest"], default="extra-trees")
    parser.add_argument("--force", type=float, default=1000, help="下向き荷重 [N]")
    parser.add_argument("--limit", type=float, default=0.5, help="中央たわみの上限 [mm]")
    parser.add_argument("--length", type=float, default=800, help="単独FEMの支点間長さ [mm]")
    parser.add_argument("--height", type=float, default=80, help="単独FEMの高さ [mm]")
    parser.add_argument("--grid", type=int, default=81, help="各寸法方向の探索点数")
    parser.add_argument("--safety-factor", type=float, default=1.15, help="予測たわみに掛ける余裕係数")
    parser.add_argument("--verify", type=int, default=12, help="FEMで再確認する候補数")
    parser.add_argument("--nx", type=int, default=120, help="長さ方向の要素数（偶数）")
    parser.add_argument("--ny", type=int, default=12, help="高さ方向の要素数（偶数）")
    args = parser.parse_args()
    s = Settings(nx=args.nx, ny=args.ny)
    try:
        if args.command in ("demo", "generate-data"):
            identifier = generate_dataset(args.data_cache, args.samples, s)
            if args.command == "generate-data":
                print(f"dataset-id: {identifier}")
        if args.command in ("demo", "train"):
            train(args.output, args.samples, s, args.model, args.data_cache, args.dataset_id)
        if args.command in ("demo", "predict"):
            predict(args.output, args.force, args.limit, args.grid, args.safety_factor, args.verify)
        if args.command == "fem":
            result = solve_fem(args.length, args.height, args.force, s)
            args.output.mkdir(parents=True, exist_ok=True)
            plot_fem(result, args.length, args.height, args.output)
            print(f"中央たわみ FEM: {result.deflection:.6f} mm")
            print(f"はり理論（参考）: {beam_theory(args.length, args.height, args.force, s):.6f} mm")
            print(f"反力 [左Fx, 左Fy, 右Fy]: {result.reactions.round(6)} N")
        print(f"出力先: {args.output.resolve()}")
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"入力エラー: {error}\n")


if __name__ == "__main__":
    main()
