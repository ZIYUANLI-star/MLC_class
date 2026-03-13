from __future__ import annotations
import os
import pickle as pkl
import shutil
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib
import numpy as np
import torch
import torchvision
import tvm
from tvm import relax
from tvm.script import relax as R, tir as T


def get_schedule_cls():
    # 同时兼容新版 (tvm.tir) 与旧版 (tvm.s_tir) 的模块布局。
    if hasattr(tvm.tir, "Schedule"):
        return tvm.tir.Schedule
    if hasattr(tvm, "s_tir") and hasattr(tvm.s_tir, "Schedule"):
        return tvm.s_tir.Schedule
    return None


def get_meta_schedule_module():
    # 同时兼容新版 (tvm.meta_schedule) 与旧版 (tvm.s_tir.meta_schedule)。
    if hasattr(tvm, "meta_schedule"):
        return tvm.meta_schedule
    if hasattr(tvm, "s_tir") and hasattr(tvm.s_tir, "meta_schedule"):
        return tvm.s_tir.meta_schedule
    return None


def ensure_clang_on_path() -> Path:
    # Windows 下 MetaSchedule 的本地 runner 会在 worker 进程里调用 clang.exe。
    # 若 PATH 中缺少 clang，所有 trial 都会变成无效记录。
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for d in path_dirs:
        if d and Path(d, "clang.exe").exists():
            return Path(d)
    candidates = [
        Path("C:/Program Files/LLVM/bin"),
        Path("D:/Microsoft_Visual_Studio/2022/Community/VC/Tools/Llvm/x64/bin"),
    ]
    for c in candidates:
        if c.exists() and Path(c, "clang.exe").exists():
            os.environ["PATH"] = str(c) + os.pathsep + os.environ.get("PATH", "")
            print(f"[MetaSchedule] Added clang path: {c}")
            return c
    raise RuntimeError(
        "MetaSchedule requires clang.exe on PATH, but none was found."
    )


def ensure_windows_tvm_cc_env(target: tvm.target.Target) -> None:
    # TVM 在 Windows 调 clang 时会读取这些环境变量。
    if os.name != "nt":
        return
    clang_dir = ensure_clang_on_path()
    os.environ.setdefault("TVM_WIN_CC", str(Path(clang_dir, "clang.exe")))
    # TVM 默认 x86_64 可能走 gcc 风格链接路径并失败，因此显式使用 mtriple。
    mtriple = str(target.attrs.get("mtriple", "x86_64-pc-windows-msvc"))
    os.environ["TVM_WIN_TARGET"] = mtriple
    print(f"[MetaSchedule] TVM_WIN_TARGET={mtriple}")


def _meta_schedule_worker_initializer() -> None:
    # Worker 侧补丁：临时模块链接时允许未解析运行时符号。
    if os.name != "nt":
        return
    import tvm.contrib.cc as tvm_cc

    original_create_shared = tvm_cc.create_shared

    def _create_shared_allow_unresolved(output, objects, options=None, **kwargs):
        opts = [] if options is None else list(options)
        force_flag = "-Wl,/FORCE:UNRESOLVED"
        if force_flag not in opts:
            opts.append(force_flag)
        return original_create_shared(output, objects, options=opts, **kwargs)

    tvm_cc.create_shared = _create_shared_allow_unresolved


def get_block_compat(sch, block_name: str):
    # API 兼容：新版本用 get_block，部分构建仅提供 get_sblock。
    if hasattr(sch, "get_block"):
        return sch.get_block(block_name, func_name="main")
    return sch.get_sblock(block_name)


def ensure_full_tvm() -> None:
    # 在 TVM 环境不完整时尽早失败，并给出可执行的修复提示。
    missing = []
    if get_schedule_cls() is None:
        missing.append("Schedule API")
    if get_meta_schedule_module() is None:
        missing.append("MetaSchedule API")
    if not hasattr(relax, "build"):
        missing.append("tvm.relax.build")
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "This environment is not a full TVM build. Missing: "
            f"{joined}. Run setup_full_tvm_windows.ps1 first."
        )


def make_llvm_target() -> tvm.target.Target:
    # 显式设置 num-cores，避免某些 TVM 构建在 MetaSchedule 阶段触发断言。
    num_cores = max(1, os.cpu_count() or 1)
    if os.name == "nt":
        return tvm.target.Target(
            {"kind": "llvm", "mtriple": "x86_64-pc-windows-msvc", "num-cores": num_cores}
        )
    return tvm.target.Target({"kind": "llvm", "num-cores": num_cores})


@tvm.script.ir_module
class Matmul128:
    # 基准 matmul 工作负载，用于对比手工调度/随机搜索/MetaSchedule。
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.sblock("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def build_and_time_tir(
    mod: tvm.IRModule,
    target: tvm.target.Target,
    a_np: np.ndarray,
    b_np: np.ndarray,
    number: int = 10,
) -> float:
    # 编译 IRModule 并返回平均时延（毫秒）。
    dev = tvm.cpu()
    a_nd = tvm.runtime.tensor(a_np, dev)
    b_nd = tvm.runtime.tensor(b_np, dev)
    c_nd = tvm.runtime.empty((128, 128), "float32", dev)

    # 把这个 TIR 模块编译成真正能在 CPU 上跑的程序。
    lib = tvm.build(mod, target=target)
    timer = lib.time_evaluator("main", dev, number=number)
    return timer(a_nd, b_nd, c_nd).mean * 1000.0


def schedule_mm(sch: tvm.tir.Schedule, jfactor: int = 8) -> tvm.tir.Schedule:
    # 确定性的手工调度，作为人工基线。
    block_c = get_block_compat(sch, "C")

    # 拿到这三层循环对象。
    i, j, k = sch.get_loops(block_c)

    # 把第二层循环分成两部分，一部分是 jfactor 的倍数，一部分是余数。
    j0, j1 = sch.split(j, factors=[None, jfactor])

    # 把第一层循环、第二层循环的前一部分、第三层循环、第二层循环的后一部分重新排序。
    sch.reorder(i, j0, k, j1)

    # 把第三层循环分解成两个循环，一个循环是第三层循环，一个循环是第二层循环。
    sch.decompose_reduction(block_c, k)

    # 返回调度器。
    return sch


def stochastic_schedule_mm(sch: tvm.tir.Schedule) -> tvm.tir.Schedule:
    # 随机化调度变体，用于演示简单的搜索过程。
    block_c = get_block_compat(sch, "C")
    i, j, k = sch.get_loops(block_c)

    # 给 j 这一层随机选一种二层分块方式。
    j_factors = sch.sample_perfect_tile(j, n=2)
    j0, j1 = sch.split(j, factors=j_factors)
    sch.reorder(i, j0, k, j1)
    sch.decompose_reduction(block_c, k)
    return sch


def random_search(
    mod: tvm.IRModule,
    target: tvm.target.Target,
    a_np: np.ndarray,
    b_np: np.ndarray,
    num_trials: int = 8,
) -> tuple[tvm.tir.Schedule, float]:
    # 在 MetaSchedule 前做一个轻量随机搜索基线。
    best_sch = None
    best_ms = None
    schedule_cls = get_schedule_cls()
    assert schedule_cls is not None
    for i in range(num_trials):
        sch = stochastic_schedule_mm(schedule_cls(mod))
        ms_cost = build_and_time_tir(sch.mod, target, a_np, b_np)
        print(f"[RandomSearch] trial={i:02d}, latency={ms_cost:.3f} ms")
        if best_ms is None or ms_cost < best_ms:
            best_ms = ms_cost
            best_sch = sch
    assert best_sch is not None and best_ms is not None
    return best_sch, best_ms


def tune_tir_with_meta_schedule(
    mod: tvm.IRModule,
    target: tvm.target.Target,
    work_dir: str,
    max_trials: int,
    conservative: bool = False,
    medium_parallel: bool = False,
) -> tvm.tir.Schedule:
    # 端到端 MetaSchedule 调优入口，包含 Windows 场景稳定性保护。
    ms = get_meta_schedule_module()
    assert ms is not None
    schedule_cls = get_schedule_cls()
    assert schedule_cls is not None
    ensure_windows_tvm_cc_env(target)
    shutil.rmtree(work_dir, ignore_errors=True)
    builder = "local"
    runner = "local"
    measure_callbacks = "default"
    strategy = "evolutionary"
    task_scheduler = "gradient"
    cost_model = "xgb"
    num_tuning_cores: str | int = "physical"
    num_trials_per_iter = min(64, max_trials)
    if os.name == "nt":
        # 使用自定义 worker initializer，修补链接行为。
        if conservative:
            builder_workers = 1
        elif medium_parallel:
            # 中任务并行：在稳定性和速度之间折中。
            builder_workers = min(4, os.cpu_count() or 1)
        else:
            builder_workers = min(16, os.cpu_count() or 1)
        builder = ms.builder.LocalBuilder(
            max_workers=builder_workers,
            initializer=_meta_schedule_worker_initializer,
        )
        runner = ms.runner.LocalRunner(initializer=_meta_schedule_worker_initializer)
        # 保留 AddToDatabase，同时避免 Windows 下默认清理回调引发文件占用问题。
        measure_callbacks = [ms.measure_callback.AddToDatabase()]
        if conservative:
            # 保守模式：牺牲搜索吞吐，换取 Windows 稳定性。
            strategy = "replay-trace"
            task_scheduler = "round-robin"
            cost_model = "random"
            num_tuning_cores = 1
            num_trials_per_iter = 1
        elif medium_parallel:
            # 中任务并行模式：仍保持稳态策略，但允许小批并行。
            strategy = "replay-trace"
            task_scheduler = "round-robin"
            cost_model = "random"
            num_tuning_cores = min(4, os.cpu_count() or 1)
            num_trials_per_iter = min(4, max_trials)
    database = ms.tune_tir(
        mod=mod,
        target=target,
        max_trials_global=max_trials,
        num_trials_per_iter=num_trials_per_iter,
        work_dir=work_dir,
        builder=builder,
        runner=runner,
        measure_callbacks=measure_callbacks,  # measure_callbacks 是测量回调，用于测量模型的性能。
        strategy=strategy,
        task_scheduler=task_scheduler,
        cost_model=cost_model,
        num_tuning_cores=num_tuning_cores,
    )
    tuned = ms.tir_integration.compile_tir(database, mod, target=target)
    if tuned is None:
        print(f"[MetaSchedule] No valid record in {work_dir}, fallback to default schedule.")
        return schedule_cls(mod)
    return tuned


def load_fashion_mnist_sample() -> tuple[np.ndarray, int, list[str]]:
    # 从测试集随机取一条样本，用于快速推理验证。
    test_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]
    img, label = next(iter(test_loader))
    img_np = img.reshape(1, 28, 28).numpy()
    return img_np, int(label[0]), class_names


def save_sample_image(img: np.ndarray, out_path: str) -> None:
    # 保存采样图片，便于可视化核对预测结果。
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.imshow(img[0])
    plt.colorbar()
    plt.grid(False)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def load_mlp_params() -> dict[str, np.ndarray]:
    # 首次下载并在本地缓存 MLP 权重。
    params_path = Path("fasionmnist_mlp_params.pkl")
    if not params_path.exists():
        urlretrieve(
            "https://github.com/mlc-ai/web-data/raw/main/models/fasionmnist_mlp_params.pkl",
            str(params_path),
        )
    with open(params_path, "rb") as f:
        return pkl.load(f)


@tvm.script.ir_module
class MLPModule:
    # 用 Relax + TIR kernel 定义的两层 MLP，用于演示调优流程。
    @T.prim_func
    def linear0(
        X: T.Buffer((1, 784), "float32"),
        W: T.Buffer((128, 784), "float32"),
        B: T.Buffer((128,), "float32"),
        Z: T.Buffer((1, 128), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "linear0", "tir.noalias": True})
        Y = T.alloc_buffer((1, 128), "float32")
        for i, j, k in T.grid(1, 128, 784):
            with T.sblock("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, 128):
            with T.sblock("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @T.prim_func
    def linear1(
        X: T.Buffer((1, 128), "float32"),
        W: T.Buffer((10, 128), "float32"),
        B: T.Buffer((10,), "float32"),
        Z: T.Buffer((1, 10), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "linear1", "tir.noalias": True})
        Y = T.alloc_buffer((1, 10), "float32")
        for i, j, k in T.grid(1, 10, 128):
            with T.sblock("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, 10):
            with T.sblock("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(
        x: R.Tensor((1, 784), "float32"),
        w0: R.Tensor((128, 784), "float32"),
        b0: R.Tensor((128,), "float32"),
        w1: R.Tensor((10, 128), "float32"),
        b1: R.Tensor((10,), "float32"),
    ) -> R.Tensor((1, 10), "float32"):
        cls = MLPModule
        with R.dataflow():
            lv0 = R.call_tir(
                cls.linear0, (x, w0, b0), out_sinfo=R.Tensor((1, 128), "float32")
            )
            lv1 = R.nn.relu(lv0)
            out = R.call_tir(
                cls.linear1, (lv1, w1, b1), out_sinfo=R.Tensor((1, 10), "float32")
            )
            R.output(out)
        return out


def bind_params(mod: tvm.IRModule, params: dict[str, tvm.runtime.Tensor]) -> tvm.IRModule:
    # 将常量权重绑定到 Relax 的 main 函数。
    return relax.transform.BindParams("main", params)(mod)


def build_vm(mod: tvm.IRModule, target: tvm.target.Target) -> relax.VirtualMachine:
    # 编译 Relax 模块并创建 CPU VM 运行时。
    ex = relax.build(mod, target=target)
    return relax.VirtualMachine(ex, tvm.cpu())


def infer_and_time(
    vm: relax.VirtualMachine, data_nd: tvm.runtime.Tensor, class_names: list[str], tag: str
) -> None:
    # 执行一次推理拿到预测类别，并测量 VM 时延。
    nd_res = vm["main"](data_nd)
    pred_kind = int(np.argmax(nd_res.numpy(), axis=1)[0])
    ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=100)
    latency_ms = ftimer(data_nd).mean * 1000.0
    print(f"{tag} Prediction: {class_names[pred_kind]}")
    print(f"{tag} Latency: {latency_ms:.3f} ms")


def tune_and_update_kernels(
    mod_with_params: tvm.IRModule, target: tvm.target.Target, max_trials: int
) -> tvm.IRModule:
    # 分别调优每个 kernel，再把调优后的函数回填到模块里。
    updated = mod_with_params
    schedule_cls = get_schedule_cls()
    assert schedule_cls is not None
    for kernel_name in ("linear0", "linear1"):
        # 从完整模块里取出一个函数，单独装成一个新模块，并把它临时命名为 main，方便 MetaSchedule 调优。
        kernel_mod = tvm.IRModule.from_expr(
            updated[kernel_name].with_attr("global_symbol", "main")
        )
        # 用 MetaSchedule 调优这个 kernel。
        tuned_sch = tune_tir_with_meta_schedule(
            kernel_mod,
            target=target,
            work_dir=f"./tune_tmp/{kernel_name}",
            max_trials=max_trials,
            conservative=(os.name == "nt"),
        )
        tuned_mod = getattr(tuned_sch, "mod", None)
        if tuned_mod is None:
            print(
                f"[MetaSchedule] Invalid tuned schedule for {kernel_name}, fallback to default schedule."
            )
            tuned_mod = schedule_cls(kernel_mod).mod
        tuned_func = tuned_mod["main"].with_attr("global_symbol", kernel_name)
        gv = updated.get_global_var(kernel_name)
        updated.update_func(gv, tuned_func)
    return updated


def main() -> None:
    # 主流程：
    # 1) 对比 matmul 调度方案；2) 执行 MLP 推理；3) 可选进行 MLP kernel 调优。
    np.random.seed(0)
    ensure_full_tvm()
    target = make_llvm_target()
    print("TVM version:", tvm.__version__)
    print("Target:", target)

    a_np = np.random.rand(128, 128).astype("float32")
    b_np = np.random.rand(128, 128).astype("float32")

    baseline_ms = build_and_time_tir(Matmul128, target, a_np, b_np)
    print(f"[Matmul] baseline latency: {baseline_ms:.3f} ms")

    schedule_cls = get_schedule_cls()
    assert schedule_cls is not None

    manual_sch = schedule_mm(schedule_cls(Matmul128), jfactor=8)
    manual_ms = build_and_time_tir(manual_sch.mod, target, a_np, b_np)
    print(f"[Matmul] manual schedule latency: {manual_ms:.3f} ms")

    _, best_random_ms = random_search(Matmul128, target, a_np, b_np, num_trials=8)
    print(f"[Matmul] random search best latency: {best_random_ms:.3f} ms")

    tuned_matmul_sch = tune_tir_with_meta_schedule(
        Matmul128,
        target=target,
        work_dir="./tune_tmp/matmul",
        max_trials=32,
    )
    tuned_matmul_mod = getattr(tuned_matmul_sch, "mod", None)
    if tuned_matmul_mod is None:
        print("[MetaSchedule] Invalid tuned schedule object, fallback to manual schedule result.")
        tuned_matmul_mod = manual_sch.mod
    tuned_ms = build_and_time_tir(tuned_matmul_mod, target, a_np, b_np)
    print(f"[Matmul] meta schedule latency: {tuned_ms:.3f} ms")

    img, label, class_names = load_fashion_mnist_sample()
    save_sample_image(img, "fashion_mnist_sample_opt.png")
    print("Ground Truth:", class_names[label])

    mlp_params = load_mlp_params()
    # 把权重转换为 TVM 的运行时张量。
    nd_params = {k: tvm.runtime.tensor(v, tvm.cpu()) for k, v in mlp_params.items()}
    data_nd = tvm.runtime.tensor(img.reshape(1, 784), tvm.cpu())

    # 把权重绑定到 Relax 的 main 函数。
    mod_with_params = bind_params(MLPModule, nd_params)
    # 编译 Relax 模块并创建 CPU VM 运行时。
    vm = build_vm(mod_with_params, target)
    # 执行一次推理拿到预测类别，并测量 VM 时延。
    infer_and_time(vm, data_nd, class_names, tag="[MLP] before tuning")

    enable_mlp_tuning = os.environ.get("TVM_ENABLE_MLP_TUNING", "1") == "1"
    if enable_mlp_tuning:
        tuned_mod = tune_and_update_kernels(mod_with_params, target, max_trials=32)
        vm_tuned = build_vm(tuned_mod, target)
        infer_and_time(vm_tuned, data_nd, class_names, tag="[MLP] after tuning")
    else:
        print("[MLP] kernel tuning skipped by default for stability. "
              "Set TVM_ENABLE_MLP_TUNING=1 to enable.")


if __name__ == "__main__":
    main()