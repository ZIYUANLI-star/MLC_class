
from pathlib import Path
from urllib.request import urlretrieve

import tvm
from tvm.ir.module import IRModule

# T表示TIR(TensorIR),R表示Relax(计算图)
from tvm.script import tir as T, relax as R

import numpy as np
from tvm import relax
import platform


import torchvision
import torch

# 第1层（模型/前端）：加载测试数据，准备端到端推理输入
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 第5层（低级编译器与代码生成）：配置目标后端（LLVM/Host）
LLVM_TARGET_CONFIG = {"kind": "llvm"}
if platform.system() == "Windows":
    LLVM_TARGET_CONFIG["mtriple"] = "x86_64-pc-windows-msvc"
LLVM_TARGET = tvm.target.Target(LLVM_TARGET_CONFIG)
LLVM_HOST = tvm.target.Target(dict(LLVM_TARGET_CONFIG))


def relax_build_compat(ir_mod):
    """Build with explicit host target when possible."""
    try:
        compat_target = tvm.target.Target(dict(LLVM_TARGET_CONFIG), host=LLVM_HOST)
        return relax.build(ir_mod, target=compat_target)
    except TypeError:
        return relax.build(ir_mod, target=LLVM_TARGET, target_host=LLVM_HOST)

# 第1层（模型/前端）：取一个样本并可视化，便于观察输入
img, label = next(iter(test_loader))
img = img.reshape(1, 28, 28).numpy()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.grid(False)
plt.savefig("fashion_mnist_sample.png", bbox_inches="tight")
plt.close()

print("Class:", class_names[label[0]])





PARAMS_PATH = Path("fasionmnist_mlp_params.pkl")
if not PARAMS_PATH.exists():
    urlretrieve(
        "https://github.com/mlc-ai/web-data/raw/main/models/fasionmnist_mlp_params.pkl",
        str(PARAMS_PATH),
    )


# 第1层（模型/前端）：Numpy高层语义实现，作为参考基线
def numpy_mlp(data, w0, b0, w1, b1):
    lv0 = data @ w0.T + b0
    lv1 = np.maximum(lv0, 0)
    lv2 = lv1 @ w1.T + b1
    return lv2




import pickle as pkl
with open(PARAMS_PATH, "rb") as f:
    mlp_params = pkl.load(f)
res = numpy_mlp(img.reshape(1, 784),
                mlp_params["w0"],
                mlp_params["b0"],
                mlp_params["w1"],
                mlp_params["b1"])
print(res)
pred_kind = res.argmax(axis=1)
print(pred_kind)
print("Numpy-MLP Prediction:", class_names[pred_kind[0]])


# 第3层（算子级表达的教学对照）：手写低层循环版本，模拟算子实现细节
def lnumpy_linear0(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
    Y = np.empty((1, 128), dtype="float32")
    for i in range(1):
        for j in range(128):
            for k in range(784):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

    for i in range(1):
        for j in range(128):
            Z[i, j] = Y[i, j] + B[j]


def lnumpy_relu0(X: np.ndarray, Y: np.ndarray):
     for i in range(1):
        for j in range(128):
            Y[i, j] = np.maximum(X[i, j], 0)

def lnumpy_linear1(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
    Y = np.empty((1, 10), dtype="float32")
    for i in range(1):
        for j in range(10):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

    for i in range(1):
        for j in range(10):
            Z[i, j] = Y[i, j] + B[j]


def lnumpy_mlp(data, w0, b0, w1, b1):
    lv0 = np.empty((1, 128), dtype="float32")
    lnumpy_linear0(data, w0, b0, lv0)

    lv1 = np.empty((1, 128), dtype="float32")
    lnumpy_relu0(lv0, lv1)

    out = np.empty((1, 10), dtype="float32")
    lnumpy_linear1(lv1, w1, b1, out)
    return out

result =lnumpy_mlp(
    img.reshape(1, 784),
    mlp_params["w0"],
    mlp_params["b0"],
    mlp_params["w1"],
    mlp_params["b1"])

pred_kind = result.argmax(axis=1)
print("Low-level Numpy MLP Prediction:", class_names[pred_kind[0]])


# 第2/3层组合：同一个IRModule中包含TensorIR算子与Relax计算图
@tvm.script.ir_module
class MyModule:
    # 第3层（算子级IR，TensorIR）：ReLU算子
    @T.prim_func
    def relu0(X: T.Buffer((1, 128), "float32"),
              Y: T.Buffer((1, 128), "float32")):
        T.func_attr({"global_symbol": "relu0", "tir.noalias": True})
        for i, j in T.grid(1, 128):
            with T.sblock("Y"):
                vi, vj = T.axis.remap("SS", [i, j])
                Y[vi, vj] = T.max(X[vi, vj], T.float32(0))
            
    # 第3层（算子级IR，TensorIR）：Linear0算子
    @T.prim_func
    def linear0(X: T.Buffer((1, 784), "float32"),
                W: T.Buffer((128, 784), "float32"),
                B: T.Buffer((128,), "float32"),
                Z: T.Buffer((1, 128), "float32")):
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
                Z[vi, vj] =  Y[vi, vj] + B[vj]
                
    # 第3层（算子级IR，TensorIR）：Linear1算子
    @T.prim_func
    def linear1(X: T.Buffer((1, 128), "float32"),
                W: T.Buffer((10, 128), "float32"),
                B: T.Buffer((10,), "float32"),
                Z: T.Buffer((1, 10), "float32")):
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


    # 第2层（高级图IR，Relax）：定义端到端计算图并编排算子调用
    @R.function
    def main(x: R.Tensor((1, 784), "float32"),
             w0: R.Tensor((128, 784), "float32"),
             b0: R.Tensor((128,), "float32"),
             w1: R.Tensor((10, 128), "float32"),
             b1: R.Tensor((10,), "float32")):
        with R.dataflow():
            cls = MyModule
            lv0 = R.call_tir(cls.linear0, (x, w0, b0), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv1 = R.call_tir(cls.relu0, (lv0,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            out = R.call_tir(cls.linear1, (lv1, w1, b1), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(out)
        return out


def lnumpy_call_tir(prim_func, inputs, shape, dtype):
    res = np.empty(shape, dtype=dtype)
    prim_func(*inputs, res)
    return res

def lnumpy_mlp_with_call_tir(data, w0, b0, w1, b1):
    lv0 = lnumpy_call_tir(lnumpy_linear0, (data, w0, b0), (1, 128), dtype="float32")
    lv1 = lnumpy_call_tir(lnumpy_relu0, (lv0, ), (1, 128), dtype="float32")
    out = lnumpy_call_tir(lnumpy_linear1, (lv1, w1, b1), (1, 10), dtype="float32")
    return out

result = lnumpy_mlp_with_call_tir(
    img.reshape(1, 784),
    mlp_params["w0"],
    mlp_params["b0"],
    mlp_params["w1"],
    mlp_params["b1"])

pred_kind = np.argmax(result, axis=1)
print("Low-level Numpy with callTIR Prediction:", class_names[pred_kind[0]])

# 第2层（高级图IR可视化）：打印模块结构
MyModule.show()


# 第4层（自动调优与性能搜索）：本脚本未启用MetaSchedule等自动调优流程
# 当前直接进入编译阶段（第5层）
ex = relax_build_compat(MyModule)
type(ex)



# 第6层（运行时）：创建Relax VM执行器
# 第7层（硬件执行）：使用tvm.cpu()在本机CPU上运行
vm = relax.VirtualMachine(ex, tvm.cpu())



# 第6层（运行时）：将输入与参数封装为TVM Runtime Tensor
data_nd = tvm.runtime.tensor(img.reshape(1, 784))
nd_params = {k: tvm.runtime.tensor(v) for k, v in mlp_params.items()}


# 第6层（运行时）：调用编译后的main进行推理
nd_res = vm["main"](data_nd,
                    nd_params["w0"],
                    nd_params["b0"],
                    nd_params["w1"],
                    nd_params["b1"])
nd_res



pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModule Prediction:", class_names[pred_kind[0]])


# 第2/3层组合 + 外部库调用：Relax图中插入call_dps_packed，部分算子走外部实现
@tvm.script.ir_module
class MyModuleWithExternCall:
    # 第3层（算子级IR，TensorIR）：身份映射算子（示例）
    @T.prim_func
    def identity0(x: T.Buffer((1, 784), "float32"),
                  out: T.Buffer((1, 784), "float32")):
        T.func_attr({"global_symbol": "identity0", "tir.noalias": True})
        for i, j in T.grid(1, 784):
            with T.sblock("out"):
                vi, vj = T.axis.remap("SS", [i, j])
                out[vi, vj] = x[vi, vj]

    @R.function
    def main(x: R.Tensor((1, 784), "float32"),
             w0: R.Tensor((128, 784), "float32"),
             b0: R.Tensor((128,), "float32"),
             w1: R.Tensor((10, 128), "float32"),
             b1: R.Tensor((10,), "float32")):
        with R.dataflow():
            cls = MyModuleWithExternCall
            x_id = R.call_tir(cls.identity0, (x,), out_sinfo=R.Tensor((1, 784), dtype="float32"))
            
            # call_dps_packed：外部库调用，将PyTorch的函数注册到TVM中，然后通过call_dps_packed调用
            lv0 = R.call_dps_packed("env.linear", (x_id, w0, b0), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            out = R.call_dps_packed("env.linear", (lv1, w1, b1), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(out)
        return out

# 第2层中的“外部库调用”分支：注册env.linear到PyTorch实现
@tvm.register_global_func("env.linear", override=True)
def torch_linear(x: tvm.runtime.Tensor,
                 w: tvm.runtime.Tensor,
                 b: tvm.runtime.Tensor,
                 out: tvm.runtime.Tensor):
    # dlpack：将TVM的Tensor转换为PyTorch的Tensor,然后调用PyTorch的函数
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    b_torch = torch.from_dlpack(b)
    out_torch = torch.from_dlpack(out)
    torch.mm(x_torch, w_torch.T, out=out_torch)
    torch.add(out_torch, b_torch, out=out_torch)

# 第2层中的“外部库调用”分支：注册env.relu到PyTorch实现
@tvm.register_global_func("env.relu", override=True)
def lnumpy_relu(x: tvm.runtime.Tensor,
                out: tvm.runtime.Tensor):
    x_torch = torch.from_dlpack(x)
    out_torch = torch.from_dlpack(out)
    torch.maximum(x_torch, torch.Tensor([0.0]), out=out_torch)


# 第5层编译 + 第6/7层执行：构建并在CPU上运行外部调用版本
try:
    extern_target = tvm.target.Target(dict(LLVM_TARGET_CONFIG), host=LLVM_HOST)
    ex = relax.build(MyModuleWithExternCall, target=extern_target)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    nd_res = vm["main"](data_nd,
                        nd_params["w0"],
                        nd_params["b0"],
                        nd_params["w1"],
                        nd_params["b1"])

    pred_kind = np.argmax(nd_res.numpy(), axis=1)
    print("MyModuleWithExternCall Prediction:", class_names[pred_kind[0]])
except Exception as err:
    print("MyModuleWithExternCall skipped:", err)

# 第2/3层混合示例：部分算子用TensorIR，部分算子走外部库
@tvm.script.ir_module
class MyModuleMixture:
    @T.prim_func
    def linear0(X: T.Buffer((1, 784), "float32"),
                W: T.Buffer((128, 784), "float32"),
                B: T.Buffer((128,), "float32"),
                Z: T.Buffer((1, 128), "float32")):
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
                Z[vi, vj] =  Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, 784), "float32"),
             w0: R.Tensor((128, 784), "float32"),
             b0: R.Tensor((128,), "float32"),
             w1: R.Tensor((10, 128), "float32"),
             b1: R.Tensor((10,), "float32")):
        with R.dataflow():
            cls = MyModuleMixture
            lv0 = R.call_tir(cls.linear0, (x, w0, b0), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            out = R.call_dps_packed("env.linear", (lv1, w1, b1), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(out)
        return out





# 第5层编译 + 第6/7层执行：运行混合后端版本
try:
    ex = relax_build_compat(MyModuleMixture)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    nd_res = vm["main"](data_nd,
                        nd_params["w0"],
                        nd_params["b0"],
                        nd_params["w1"],
                        nd_params["b1"])

    pred_kind = np.argmax(nd_res.numpy(), axis=1)
    print("MyModuleMixture Prediction:", class_names[pred_kind[0]])
except Exception as err:
    print("MyModuleMixture skipped:", err)





# 第2层（图级优化）：将参数绑定进main，减少运行时传参开销
try:
    MyModuleWithParams = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
    MyModuleWithParams.show()

    ex = relax_build_compat(MyModuleWithParams)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    nd_res = vm["main"](data_nd)

    pred_kind = np.argmax(nd_res.numpy(), axis=1)
    print("MyModuleWithParams Prediction:", class_names[pred_kind[0]])
except Exception as err:
    print("MyModuleWithParams skipped:", err)
