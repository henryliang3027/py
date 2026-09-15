import time

import onnxruntime as ort
import numpy as np

model_path = "/py/NNstreamer/resnet50_market1501_aicity156.onnx"

# 1. 確認這份 onnxruntime build 是否真的內含 Neuron EP
print(ort.get_available_providers())
assert "NeuronExecutionProvider" in ort.get_available_providers()

# 2. Neuron 專屬編譯參數（對應 build_neuron_options）
neuron_options = {
    "NEURON_FLAG_USE_FP16": "1",
    "NEURON_FLAG_MIN_GROUP_SIZE": "0",
    "NEURON_FLAG_OPTIMIZATION_STRING": "--opt=3 --num-mdla=1 --reshape-to-4d",
}

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 3. 這行觸發 graph partition + NPU 編譯
session = ort.InferenceSession(
    model_path,
    sess_options,
    providers=[("NeuronExecutionProvider", neuron_options)],
)

# 4. 準備輸入並跑推論
input_meta = session.get_inputs()[0]
shape = [1 if isinstance(d, (str, type(None))) else d for d in input_meta.shape]
dummy_input = np.random.randn(*shape).astype(np.float32)

# warm-up run (first inference includes NPU compilation overhead)
outputs = session.run(None, {input_meta.name: dummy_input})
print(outputs[0].shape)

# 5. 量測推論時間
num_runs = 50
timings = []
for _ in range(num_runs):
    start = time.perf_counter()
    session.run(None, {input_meta.name: dummy_input})
    timings.append(time.perf_counter() - start)

timings_ms = np.array(timings) * 1000
print(f"inference time: avg={timings_ms.mean():.2f} ms, "
      f"min={timings_ms.min():.2f} ms, max={timings_ms.max():.2f} ms "
      f"({num_runs} runs)")
