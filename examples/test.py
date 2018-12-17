import onnx
import onnxruntime as rt
import numpy as np

model_path = r"test_dynamic_bidirectional_but_one_gru_and_output_consumed_only.onnx"
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
print('The model is checked!')

units = 5
batch_size = 6
x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
x_val = np.stack([x_val] * batch_size)

feed_dict = {"input_1:0": x_val}
input_names_with_port = ["input_1:0"]
output_names_with_port = ["output:0"]

print("load model")
m = rt.InferenceSession(model_path)
print(str(m.get_inputs()))
print(str(m.get_outputs()))

print("run model")
results = m.run(output_names_with_port, feed_dict)
print(results)


