import onnxruntime as rt
import numpy as np
model_path = r"test_dynamic_bidirectional_but_one_gru_and_output_consumed_only.onnx"

units = 5
batch_size = 6
x_val = np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=np.float32)
x_val = np.stack([x_val] * batch_size)

feed_dict = {"input_1:0": x_val}
input_names_with_port = ["input_1:0"]
output_names_with_port = ["output:0"]
m = rt.InferenceSession(model_path)
results = m.run(output_names_with_port, feed_dict)
print(results)


