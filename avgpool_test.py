import onnx
import onnxruntime as rt
import numpy as np

model_path = r"test_avgpool.onnx"
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
print('The model is checked!')

def make_xval(shape):
    x_val = np.arange(np.prod(shape)).astype("float32").reshape(shape)
    return x_val


padding, x_shape, ksize, strides = ('SAME', [32, 35, 35, 32], [1, 5, 5, 1], [1, 2, 2, 1])
x_val = make_xval(x_shape)

feed_dict = {"input:0": x_val}
output_names_with_port = ["output:0"]

print("load model")
m = rt.InferenceSession(model_path)

print("run model")
result = m.run(output_names_with_port, feed_dict)[0]
print(result.shape)

# np.save("avgpool.npy", result)
np.savetxt("avgpool.txt", result.flatten(), fmt='%s')

