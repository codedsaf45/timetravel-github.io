import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

# TensorRT 엔진 로드
def load_engine(trt_file_path):
    with open(trt_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(trt_path)

# 엔진에서 입력 및 출력 정보 가져오기
input_shape = (1, 3, 640, 640)
output_shape = (1, 25200, 85)  # YOLOv8의 출력 형식에 맞게 조정

# GPU 메모리 할당
d_input = cuda.mem_alloc(1 * np.prod(input_shape) * np.float32().nbytes)
d_output = cuda.mem_alloc(1 * np.prod(output_shape) * np.float32().nbytes)
bindings = [int(d_input), int(d_output)]

# 스트림 생성
stream = cuda.Stream()

# 추론 수행
context = engine.create_execution_context()
input_data = np.random.random(input_shape).astype(np.float32)

# 입력 데이터 복사 및 추론 실행
cuda.memcpy_htod_async(d_input, input_data, stream)
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
output_data = np.empty(output_shape, dtype=np.float32)
cuda.memcpy_dtoh_async(output_data, d_output, stream)
stream.synchronize()

print("추론 결과:", output_data)
