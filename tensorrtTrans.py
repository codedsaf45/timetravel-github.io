import tensorrt as trt
import onnx

# ONNX 모델 파일 경로
onnx_path = "best.onnx"
trt_path = "best.engine"

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30  # 워크스페이스 메모리 설정 (1GB)
        builder.fp16_mode = True  # FP16 모드 활성화 (지원하는 경우)

        # ONNX 파일 로드 및 파싱
        with open(onnx_file_path, "rb") as model:
            if not parser.parse(model.read()):
                print("ERROR: Failed to parse the ONNX file.")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # TensorRT 엔진 빌드
        engine = builder.build_cuda_engine(network)
        return engine

# 엔진 빌드
engine = build_engine(onnx_path)

# 엔진 파일 저장
with open(trt_path, "wb") as f:
    f.write(engine.serialize())

print("TensorRT 엔진 변환 완료:", trt_path)
