# KV Cache Scale Statistics

이 예제는 KV cache quantization (Q4_0) 시 runtime에 동적으로 계산되는 scale 값들의 통계를 추적합니다.

## 기능

- **Min/Max**: 모든 블록의 scale 값 중 최소/최대값
- **Mean**: scale 값들의 평균
- **Variance**: scale 값들의 분산
- **Std Dev**: scale 값들의 표준편차

각 블록(32개 요소)마다 scale이 계산되며, 이 통계는 1000개 블록마다 자동으로 출력됩니다.

## 빌드

```bash
cd /home/yjkim00/specinfer.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make kv-cache-scale-stats
```

## 사용법

```bash
# 기본 사용
./bin/kv-cache-scale-stats -m <model_path> -p "Your prompt here" --cache-type-k q4_0

# 예시
./bin/kv-cache-scale-stats \
    -m models/llama-7b.gguf \
    -p "Once upon a time" \
    -n 100 \
    --cache-type-k q4_0
```

## 출력 예시

```
=== Q4_0 Scale Statistics (blocks=1000) ===
  Min:      0.000123
  Max:      0.456789
  Mean:     0.123456
  Variance: 0.012345
  Std Dev:  0.111111
==========================================
```

## 참고사항

- Q4_0 quantization을 사용하는 경우에만 통계가 수집됩니다
- `--cache-type-k q4_0` 옵션을 반드시 사용해야 합니다
- 통계는 KV cache에 저장되는 K 텐서의 quantization 시에만 수집됩니다
