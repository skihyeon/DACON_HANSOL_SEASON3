FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 비대화형 설치 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 기본 패키지 및 필요한 라이브러리 설치
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y gcc-11 g++-11 \
    && rm -rf /var/lib/apt/lists/*

# gcc/g++ 버전 업데이트
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

# 작업 디렉토리 생성
WORKDIR /app

# 필요한 Python 패키지 설치
RUN pip install --no-cache-dir pandas \
    langchain \
    langchain-community \
    langchain-huggingface \
    faiss-cpu \
    transformers \
    sentence-transformers \
    accelerate \
    bitsandbytes>=0.43.2 \
    scipy \
    datasets \
    tqdm \
    setproctitle
# CUDA 환경 변수 설정
ENV CUDA_VISIBLE_DEVICES=0

# 성능 최적화 환경 변수 설정
ENV OMP_NUM_THREADS=8
ENV TOKENIZERS_PARALLELISM=true

# 소스 코드 복사
COPY main.py /app/
COPY data/ /app/data/

# 결과 저장을 위한 디렉토리 생성
RUN mkdir -p /app/submission

# 실행 명령어 (로그 버퍼링 비활성화)
CMD ["python3", "-u", "main.py"]
