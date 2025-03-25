import pandas as pd
import torch
import os
import gc
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from tqdm import tqdm
import subprocess

# RTX 4000 시리즈를 위한 환경 변수 설정
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 설정 - 메모리 관리 및 GPU 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 모든 GPU 사용

def clean_memory():
    """메모리 정리 함수"""
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(i)
    logger.info(f"메모리 정리 완료")

def check_gpu_status():
    """GPU 상태 확인 함수"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"사용 가능한 GPU 수: {gpu_count}")
        for i in range(gpu_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GiB")
            logger.info(f"현재 사용 중인 메모리: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GiB")
    else:
        logger.warning("GPU를 찾을 수 없습니다.")

def load_and_preprocess_data(file_path='./data/train.csv', sample_size=None):
    """데이터 로드 및 전처리 함수"""
    logger.info(f"데이터 로드 중: {file_path}")
    train_df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    # 샘플 크기 제한 (메모리 테스트용)
    if sample_size and sample_size < len(train_df):
        train_df = train_df.sample(sample_size, random_state=42)
        logger.info(f"샘플 데이터 사용: {len(train_df)}개")
    
    # 데이터 전처리
    train_df['공사종류(대분류)'] = train_df['공사종류'].str.split(' / ').str[0]
    train_df['공사종류(중분류)'] = train_df['공사종류'].str.split(' / ').str[1]
    train_df['공종(대분류)'] = train_df['공종'].str.split(' > ').str[0]
    train_df['공종(중분류)'] = train_df['공종'].str.split(' > ').str[1]
    train_df['사고객체(대분류)'] = train_df['사고객체'].str.split(' > ').str[0]
    train_df['사고객체(중분류)'] = train_df['사고객체'].str.split(' > ').str[1]
    
    logger.info(f"데이터 전처리 완료: {len(train_df)}개 샘플")
    return train_df

def create_instruction_dataset(train_df, chunk_size=1000):
    """효율적인 데이터셋 생성 (청크 단위로 처리)"""
    logger.info("지시 데이터셋 생성 중...")
    all_instruction_data = []
    
    # 청크 단위로 처리
    num_chunks = (len(train_df) + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(train_df))
        chunk_df = train_df.iloc[start_idx:end_idx]
        
        chunk_data = []
        for _, row in tqdm(chunk_df.iterrows(), total=len(chunk_df), 
                         desc=f"청크 {chunk_idx+1}/{num_chunks} 처리 중"):
            # 질문 생성
            question = (
                f"{row['공사종류(대분류)']} {row['공사종류(중분류)']} 공사의 {row['공종(대분류)']} {row['공종(중분류)']} 작업 중 "
                f"{row['사고객체(대분류)']}({row['사고객체(중분류)']}) 관련 사고가 발생했습니다. "
                f"작업 중 '{row['작업프로세스']}' 과정에서 '{row['사고원인']}' 원인으로 사고가 발생했습니다. "
                f"이러한 사고의 재발을 방지하기 위한 구체적인 대책과 향후 조치 방안을 간략히 제시해 주세요."
            )
            
            # 응답
            answer = row["재발방지대책 및 향후조치계획"]
            
            # 지시 형식 데이터
            instruction = {
                "instruction": question,
                "input": "",
                "output": answer
            }
            
            chunk_data.append(instruction)
        
        all_instruction_data.extend(chunk_data)
        clean_memory()  # 각 청크 처리 후 메모리 정리
    
    # 데이터셋 생성
    dataset = Dataset.from_list(all_instruction_data)
    logger.info(f"지시 데이터셋 생성 완료: {len(dataset)}개 샘플")
    return dataset

def format_instruction(example):
    """지시 형식 프롬프트 포맷팅"""
    instruction = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 건설 현장의 안전 관리 전문가입니다. 건설 사고에 대한 구체적이고 실용적인 재발 방지 대책을 제시해야 합니다.

대답 형식 지침:
1. 서론이나 배경 설명 없이 곧바로 핵심 대책을 제시하세요.
2. 재발 방지 대책은 최대한 간결하게 작성하세요.
3. 답변은 '~설치', '~교육', '~점검', '~마련' 등의 명사형으로 끝내세요.
4. 불필요한 반복이나 추상적인 표현을 피하세요.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
{example['instruction']}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
{example['output']}<|eot_id|>
"""
    return {"formatted_instruction": instruction}

def tokenize_dataset(dataset, tokenizer, batch_size=32):
    """데이터셋 토크나이징 (배치 단위로 효율적으로 처리)"""
    logger.info("데이터셋 토크나이징 중...")
    
    # 먼저 format_instruction 적용
    dataset = dataset.map(
        format_instruction,
        batched=False,
        desc="프롬프트 포맷팅 중",
        num_proc=4  # 멀티프로세싱으로 속도 향상
    )
    
    def tokenize_function(examples):
        results = tokenizer(
            examples["formatted_instruction"],
            truncation=True,
            max_length=1024,
            padding=False,
            return_tensors=None,
        )
        
        # 입력 ID와 어텐션 마스크 설정
        examples["input_ids"] = results["input_ids"]
        examples["attention_mask"] = results["attention_mask"]
        
        # 라벨 설정
        examples["labels"] = examples["input_ids"].copy()
        
        return examples
    
    # 배치로 토크나이징 적용
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        desc="토크나이징 중",
        remove_columns=["instruction", "input", "output", "formatted_instruction"],
        num_proc=1  # 토크나이징은 단일 프로세스로 (GPU 메모리 관리)
    )
    
    logger.info("토크나이징 완료")
    return tokenized_dataset

def load_model_for_training_multi_gpu(model_id="SEOKDONG/llama3.1_korean_v1.1_sft_by_aidx"):
    """여러 GPU에 모델 분산 로드"""
    logger.info(f"모델 로드 중: {model_id}")
    
    # GPU 상태 확인
    check_gpu_status()
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 메모리 초기화
    clean_memory()
    
    # 여러 GPU에 모델 분산을 위한 device_map 설정
    device_map = "auto"  # 자동으로 GPU 분배
    
    # 4비트 양자화 설정
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # 모델 로드 - 멀티 GPU 설정 (FlashAttention 제거)
    logger.info("멀티 GPU로 모델 로드 시도...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,  # auto로 설정하여 여러 GPU에 분산
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
        )
        logger.info("모델이 여러 GPU에 분산 로드됨")
        
        # 모델이 어떻게 분산되었는지 확인
        if hasattr(model, 'hf_device_map'):
            logger.info("모델 분산 맵:")
            for layer, device in model.hf_device_map.items():
                logger.info(f"{layer}: {device}")
    
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {e}")
        raise RuntimeError("모델 로드 실패")
    
    # 학습을 위한 모델 준비
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # 그래디언트 체크포인팅 활성화
    model.gradient_checkpointing_enable()
    
    # 훈련 가능한 파라미터 비율 계산 (변경 전)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"총 파라미터 수: {total_params:,}")
    
    # LoRA 설정 - 더 많은 모듈에 적용
    target_modules = [
        # 어텐션 모듈
        "q_proj", "k_proj", "v_proj", "o_proj", 
        # MLP 모듈
        "gate_proj", "up_proj", "down_proj"
    ]
    
    logger.info(f"LoRA 적용 대상 모듈: {target_modules}")
    
    lora_config = LoraConfig(
        r=16,                      # 랭크 값 증가 (8 -> 16)
        lora_alpha=32,             # 알파 값 증가 (16 -> 32)
        target_modules=target_modules,
        lora_dropout=0.1,          # 드롭아웃 증가 (0.05 -> 0.1)
        bias="lora_only",          # 바이어스 학습 활성화
        task_type="CAUSAL_LM"
    )
    
    # LoRA 적용
    model = get_peft_model(model, lora_config)
    
    # 훈련 가능한 파라미터 출력
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model_distributed(model, tokenizer, tokenized_dataset, output_dir="./models/llama_construction_safety_finetune"):
    """분산 학습 실행"""
    logger.info("분산 학습 설정 초기화 중...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 다중 GPU 학습 설정 최적화
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,                # 에폭 수 증가 (3 -> 5)
        per_device_train_batch_size=1,     # 배치 크기 유지
        gradient_accumulation_steps=8,     # 누적 단계 감소 (8 -> 4)
        warmup_ratio=0.05,                 # 웜업 비율 증가 (0.03 -> 0.05)
        learning_rate=5e-5,                # 학습률 증가 (2e-5 -> 5e-5)
        weight_decay=0.01,
        max_grad_norm=1.0,                 # 그래디언트 클리핑 값 증가
        fp16=True,
        logging_steps=5,                   # 더 자주 로깅 (10 -> 5)
        save_steps=100,                    # 더 자주 저장 (200 -> 100)
        save_total_limit=3,
        report_to="tensorboard",
        remove_unused_columns=False,
        
        # 분산 학습 설정
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        
        # 성능 최적화
        optim="adamw_torch_fused",
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        
        # RTX 4000 시리즈를 위한 통신 설정
        local_rank=-1,  # 수동 DDP 비활성화
    )
    
    # 데이터 콜레이터 설정
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # 트레이너 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 학습 실행
    logger.info("분산 학습 시작...")
    clean_memory()  # 학습 전 메모리 정리
    
    # 학습
    trainer.train()
    
    # 모델 저장
    logger.info(f"학습 완료, 모델 저장 중: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"모델 학습 및 저장 완료!")
    return output_dir

def finetune_distributed(data_path='./data/train.csv', model_id="SEOKDONG/llama3.1_korean_v1.1_sft_by_aidx", test_size=500):
    """분산 GPU로 파인튜닝 실행"""
    logger.info("===== 멀티 GPU 파인튜닝 시작 =====")
    
    # GPU 확인
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"사용 가능한 GPU 수: {gpu_count}")
        for i in range(gpu_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        if gpu_count < 2:
            logger.warning("분산 학습에는 2개 이상의 GPU가 권장됩니다.")
    else:
        logger.error("GPU를 찾을 수 없습니다. 분산 학습이 불가능합니다.")
        return
    
    # 1. 데이터 로드 및 전처리
    logger.info("1단계: 데이터 로드")
    train_df = load_and_preprocess_data(data_path)
    
    # 2. 소규모 테스트 데이터셋으로 먼저 테스트 (선택사항)
    logger.info(f"2단계: 테스트 데이터셋 생성 (샘플 크기: {test_size})")
    test_df = train_df.sample(min(test_size, len(train_df)), random_state=42)
    test_dataset = create_instruction_dataset(test_df)
    
    # 3. 모델 및 토크나이저 로드 (여러 GPU에 분산)
    logger.info("3단계: 멀티 GPU에 모델 및 토크나이저 로드")
    model, tokenizer = load_model_for_training_multi_gpu(model_id)
    
    # 4. 테스트 데이터 토크나이징
    logger.info("4단계: 테스트 데이터 토크나이징")
    test_tokenized_dataset = tokenize_dataset(test_dataset, tokenizer)
    
    # 5. 테스트 분산 학습 실행
    logger.info("5단계: 테스트 분산 학습 실행")
    test_output_dir = train_model_distributed(model, tokenizer, test_tokenized_dataset, 
                                            output_dir="./models/test_finetune")
    
    # 테스트 성공 시 전체 데이터 학습으로 진행
    logger.info("테스트 학습 성공! 전체 데이터셋으로 진행")
    
    # 메모리 정리
    del model, tokenizer, test_dataset, test_tokenized_dataset
    clean_memory()
    
    # 6. 전체 데이터셋 생성
    logger.info("6단계: 전체 데이터셋 생성")
    full_dataset = create_instruction_dataset(train_df)
    
    # 7. 모델 다시 로드 (여러 GPU에 분산)
    logger.info("7단계: 멀티 GPU에 모델 다시 로드")
    full_model, full_tokenizer = load_model_for_training_multi_gpu(model_id)
    
    # 8. 전체 데이터 토크나이징
    logger.info("8단계: 전체 데이터 토크나이징")
    full_tokenized_dataset = tokenize_dataset(full_dataset, full_tokenizer)
    
    # 9. 전체 분산 학습 실행
    logger.info("9단계: 전체 데이터 분산 학습 실행")
    final_output_dir = train_model_distributed(full_model, full_tokenizer, full_tokenized_dataset, 
                                             output_dir="./models/llama_construction_safety_finetune")
    
    logger.info(f"===== 멀티 GPU 파인튜닝 성공적으로 완료! 최종 모델: {final_output_dir} =====")
    return final_output_dir

if __name__ == "__main__":
    try:
        # 환경 변수 확인
        logger.info(f"NCCL_P2P_DISABLE: {os.environ.get('NCCL_P2P_DISABLE', '설정되지 않음')}")
        logger.info(f"NCCL_IB_DISABLE: {os.environ.get('NCCL_IB_DISABLE', '설정되지 않음')}")
        
        # GPU 확인
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"사용 가능한 GPU 수: {gpu_count}")
            for i in range(gpu_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GiB")
        else:
            logger.error("GPU를 찾을 수 없습니다.")
            exit(1)
        
        # 메모리 정리 및 GPU 설정 확인
        subprocess.run("nvidia-smi", shell=True)
        clean_memory()
        
        # 분산 파인튜닝 실행
        finetune_distributed(test_size=500)  # 테스트 크기 조정 가능
        
    except Exception as e:
        logger.error(f"파인튜닝 중 오류 발생: {e}", exc_info=True) 