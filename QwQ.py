import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from tqdm.auto import tqdm
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json  # JSON 파일 저장을 위한 모듈 추가
from PyPDF2 import PdfReader
import glob

def load_data():
    """데이터 로드 함수"""
    train = pd.read_csv('./data/train.csv', encoding='utf-8-sig')
    test = pd.read_csv('./data/test.csv', encoding='utf-8-sig')
    return train, test

def preprocess_data(df):
    """데이터 전처리 함수"""
    df['공사종류(대분류)'] = df['공사종류'].str.split(' / ').str[0]
    df['공사종류(중분류)'] = df['공사종류'].str.split(' / ').str[1]
    df['공종(대분류)'] = df['공종'].str.split(' > ').str[0]
    df['공종(중분류)'] = df['공종'].str.split(' > ').str[1]
    df['사고객체(대분류)'] = df['사고객체'].str.split(' > ').str[0]
    df['사고객체(중분류)'] = df['사고객체'].str.split(' > ').str[1]
    return df

def create_combined_data(df, is_train=True):
    """통합 데이터 생성 함수"""
    def create_question(row):
        question = (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        )
        return {"question": question, "answer": row["재발방지대책 및 향후조치계획"]} if is_train else {"question": question}
    
    combined_data = df.apply(create_question, axis=1)
    return pd.DataFrame(list(combined_data))

def print_gpu_memory():
    """GPU 메모리 사용량 출력 함수"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_properties(i).name}")
            print(f"  메모리 사용량: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")

def get_optimal_batch_size():
    """GPU 메모리에 따라 최적의 배치 크기 반환"""
    # 배치 크기 2로 고정
    return 4

def setup_model():
    """모델 설정 함수"""
    # Qwen2.5-72B-Instruct 모델로 변경
    # model_id = "Qwen/Qwen2.5-72B-Instruct"
    model_id = "Qwen/Qwen2.5-14B-Instruct"

    # 4비트 양자화 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        # quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    return tokenizer, model

def create_train_documents(combined_training_data, train_df):
    """개선된 학습 문서 생성 함수 - 메타데이터 활용"""
    train_questions = combined_training_data['question'].tolist()
    train_answers = combined_training_data['answer'].tolist()
    raw_documents = [f"Q: {q}\nA: {a}" for q, a in zip(train_questions, train_answers)]
    
    # 메타데이터에 분류 정보 추가
    metadatas = []
    for i, (q, a) in enumerate(zip(train_questions, train_answers)):
        meta = {
            "source": f"doc_{i}",
            "type": "QA",
            "question": q,
            "answer": a,
            "공사종류_대분류": train_df.iloc[i]['공사종류(대분류)'],
            "공사종류_중분류": train_df.iloc[i]['공사종류(중분류)'],
            "공종_대분류": train_df.iloc[i]['공종(대분류)'],
            "공종_중분류": train_df.iloc[i]['공종(중분류)'],
            "사고객체_대분류": train_df.iloc[i]['사고객체(대분류)'],
            "사고객체_중분류": train_df.iloc[i]['사고객체(중분류)'],
            "작업프로세스": train_df.iloc[i]['작업프로세스'],
            "사고원인": train_df.iloc[i]['사고원인']
        }
        metadatas.append(meta)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=240,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""],
        keep_separator=True
    )
    
    chunks = []
    chunk_metadatas = []
    
    for i, (doc, meta) in enumerate(zip(raw_documents, metadatas)):
        doc_chunks = text_splitter.split_text(doc)
        for j, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            chunk_meta = meta.copy()
            chunk_meta["chunk_id"] = j
            chunk_meta["total_chunks"] = len(doc_chunks)
            chunk_metadatas.append(chunk_meta)
    
    print(f"원본 문서 {len(raw_documents)}개에서 청킹 후 {len(chunks)}개 청크 생성됨")
    
    return chunks, chunk_metadatas

def setup_retriever(train_documents, train_metadatas):
    """리트리버 설정 함수 - Chroma 사용 (개선됨)"""
    print("Chroma 벡터 저장소 설정 중...")
    start_time = time.time()
    
    embedding = HuggingFaceEmbeddings(
        model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        model_kwargs={"device": "cuda"}
    )
    
    persist_directory = "./chroma_db_improved"
    
    if os.path.exists(persist_directory) and len(os.listdir(persist_directory)) > 0:
        print(f"기존 Chroma DB 로드 중: {persist_directory}")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding
        )
    else:
        print(f"새 Chroma DB 생성 중: {persist_directory}")
        ids = [f"chunk_{i}" for i in range(len(train_documents))]
        
        vector_store = Chroma.from_texts(
            texts=train_documents,
            metadatas=train_metadatas,
            embedding=embedding,
            persist_directory=persist_directory,
            ids=ids
        )
    
    end_time = time.time()
    print(f"Chroma 벡터 저장소 설정 완료! 소요 시간: {end_time - start_time:.2f}초")
    
    return vector_store.as_retriever(
        search_type="mmr", 
        search_kwargs={
            "k": 12,  # 7에서 증가
            "fetch_k": 35,  # 20에서 증가
            "lambda_mult": 0.6  # 0.7에서 감소
        }
    )

def setup_qa_chain(model, tokenizer, retriever):
    """QA 체인 설정 함수 - 최적화된 파이프라인 (개선됨)"""
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # 하이퍼파라미터 설정
    hyperparams = {
        "temperature": 0.3,  # 더 결정적인 응답을 위해 낮춤
        "top_p": 0.85,  # 약간 높임
        "top_k": 50,  # 관련성 높은 토큰에 더 집중
        "repetition_penalty": 1.5,  # 반복 방지 강화
        "max_new_tokens": 200,  # 더 긴 응답 허용
        "batch_size": 4  # 배치 크기 유지
    }
    
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=hyperparams["temperature"],
        top_p=hyperparams["top_p"],
        top_k=hyperparams["top_k"],
        repetition_penalty=hyperparams["repetition_penalty"],
        return_full_text=False,
        max_new_tokens=hyperparams["max_new_tokens"],
        batch_size=hyperparams["batch_size"],
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        device_map="auto"
    )
    
    # 개선된 상세 프롬프트 템플릿 - 범용 패턴 강제 포함
    prompt_template = """
    <|im_start|>system
    당신은 건설 현장 안전 관리 전문가로, 건설 사고 예방과 재발 방지를 위한 구체적인 대책을 제시하는 역할을 맡고 있습니다.

    답변 작성 지침:
    1. 모든 답변은 반드시 "작업 전 안전교육 실시와 안전관리자 안전점검 실시"를 첫 번째 항목으로 포함해야 합니다.
    2. 핵심적인 안전 대책만 간결하고 명확하게 나열하세요.
    3. 각 대책은 구체적이고 실행 가능해야 합니다.
    4. 서론, 배경 설명, 결론 등은 포함하지 마세요.
    5. "다음과 같은 조치를 취할 것을 제안합니다", "이러한 대책이 필요합니다" 등의 불필요한 문구는 사용하지 마세요.
    6. 모든 대책은 "-" 또는 "•" 기호로 시작하는 항목으로 나열하세요.
    7. 유사한 대책을 중복해서 제시하지 마세요.
    8. 작업 환경, 안전 장비, 교육, 관리 감독, 제도적 측면 등 다양한 관점에서 대책을 제시하세요.
    9. 제시된 사고 유형, 작업 프로세스, 사고 원인에 직접적으로 관련된 대책을 우선적으로 제시하세요.
    10. 모든 대책은 한국 건설 현장의 실정에 맞게 현실적이고 적용 가능해야 합니다.<|im_end|>

    <|im_start|>user
    아래 제공된 참고 자료를 기반으로 답변하세요:
    {context}

    질문:
    {question}<|im_end|>

    <|im_start|>assistant
    재발 방지 대책 및 향후 조치 계획:
    - 작업자 안전교육 실시와 안전관리자 안전점검 실시
    """

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    # 프롬프트 템플릿과 하이퍼파라미터를 함께 반환
    return qa_chain, prompt_template, hyperparams

class BatchInference:
    def __init__(self, pipeline_model, all_questions, batch_size):
        self.pipeline_model = pipeline_model
        self.all_questions = all_questions
        self.batch_size = batch_size
        self.batch_count = 0
        
    def __call__(self, examples):
        # 배치의 첫 번째 질문 출력
        if len(examples["prompt"]) > 0:
            question_idx = self.batch_count * self.batch_size
            if question_idx < len(self.all_questions):
                print("\n" + "="*80)
                print(f"질문 {question_idx+1}/{len(self.all_questions)}:")
                print(self.all_questions[question_idx])
        
        with torch.no_grad():
            outputs = self.pipeline_model(
                examples["prompt"],
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                top_k=50,
                repetition_penalty=1.2,
                max_new_tokens=128,
                return_full_text=False,
                batch_size=2
            )
        
        batch_results = []
        for i, output in enumerate(outputs):
            if isinstance(output, list) and len(output) > 0:
                generated_text = output[0]['generated_text']
                batch_results.append(generated_text)
                
                # 첫 번째 결과만 출력
                if i == 0:
                    print("\n[원본 응답]:")
                    print(generated_text)
                    
                    # 후처리된 결과 출력
                    processed_text = postprocess_results([generated_text])[0]
                    print("\n[후처리된 응답]:")
                    print(processed_text)
                    print("="*80 + "\n")
            else:
                batch_results.append("결과를 생성할 수 없습니다.")
                
                # 첫 번째 결과가 실패한 경우에만 출력
                if i == 0:
                    print("\n응답 생성 실패")
                    print("="*80 + "\n")
        
        # 배치 카운트 증가
        self.batch_count += 1
        
        torch.cuda.empty_cache()
        
        return {"result": batch_results}

def run_test(qa_chain, combined_test_data, test_df):
    """테스트 실행 함수 - 컨텍스트 처리 개선"""
    print(f"테스트 실행 시작... 총 테스트 샘플 수: {len(combined_test_data)}")
    print_gpu_memory()
    
    all_questions = combined_test_data['question'].tolist()
    
    pipeline_model = qa_chain.combine_documents_chain.llm_chain.llm.pipeline
    prompt_template = qa_chain.combine_documents_chain.llm_chain.prompt.template
    
    collection = qa_chain.retriever.vectorstore._collection
    embedding_func = qa_chain.retriever.vectorstore._embedding_function
    
    def get_context(idx_and_question):
        idx, question = idx_and_question
        query_embedding = embedding_func.embed_query(question)
        
        # 현재 질문에 해당하는 테스트 데이터 행
        current_test_row = test_df.iloc[idx]
        
        # 메타데이터 기반 필터링 조건 생성
        where = {
            "$or": [
                {"공사종류_대분류": current_test_row['공사종류(대분류)']},
                {"공사종류_중분류": current_test_row['공사종류(중분류)']},
                {"공종_대분류": current_test_row['공종(대분류)']},
                {"공종_중분류": current_test_row['공종(중분류)']},
                {"사고객체_대분류": current_test_row['사고객체(대분류)']},
                {"사고객체_중분류": current_test_row['사고객체(중분류)']},
                {"작업프로세스": current_test_row['작업프로세스']},
                {"사고원인": current_test_row['사고원인']}
            ]
        }
        
        results = collection.query(
            query_embeddings=[query_embedding],
            where=where,
            n_results=7,
            include=["documents", "metadatas", "distances"] 
        )
        
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        # 거리에 따른 가중치 적용
        weighted_docs = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            weight = 1.0 - min(dist, 0.99)  
            source_info = f"[출처: {meta.get('source', 'unknown')}]"
            category_info = f"[분류: {meta.get('공사종류_대분류', '')} > {meta.get('공종_대분류', '')} > {meta.get('사고객체_대분류', '')}]"
            weighted_docs.append(f"[관련도: {weight:.2f}] {source_info} {category_info}\n{doc}")
        
        return "\n\n".join(weighted_docs)
    
    print("병렬 컨텍스트 검색 중...")
    retrieval_start = time.time()
    
    import multiprocessing
    max_workers = min(16, multiprocessing.cpu_count() * 2)
    print(f"병렬 검색에 {max_workers}개의 스레드 사용")
    
    # 인덱스와 질문을 함께 전달
    indexed_questions = list(enumerate(all_questions))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_contexts = list(tqdm(
            executor.map(get_context, indexed_questions),
            total=len(all_questions),
            desc="컨텍스트 검색"
        ))
    
    retrieval_time = time.time() - retrieval_start
    print(f"컨텍스트 검색 완료! 소요 시간: {retrieval_time:.2f}초")
    
    print("프롬프트 생성 중...")
    prompt_start = time.time()
    
    all_prompts = []
    for question, context in zip(all_questions, all_contexts):
        prompt = prompt_template.replace("{context}", context).replace("{question}", question)
        all_prompts.append(prompt)
    
    prompt_time = time.time() - prompt_start
    print(f"프롬프트 생성 완료! 소요 시간: {prompt_time:.2f}초")
    
    print("Dataset 생성 및 배치 추론 준비 중...")
    dataset = Dataset.from_dict({"prompt": all_prompts})
    
    print("배치 추론 실행 중...")
    inference_start = time.time()
    
    batch_size = get_optimal_batch_size()
    print(f"추론에 배치 크기 {batch_size} 사용")
    
    # BatchInference 클래스 인스턴스 생성
    batch_inferencer = BatchInference(pipeline_model, all_questions, batch_size)
    
    results = dataset.map(
        batch_inferencer,
        batched=True,
        batch_size=batch_size,
        desc="추론",
        remove_columns=["prompt"]
    )
    
    inference_time = time.time() - inference_start
    print(f"배치 추론 완료! 소요 시간: {inference_time:.2f}초")
    
    test_results = results["result"]
    
    print("결과 후처리 중...")
    test_results = postprocess_results(test_results)
    
    total_time = retrieval_time + prompt_time + inference_time
    avg_time_per_sample = total_time / len(all_questions)
    print(f"\n테스트 실행 완료! 총 결과 수: {len(test_results)}")
    print(f"총 처리 시간: {total_time:.2f}초, 샘플당 평균 시간: {avg_time_per_sample:.2f}초")
    print_gpu_memory()
    
    return test_results

def postprocess_results(results):
    processed_results = []
    
    for result in results:
        # 기존 정제 로직 유지
        result = re.sub(r'^(다음과 같은|다음과|이에 대한|이를 위한|재발 방지를 위한|안전사고 예방을 위한)', '', result)
        result = re.sub(r'(조치를 취할 것을 제안합니다|조치가 필요합니다)[:.]?', '', result)
        result = re.sub(r'\n+', '\n', result)
        result = re.sub(r'^\d+\.\s*', '', result, flags=re.MULTILINE)
        
        # 더 강화된 문장 처리
        sentences = []
        for line in result.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('-') or line.startswith('•'):
                # 불필요한 접두사 제거
                line = re.sub(r'^[•-]\s*', '- ', line)
                sentences.append(line)
            else:
                # 문장 분리 로직 개선
                for sentence in re.split(r'(?<=[.!?])\s+', line):
                    if sentence.strip():
                        # 문장이 너무 짧으면 건너뛰기
                        if len(sentence.strip()) > 10:
                            sentences.append(f"- {sentence.strip()}")
        
        # 중복 제거 로직 강화
        unique_sentences = []
        seen = set()
        for sentence in sentences:
            # 더 정교한 정규화
            normalized = re.sub(r'[^\w\s]', '', sentence.lower())
            normalized = re.sub(r'\s+', ' ', normalized).strip()
            
            # 유사도 기반 중복 제거 (간단한 구현)
            is_duplicate = False
            for existing in seen:
                # 80% 이상 유사하면 중복으로 간주
                if len(set(normalized.split()) & set(existing.split())) / max(len(set(normalized.split())), len(set(existing.split()))) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate and len(normalized) > 10:
                seen.add(normalized)
                unique_sentences.append(sentence)
        
        # 문장 정렬 - 짧은 문장을 먼저 배치
        unique_sentences.sort(key=lambda x: len(x))
        
        final_result = "\n".join(unique_sentences)
        processed_results.append(final_result)
    
    return processed_results

def create_embeddings(test_results):
    """임베딩 생성 함수 - 원래 모델 유지"""
    embedding_model = SentenceTransformer("jhgan/ko-sbert-sts")
    
    batch_size = min(64, len(test_results) // 10 + 1) 
    
    # 배치 처리로 임베딩 생성
    print(f"임베딩 생성 중... 배치 크기: {batch_size}")
    start_time = time.time()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = embedding_model.to(device)
    
    pred_embeddings = embedding_model.encode(
        test_results,
        batch_size=batch_size,
        show_progress_bar=True,
        device=device
    )
    
    end_time = time.time()
    print(f"임베딩 생성 완료! 소요 시간: {end_time - start_time:.2f}초")
    print(f"임베딩 형태: {pred_embeddings.shape}")
    
    return pred_embeddings

def save_results(test_results, pred_embeddings, model_id, prompt_template, hyperparams):
    """결과 저장 함수"""
    submission = pd.read_csv('./data/sample_submission.csv', encoding='utf-8-sig')
    submission.iloc[:,1] = test_results
    submission.iloc[:,2:] = pred_embeddings
    
    from datetime import datetime
    current_time = datetime.now().strftime('%y%m%d_%H%M')
    
    # 모델 ID에서 모델명 추출
    model_name = model_id.split('/')[-1]
    
    # 파일 이름 기본 형식 (확장자 제외)
    base_filename = f'./submissions/submission_{current_time}_{model_name}'
    
    # CSV 파일 저장
    csv_filename = f'{base_filename}.csv'
    submission.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    # 프롬프트와 하이퍼파라미터를 JSON 파일로 저장
    config_data = {
        "model_id": model_id,
        "timestamp": current_time,
        "prompt_template": prompt_template,
        "hyperparameters": hyperparams,
        "batch_size": hyperparams.get("batch_size", 2),
        "temperature": hyperparams.get("temperature", 0.3),
        "top_p": hyperparams.get("top_p", 0.85),
        "top_k": hyperparams.get("top_k", 50),
        "repetition_penalty": hyperparams.get("repetition_penalty", 1.2),
        "max_new_tokens": hyperparams.get("max_new_tokens", 128)
    }
    
    json_filename = f'{base_filename}.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    print(f"결과가 '{csv_filename}'에 저장되었습니다.")
    print(f"설정 정보가 '{json_filename}'에 저장되었습니다.")

def load_markdown_files(md_directory):
    """마크다운 파일 로드 함수 - 각 파일을 하나의 청크로 유지"""
    print(f"마크다운 파일 로드 중... 경로: {md_directory}")
    
    md_chunks = []
    md_chunk_metadatas = []
    
    for root, dirs, files in os.walk(md_directory):
        # 현재 문서 이름 (폴더명)
        doc_name = os.path.basename(root)
        if not doc_name:  # 루트 디렉토리 스킵
            continue
            
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if text:  # 빈 파일 제외
                        # 페이지 번호 추출 (파일명에서)
                        page_num = int(file.split('.')[0]) if file.split('.')[0].isdigit() else 0
                        
                        # 각 파일을 하나의 청크로 처리
                        md_chunks.append(text)
                        md_chunk_metadatas.append({
                            "source": doc_name,
                            "type": "Markdown",
                            "page": page_num,
                            "path": file_path,
                            "chunk_id": 0,  # 파일당 하나의 청크
                            "total_chunks": 1
                        })
                        
                except Exception as e:
                    print(f"파일 처리 중 오류 발생: {file_path}, 오류: {str(e)}")
    
    print(f"마크다운 파일 로드 완료! 총 {len(md_chunks)}개의 청크 생성됨")
    return md_chunks, md_chunk_metadatas

def main():
    print("프로그램 시작...")
    total_start_time = time.time()
    
    print("데이터 로드 및 전처리 중...")
    train, test = load_data()
    train = preprocess_data(train)
    test = preprocess_data(test)
    
    print("통합 데이터 생성 중...")
    combined_training_data = create_combined_data(train, is_train=True)
    combined_test_data = create_combined_data(test, is_train=False)
    
    print("모델 및 토크나이저 설정 중...")
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    tokenizer, model = setup_model()
    
    print("학습 문서 생성 중...")
    train_documents, train_metadatas = create_train_documents(combined_training_data, train)
    
    # 마크다운 파일 로드 및 처리 - 청킹 단계 제거
    md_directory = "./data/건설안전지침_md"
    md_chunks, md_chunk_metadatas = load_markdown_files(md_directory)
    
    print(f"문서 통합 중... (학습 문서: {len(train_documents)}개, 마크다운: {len(md_chunks)}개)")
    
    # 기존 학습 문서와 마크다운 문서 결합
    all_documents = train_documents + md_chunks
    all_metadatas = train_metadatas + md_chunk_metadatas
    
    print(f"총 {len(all_documents)}개의 문서로 벡터 저장소 구성")
    
    # 벡터 저장소 설정 (모든 문서 포함)
    retriever = setup_retriever(all_documents, all_metadatas)
    
    print("QA 체인 설정 중...")
    qa_chain, prompt_template, hyperparams = setup_qa_chain(model, tokenizer, retriever)
    
    print("테스트 실행 중...")
    test_results = run_test(qa_chain, combined_test_data, test)
    
    print("임베딩 생성 중...")
    pred_embeddings = create_embeddings(test_results)
    
    print("결과 저장 중...")
    os.makedirs('./submissions', exist_ok=True)
    save_results(test_results, pred_embeddings, model_id, prompt_template, hyperparams)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"프로그램 완료! 총 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")

if __name__ == "__main__":
    main()