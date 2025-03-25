import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
from datasets import Dataset
from tqdm.auto import tqdm
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

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
            f"{row['공사종류(대분류)']} {row['공사종류(중분류)']} 공사의 {row['공종(대분류)']} {row['공종(중분류)']} 작업 중 "
            f"{row['사고객체(대분류)']}({row['사고객체(중분류)']}) 관련 사고가 발생했습니다. "
            f"작업 중 '{row['작업프로세스']}' 과정에서 '{row['사고원인']}' 원인으로 사고가 발생했습니다. "
            f"이러한 사고의 재발을 방지하기 위한 간략한 대책과 향후 조치 방안을 간략히 제시해 주세요."
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
    return 4

def setup_model():
    """모델 설정 함수 - 파인튜닝된 모델 사용"""
    # 파인튜닝된 모델 경로
    model_id = "./models/llama_construction_safety_finetune"
    
    # 기존 모델 사용 (파인튜닝되지 않았을 경우에 대비)
    if not os.path.exists(model_id):
        model_id = "SEOKDONG/llama3.1_korean_v1.1_sft_by_aidx"
        print(f"파인튜닝된 모델을 찾을 수 없습니다. 기본 모델을 사용합니다: {model_id}")
    else:
        print(f"파인튜닝된 모델을 사용합니다: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    return tokenizer, model

def create_train_documents(combined_training_data, train_df):
    """개선된 학습 문서 생성 함수 - 메타데이터 활용"""
    train_questions = combined_training_data['question'].tolist()
    train_answers = combined_training_data['answer'].tolist()
    raw_documents = [f"Q: {q}\nA: {a}" for q, a in zip(train_questions, train_answers)]
    
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
            "k": 12,
            "fetch_k": 35,
            "lambda_mult": 0.6
        }
    )

def create_optimized_prompt(question, context, metadata=None):
    """질문 구성 변경에 맞추어 최적화된 프롬프트 생성 함수"""
    
    # 메타데이터가 제공된 경우 활용
    if metadata:
        construction_type = f"{metadata.get('공사종류_대분류', '')} {metadata.get('공사종류_중분류', '')}"
        work_type = f"{metadata.get('공종_대분류', '')} {metadata.get('공종_중분류', '')}"
        accident_object = f"{metadata.get('사고객체_대분류', '')}({metadata.get('사고객체_중분류', '')})"
        work_process = metadata.get('작업프로세스', '')
        accident_cause = metadata.get('사고원인', '')
        
        situation_context = f"""사고 상황 정보:
- 공사 유형: {construction_type}
- 작업 종류: {work_type}
- 사고 객체: {accident_object}
- 작업 프로세스: {work_process}
- 사고 원인: {accident_cause}
"""
    else:
        situation_context = ""

    # 간결하고 실용적인 응답을 유도하는 프롬프트 템플릿
    prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 건설 현장의 안전 관리 전문가입니다. 건설 사고에 대한 실용적이고 간략한 재발 방지 대책을 제시해야 합니다.

{situation_context}
대답 형식 지침:
1. 서론이나 배경 설명 없이 곧바로 핵심 대책을 제시하세요.
2. 재발 방지 대책은 최대한 간결하게 작성하세요.
3. 답변은 '~설치', '~교육', '~점검', '~마련' 등의 명사형으로 끝내세요.
4. 불필요한 반복이나 추상적인 표현을 피하세요.
5. "다음과 같은 조치를 취할 것을 제안합니다"와 같은 표현은 사용하지 마세요.
6. 번호 매김이나 목록 형태로 작성하지 마세요.
7. 답변 끝에 반드시 마침표를 포함하세요.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
다음 참고 자료를 기반으로 답변하세요:
{context}

질문:
{question}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
    
    return prompt_template

def setup_qa_chain(model, tokenizer, retriever):
    """QA 체인 설정 함수 - 최적화된 파이프라인 (개선됨)"""
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    hyperparams = {
        "temperature": 0.5,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.15,
        "max_new_tokens": 250,
        "batch_size": 4
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
    
    # 기본 프롬프트 템플릿 (이후 동적으로 대체됨)
    basic_prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
당신은 건설 현장 안전 관리 전문가로, 건설 사고 예방과 재발 방지를 위한 대책을 제시하는 역할을 맡고 있습니다.
모든 답변은 반드시 한국어로만 작성해야 합니다.<|eot_id|><|start_header_id|>user<|end_header_id|>
아래 제공된 참고 자료를 기반으로 답변하세요:
{context}

질문:
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

재발 방지 대책 및 향후 조치는 다음과 같습니다:

"""

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    prompt = PromptTemplate(input_variables=["context", "question"], template=basic_prompt_template)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain, basic_prompt_template, hyperparams

class BatchInference:
    def __init__(self, pipeline_model, all_questions, batch_size):
        self.pipeline_model = pipeline_model
        self.all_questions = all_questions
        self.batch_size = batch_size
        self.batch_count = 0
        
    def __call__(self, examples):
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
                temperature=0.5,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.15,
                max_new_tokens=100,
                return_full_text=False,
                batch_size=2
            )
        
        batch_results = []
        for i, output in enumerate(outputs):
            if isinstance(output, list) and len(output) > 0:
                generated_text = output[0]['generated_text']
                batch_results.append(generated_text)
                
                if i == 0:
                    print("\n[원본 응답]")
                    print(generated_text)
                    
                    processed_text = postprocess_results([generated_text])[0]
                    print("\n[후처리된 응답]")
                    print(processed_text)
                    print("="*80 + "\n")
            else:
                batch_results.append("결과를 생성할 수 없습니다.")
                
                if i == 0:
                    print("\n응답 생성 실패")
                    print("="*80 + "\n")
        
        self.batch_count += 1
        
        torch.cuda.empty_cache()
        
        return {"result": batch_results}

def run_test(qa_chain, combined_test_data, test_df):
    """테스트 실행 함수 - 컨텍스트 처리 및 프롬프트 최적화"""
    print(f"테스트 실행 시작... 총 테스트 샘플 수: {len(combined_test_data)}")
    print_gpu_memory()
    
    all_questions = combined_test_data['question'].tolist()
    
    pipeline_model = qa_chain.combine_documents_chain.llm_chain.llm.pipeline
    
    collection = qa_chain.retriever.vectorstore._collection
    embedding_func = qa_chain.retriever.vectorstore._embedding_function
    
    def get_context(idx_and_question):
        idx, question = idx_and_question
        query_embedding = embedding_func.embed_query(question)
        
        current_test_row = test_df.iloc[idx]
        
        # 향상된 메타데이터 필터링
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
        
        weighted_docs = []
        for doc, meta, dist in zip(docs, metadatas, distances):
            weight = 1.0 - min(dist, 0.99)  
            source_info = f"[출처: {meta.get('source', 'unknown')}]"
            category_info = f"[분류: {meta.get('공사종류_대분류', '')} > {meta.get('공종_대분류', '')} > {meta.get('사고객체_대분류', '')}]"
            weighted_docs.append(f"[관련도: {weight:.2f}] {source_info} {category_info}\n{doc}")
        
        return "\n\n".join(weighted_docs), current_test_row
    
    print("병렬 컨텍스트 검색 중...")
    retrieval_start = time.time()
    
    import multiprocessing
    max_workers = min(16, multiprocessing.cpu_count() * 2)
    print(f"병렬 검색에 {max_workers}개의 스레드 사용")
    
    indexed_questions = list(enumerate(all_questions))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        search_results = list(tqdm(
            executor.map(get_context, indexed_questions),
            total=len(all_questions),
            desc="컨텍스트 검색"
        ))
    
    all_contexts = [result[0] for result in search_results]
    test_rows = [result[1] for result in search_results]
    
    retrieval_time = time.time() - retrieval_start
    print(f"컨텍스트 검색 완료! 소요 시간: {retrieval_time:.2f}초")
    
    print("최적화된 프롬프트 생성 중...")
    prompt_start = time.time()
    
    all_prompts = []
    for idx, (question, context, test_row) in enumerate(zip(all_questions, all_contexts, test_rows)):
        # 메타데이터 딕셔너리 생성
        metadata = {
            "공사종류_대분류": test_row['공사종류(대분류)'],
            "공사종류_중분류": test_row['공사종류(중분류)'],
            "공종_대분류": test_row['공종(대분류)'],
            "공종_중분류": test_row['공종(중분류)'],
            "사고객체_대분류": test_row['사고객체(대분류)'],
            "사고객체_중분류": test_row['사고객체(중분류)'],
            "작업프로세스": test_row['작업프로세스'],
            "사고원인": test_row['사고원인']
        }
        
        # 최적화된 프롬프트 생성
        optimized_prompt = create_optimized_prompt(question, context, metadata)
        all_prompts.append(optimized_prompt)
    
    prompt_time = time.time() - prompt_start
    print(f"프롬프트 생성 완료! 소요 시간: {prompt_time:.2f}초")
    
    print("Dataset 생성 및 배치 추론 준비 중...")
    dataset = Dataset.from_dict({"prompt": all_prompts})
    
    print("배치 추론 실행 중...")
    inference_start = time.time()
    
    batch_size = get_optimal_batch_size()
    print(f"추론에 배치 크기 {batch_size} 사용")
    
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
        # Llama 3.1 특수 토큰 제거
        result = re.sub(r'<\|begin_of_text\|>', '', result)
        result = re.sub(r'<\|end_of_text\|>', '', result)
        result = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', result)
        result = re.sub(r'<\|eot_id\|>', '', result)
        result = re.sub(r'<\|eom_id\|>', '', result)
        result = re.sub(r'<\|python_tag\|>', '', result)
        result = re.sub(r'<\|finetune_right_pad_id\|>', '', result)
        
        # "답변:" 다음 부분만 추출
        answer_pattern = r'답변:(.+)$'
        answer_match = re.search(answer_pattern, result, re.DOTALL)
        
        if answer_match:
            result = answer_match.group(1).strip()
        
        # 서론 및 결론 관련 표현 제거
        result = re.sub(r'^재발 방지 대책 및 향후 조치는 다음과 같습니다:?', '', result)
        result = re.sub(r'^이러한 사고의 재발 방지를 위한 구체적인 대책과 향후 조치 방안은 다음과 같습니다:?', '', result)
        result = re.sub(r'위와 같은 조치들을 통해 유사한 사고.*?$', '', result, flags=re.MULTILINE)
        result = re.sub(r'위 대책들을 철저히 이행하면.*?$', '', result, flags=re.MULTILINE)
        result = re.sub(r'이러한 조치들을 통해 안전한 작업환경을 조성.*?$', '', result, flags=re.MULTILINE)
        result = re.sub(r'이러한 대책과 조치를 통해 유사 사고.*?$', '', result, flags=re.MULTILINE)
        
        # 일반적인 정제 - 더 자연스러운 질문에 맞게 추가 패턴 정의
        result = re.sub(r'^(다음과 같은|다음과|이에 대한|이를 위한|재발 방지를 위한|안전사고 예방을 위한)', '', result)
        result = re.sub(r'(조치를 취할 것을 제안합니다|조치가 필요합니다|방안을 마련해야 합니다)[:.]?', '', result)
        result = re.sub(r'구체적인 대책으로는', '', result)
        result = re.sub(r'주요 대책은', '', result)
        
        # 추가적인 서론 표현 제거
        result = re.sub(r'해당 사고의 재발을 방지하기 위해서는', '', result)
        result = re.sub(r'이러한 사고를 예방하기 위해서는', '', result)
        result = re.sub(r'사고 재발 방지를 위해', '', result)
        
        # "답변 종료" 및 유사 문구 제거
        result = re.sub(r'답변 종료\.?', '', result)
        result = re.sub(r'이상입니다\.?', '', result)
        
        # 번호 및 불릿 포인트 제거 - 더 다양한 패턴 지원
        numbered_pattern = r'^\d+[\.\)]\s+'
        result = re.sub(numbered_pattern, '', result)
        result = re.sub(r'^[•\-\*]\s+', '', result)
        
        # 다중 공백 제거
        result = re.sub(r'\s+', ' ', result).strip()
        
        # 새로운 코드: 응답을 문장 단위로 분리하고 마지막 문장이 온점으로 끝나지 않으면 제거
        sentences = re.split(r'(?<=[.!?])\s+', result)
        
        # 마지막 문장이 온점으로 끝나지 않으면 제거
        if sentences and not sentences[-1].strip().endswith('.'):
            print(f"[INFO] 불완전한 마지막 문장 제거: '{sentences[-1]}'")
            sentences = sentences[:-1]
        
        # 문장이 하나도 남지 않으면 기본 응답 사용
        if not sentences:
            result = "작업전 안전교육 강화 및 작업장 위험요소 점검을 통한 재발 방지와 안전관리 교육 철저를 통한 향후 조치 계획."
        else:
            # 남은 문장들을 다시 합침
            result = ' '.join(sentences)
        
        # 짧은 응답 처리
        if len(result) < 10:
            result = "작업전 안전교육 강화 및 작업장 위험요소 점검을 통한 재발 방지와 안전관리 교육 철저를 통한 향후 조치 계획."
            
        processed_results.append(result)
    
    return processed_results

def create_embeddings(test_results):
    """임베딩 생성 함수 - 원래 모델 유지"""
    embedding_model = SentenceTransformer("jhgan/ko-sbert-sts")
    
    batch_size = min(64, len(test_results) // 10 + 1) 
    
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

def save_results(test_results, pred_embeddings, model_id, prompt_template, hyperparams, combined_test_data=None):
    """결과 저장 함수 - 임베딩 파일과 질문+답변 파일 두 가지 생성"""
    # 원래 제출 파일 저장 (임베딩 포함)
    submission = pd.read_csv('./data/sample_submission.csv', encoding='utf-8-sig')
    submission.iloc[:,1] = test_results
    submission.iloc[:,2:] = pred_embeddings
    
    from datetime import datetime
    current_time = datetime.now().strftime('%y%m%d_%H%M')
    
    model_name = model_id.split('/')[-1]
    
    base_filename = f'./submissions/submission_{current_time}_{model_name}'
    
    csv_filename = f'{base_filename}.csv'
    submission.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    
    # 추가: 질문과 답변만 있는 파일 저장
    if combined_test_data is not None:
        qa_df = pd.DataFrame({
            'ID': submission.iloc[:,0],
            'question': combined_test_data['question'],
            'answer': test_results
        })
        
        qa_filename = f'{base_filename}_qa_only.csv'
        qa_df.to_csv(qa_filename, index=False, encoding='utf-8-sig')
        print(f"질문과 답변만 포함된 결과가 '{qa_filename}'에 저장되었습니다.")
    
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
    
    return base_filename

def load_markdown_files(md_directory):
    """마크다운 파일 로드 함수 - 각 파일을 하나의 청크로 유지"""
    print(f"마크다운 파일 로드 중... 경로: {md_directory}")
    
    md_chunks = []
    md_chunk_metadatas = []
    
    for root, dirs, files in os.walk(md_directory):
        doc_name = os.path.basename(root)
        if not doc_name:
            continue
            
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                    
                    if text:
                        page_num = int(file.split('.')[0]) if file.split('.')[0].isdigit() else 0
                        
                        md_chunks.append(text)
                        md_chunk_metadatas.append({
                            "source": doc_name,
                            "type": "Markdown",
                            "page": page_num,
                            "path": file_path,
                            "chunk_id": 0,
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
    model_id = "SEOKDONG/llama3.1_korean_v1.1_sft_by_aidx"
    tokenizer, model = setup_model()
    
    print("마크다운 문서 로드 중...")
    md_directory = "./data/건설안전지침_md"
    md_chunks, md_chunk_metadatas = load_markdown_files(md_directory)
    
    print(f"문서 통합 중... (학습 문서: {len(md_chunks)}개, 마크다운: {len(md_chunks)}개)")
    
    all_documents = md_chunks
    all_metadatas = md_chunk_metadatas
    
    print(f"총 {len(all_documents)}개의 문서로 벡터 저장소 구성")
    
    retriever = setup_retriever(all_documents, all_metadatas)
    
    print("QA 체인 설정 중...")
    qa_chain, prompt_template, hyperparams = setup_qa_chain(model, tokenizer, retriever)
    
    print("테스트 실행 중...")
    test_results = run_test(qa_chain, combined_test_data, test)
    
    print("임베딩 생성 중...")
    pred_embeddings = create_embeddings(test_results)
    
    print("결과 저장 중...")
    os.makedirs('./submissions', exist_ok=True)
    
    # combined_test_data를 추가로 전달
    save_results(test_results, pred_embeddings, model_id, prompt_template, hyperparams, combined_test_data)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"프로그램 완료! 총 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")

if __name__ == "__main__":
    main() 