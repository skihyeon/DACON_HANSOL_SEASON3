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
    # 요청대로 배치 크기 4 고정
    return 4

def setup_model():
    """모델 설정 함수"""
    # Qwen2.5-14B-Instruct 모델로 변경
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    
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

def create_train_documents(combined_training_data):
    """개선된 학습 문서 생성 함수 - 청킹 적용"""
    train_questions = combined_training_data['question'].tolist()
    train_answers = combined_training_data['answer'].tolist()
    raw_documents = [f"Q: {q}\nA: {a}" for q, a in zip(train_questions, train_answers)]
    
    metadatas = [{"source": f"doc_{i}", "type": "QA", "question": q, "answer": a} 
                for i, (q, a) in enumerate(zip(train_questions, train_answers))]
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
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
            "k": 7, 
            "fetch_k": 20, 
            "lambda_mult": 0.7 
        }
    )

def setup_qa_chain(model, tokenizer, retriever):
    """QA 체인 설정 함수 - 최적화된 파이프라인 (개선됨)"""
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.3, 
        top_p=0.85,
        top_k=50,
        repetition_penalty=1.2,
        return_full_text=False,
        max_new_tokens=128, 
        batch_size=4,  # 요청대로 배치 크기 4 유지
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
        device_map="auto"
    )
    
    # Qwen 모델에 맞게 프롬프트 템플릿 수정 (채팅 형식)
    prompt_template = """
    <|im_start|>system
    당신은 건설 안전 전문가입니다.
    질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
    - 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
    - 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 내용을 포함하지 마세요.
    - 구체적인 안전 대책과 조치 계획을 명확하게 나열하세요.<|im_end|>
    <|im_start|>user
    아래 제공된 참고 자료를 기반으로 답변하세요:
    {context}

    질문:
    {question}<|im_end|>
    <|im_start|>assistant
    재발 방지 대책 및 향후 조치 계획:
    """

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def run_test(qa_chain, combined_test_data):
    """테스트 실행 함수 - 컨텍스트 처리 개선"""
    print(f"테스트 실행 시작... 총 테스트 샘플 수: {len(combined_test_data)}")
    print_gpu_memory()
    
    all_questions = combined_test_data['question'].tolist()
    
    pipeline_model = qa_chain.combine_documents_chain.llm_chain.llm.pipeline
    prompt_template = qa_chain.combine_documents_chain.llm_chain.prompt.template
    
    collection = qa_chain.retriever.vectorstore._collection
    embedding_func = qa_chain.retriever.vectorstore._embedding_function
    
    def get_context(question):
        query_embedding = embedding_func.embed_query(question)
        results = collection.query(
            query_embeddings=[query_embedding],
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
            weighted_docs.append(f"[관련도: {weight:.2f}] {source_info}\n{doc}")
        
        return "\n\n".join(weighted_docs)
    
    print("병렬 컨텍스트 검색 중...")
    retrieval_start = time.time()
    
    import multiprocessing
    max_workers = min(16, multiprocessing.cpu_count() * 2)
    print(f"병렬 검색에 {max_workers}개의 스레드 사용")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_contexts = list(tqdm(
            executor.map(get_context, all_questions),
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
    
    def batch_inference(examples):
        with torch.no_grad():
            outputs = pipeline_model(
                examples["prompt"],
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                top_k=50,
                repetition_penalty=1.2,
                max_new_tokens=128,
                return_full_text=False,
                batch_size=4  # 요청대로 배치 크기 4 고정
            )
        
        results = []
        for output in outputs:
            if isinstance(output, list) and len(output) > 0:
                results.append(output[0]['generated_text'])
            else:
                results.append("결과를 생성할 수 없습니다.")
        
        torch.cuda.empty_cache()
        
        return {"result": results}
    
    print("배치 추론 실행 중...")
    inference_start = time.time()
    
    batch_size = get_optimal_batch_size()
    print(f"추론에 배치 크기 {batch_size} 사용")
    
    results = dataset.map(
        batch_inference,
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
    """결과 후처리 함수"""
    processed_results = []
    
    for result in results:
        result = re.sub(r'^(다음과 같은|다음과|이에 대한|이를 위한|재발 방지를 위한|안전사고 예방을 위한)', '', result)
        
        result = re.sub(r'(조치를 취할 것을 제안합니다|조치가 필요합니다)[:.]?', '', result)
        
        result = re.sub(r'\n+', '\n', result)
        
        result = re.sub(r'^\d+\.\s*', '', result, flags=re.MULTILINE)
        
        sentences = []
        for line in result.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('-') or line.startswith('•'):
                sentences.append(line)
            else:
                for sentence in re.split(r'(?<=[.!?])\s+', line):
                    if sentence.strip():
                        sentences.append(f"- {sentence.strip()}")
        

        unique_sentences = []
        seen = set()
        for sentence in sentences:
            normalized = re.sub(r'[^\w\s]', '', sentence.lower())
            if normalized not in seen and len(normalized) > 5:  
                seen.add(normalized)
                unique_sentences.append(sentence)
        
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

def save_results(test_results, pred_embeddings):
    """결과 저장 함수"""
    submission = pd.read_csv('./data/sample_submission.csv', encoding='utf-8-sig')
    submission.iloc[:,1] = test_results
    submission.iloc[:,2:] = pred_embeddings
    from datetime import datetime
    current_time = datetime.now().strftime('%y%m%d_%H%M')
    filename = f'./submissions/submission_{current_time}.csv'
    submission.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"결과가 '{filename}'에 저장되었습니다.")

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
    tokenizer, model = setup_model()
    
    print("학습 문서 생성 중...")
    train_documents, train_metadatas = create_train_documents(combined_training_data)
    retriever = setup_retriever(train_documents, train_metadatas)
    
    print("QA 체인 설정 중...")
    qa_chain = setup_qa_chain(model, tokenizer, retriever)
    
    print("테스트 실행 중...")
    test_results = run_test(qa_chain, combined_test_data)
    
    print("임베딩 생성 중...")
    pred_embeddings = create_embeddings(test_results)
    
    print("결과 저장 중...")
    os.makedirs('./submissions', exist_ok=True)
    save_results(test_results, pred_embeddings)
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"프로그램 완료! 총 소요 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")

if __name__ == "__main__":
    main()