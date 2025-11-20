#!/usr/bin/env python3
"""
General utilities for document processing, deterministic operations, and other common functions.
"""

import json
import os
import requests
import torch
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from params import add_all_args
from retrieve import BM25DB, VectorDB


DEFAULT_CHARS_PER_TOKEN = 4.0
TOKEN_LIMIT_ERROR_PATTERNS = (
    "max context length",
    "maximum context length",
    "context length exceeded",
    "token limit",
    "too many tokens",
    "request too large",
    "context window",
)



def load_url_mapping(directory: str) -> Dict[str, str]:
    """Load URL mapping from url_mapping.json in specified directory."""
    mapping_path = Path(directory) / "url_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def get_base_filename(filename: str) -> str:
    """Extract base filename without extension."""
    if '.' in filename:
        return '.'.join(filename.split('.')[:-1])
    return filename


def save_url_mapping(directory: str, url_mapping: Dict[str, str]) -> None:
    """Save URL mapping to url_mapping.json in specified directory."""
    mapping_path = Path(directory) / "url_mapping.json"
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(url_mapping, f, indent=2, ensure_ascii=False)


def serialize_cli_args(args: Any) -> Dict[str, Any]:
    """Convert parsed CLI args into JSON-serializable primitive types."""
    params: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            params[key] = value
        else:
            params[key] = str(value)
    return params


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set PyTorch seed for reproducible results.
    
    Based on systematic analysis, torch.manual_seed() is the only component
    needed to ensure deterministic behavior in our retrieval system.
    """
    import torch
    torch.manual_seed(seed)


def filter_dataset_by_difficulty(df, difficulty: int = 0):
    """
    Filter dataset by minimum number of answer links (difficulty level).
    
    Args:
        df: pandas DataFrame with dataset
        difficulty: Minimum number of answer links required (0 = no filtering)
        
    Returns:
        Filtered DataFrame with queries having >= difficulty answer links
    """
    if difficulty <= 0:
        return df
    
    # Count answer links for each row
    link_counts = df.apply(
        lambda row: sum(1 for col in df.columns 
                       if col.startswith('wikipedia_link_') and row.notna()[col]), 
        axis=1
    )
    
    filtered_df = df[link_counts >= difficulty].reset_index(drop=True)
    print(f"Filtered dataset by difficulty >= {difficulty}: {len(filtered_df)} queries remaining (from {len(df)} total)")
    
    return filtered_df


def detect_device() -> str:
    """Auto-detect the best available device."""
    if torch is None:
        return "cpu"

    try:
        import habana_frameworks.torch.core as htcore
        if torch.hpu.is_available():
            os.environ["PT_HPU_LAZY_MODE"] = "1"
            return "hpu"
    except ImportError:
            pass
    
    if torch.cuda.is_available():
        return "cuda"
    
    if torch.xpu.is_available():
        return "xpu"
    
    # Default to CPU
    return "cpu"


def get_model_info_from_service(service_url: str) -> Optional[Dict]:
    """Get model information from LLM service."""
    try:
        # Try OpenAI-compatible API first
        models_response = requests.get(f"{service_url.rstrip('/v1/chat/completions').rstrip('/v1')}/v1/models", timeout=10)
        if models_response.status_code == 200:
            models_data = models_response.json()
            if "data" in models_data and len(models_data["data"]) > 0:
                return models_data["data"][0]
        
        # Try alternative endpoints
        base_url = service_url.rstrip('/v1/chat/completions').rstrip('/v1')
        for endpoint in ["/models", "/info", "/v1/model"]:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
                if response.status_code == 200:
                    return response.json()
            except:
                continue
                
    except Exception as e:
        print(f"Warning: Could not auto-detect model from {service_url}: {e}")
    
    return None


def get_model_name_from_service(service_url: str) -> str:
    """Auto-detect model name from LLM service."""
    model_info = get_model_info_from_service(service_url)
    
    if model_info:
        # Try different possible fields for model name
        for field in ["id", "model", "name", "model_name"]:
            if field in model_info:
                return model_info[field]
    
    # Default fallback
    return "/mnt/weka/data/pytorch/llama3.3/Meta-Llama-3.3-70B-Instruct/"


def get_max_tokens_from_service(service_url: str) -> int:
    """Auto-detect max tokens from LLM service."""
    model_info = get_model_info_from_service(service_url)
    
    if model_info:
        # Try different possible fields for max tokens
        for field in ["max_model_len"]:
            if field in model_info and isinstance(model_info[field], int):
                return model_info[field]
    
    # Default fallback based on common models
    return 10240


def resolve_config_value(value: Union[str, int], auto_func, *args) -> Union[str, int]:
    """Resolve configuration value that might be 'auto'."""
    if value == "auto":
        return auto_func(*args)
    return value


def get_device_config():
    """Get comprehensive device configuration."""
    config = {
        "device_type": detect_device(),
        "device_count": 1,
        "device_memory": None
    }
    
    if torch is None:
        return config
    
    if config["device_type"] == "hpu":
        config["device_count"] = torch.hpu.device_count()
    
    elif config["device_type"] == "cuda":
        config["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            config["device_memory"] = torch.cuda.get_device_properties(0).total_memory
    
    elif config["device_type"] == "xpu":
        config["device_count"] = torch.xpu.device_count()
    
    return config


def setup_llm_config(args):
    """Setup LLM configuration with auto-detection."""
    device = resolve_config_value(args.device, detect_device)

    model_name = resolve_config_value(
        args.llm_model,
        get_model_name_from_service,
        args.llm_service_url,
    )

    output_token_limit = args.llm_output_token_limit
    if isinstance(output_token_limit, str):
        output_token_limit = resolve_config_value(
            output_token_limit,
            get_max_tokens_from_service,
            args.llm_service_url,
        )

    if isinstance(output_token_limit, str):
        try:
            output_token_limit = int(output_token_limit)
        except ValueError:
            output_token_limit = get_max_tokens_from_service(args.llm_service_url)

    context_token_limit = args.llm_context_token_limit
    if isinstance(context_token_limit, str):
        try:
            context_token_limit = int(context_token_limit)
        except ValueError:
            context_token_limit = get_max_tokens_from_service(args.llm_service_url)

    chars_per_token = args.llm_chars_per_token
    if not isinstance(chars_per_token, (int, float)) or chars_per_token <= 0:
        chars_per_token = DEFAULT_CHARS_PER_TOKEN

    context_char_limit = int(context_token_limit * chars_per_token) if context_token_limit and context_token_limit > 0 else 0

    request_timeout = args.llm_timeout
    if isinstance(request_timeout, str):
        try:
            request_timeout = int(request_timeout)
        except ValueError:
            request_timeout = 0

    llm_config = {
        "service_url": args.llm_service_url,
        "model_name": model_name,
        "output_token_limit": output_token_limit,
        "context_token_limit": context_token_limit,
        "context_char_limit": context_char_limit,
        "chars_per_token": chars_per_token,
        "device": device,
        "request_timeout": request_timeout,
        "reasoning_effort": args.reasoning,
    }

    print(f"LLM Config: {llm_config}")
    print(f"Context limits -> tokens: {context_token_limit}, chars: {context_char_limit}, chars/token: {chars_per_token}")

    return llm_config

def is_token_limit_error(response: Optional[requests.Response] = None, message: Optional[str] = None) -> bool:
    candidates = []
    if response is not None:
        try:
            data = response.json()
            if isinstance(data, dict):
                if "error" in data and isinstance(data["error"], dict):
                    candidates.append(str(data["error"].get("message", "")))
                elif "message" in data:
                    candidates.append(str(data.get("message", "")))
        except ValueError:
            pass
        candidates.append(response.text or "")
        status_code = response.status_code
    else:
        status_code = None

    if message:
        candidates.append(message)

    combined = " ".join(filter(None, candidates)).lower()
    if not combined:
        return False

    if status_code in {400, 401, 403, 413, 422}:
        return any(pattern in combined for pattern in TOKEN_LIMIT_ERROR_PATTERNS)

    return any(pattern in combined for pattern in TOKEN_LIMIT_ERROR_PATTERNS)


def ensure_unique_path(path: Union[str, Path]) -> str:
    """Return a filesystem path that won't overwrite existing files."""

    file_path = Path(path)
    if not file_path.exists():
        return str(file_path)

    stem = file_path.stem
    suffix = file_path.suffix
    counter = 2
    while True:
        candidate = file_path.with_name(f"{stem}-{counter}{suffix}")
        if not candidate.exists():
            return str(candidate)
        counter += 1


def print_evaluation_summary(
    avg_metrics: Dict[str, float],
    valid_queries: int,
    *,
    title: str = "EVALUATION RESULTS",
    extra_retrieval_fields: Optional[List[Tuple[str, str, Optional[Callable[[float], str]]]]] = None,
    extra_timing_fields: Optional[List[Tuple[str, str, Optional[Callable[[float], str]]]]] = None,
) -> None:
    """Pretty-print shared evaluation metrics summary for single or multi-shot runs."""

    def _format_value(value: Any, formatter: Optional[Callable[[float], str]]) -> str:
        if formatter:
            return formatter(value)
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        return str(value)

    def _print_metric(label: str, key: str) -> None:
        if key in avg_metrics:
            print(f"  {label:<28} {_format_value(avg_metrics[key], None)}")

    print(f"\n{'=' * 80}")
    print(f"{title} ({valid_queries} queries)")
    print(f"{'=' * 80}")

    # Precision metrics
    print(f"\nPRECISION METRICS:")
    _print_metric("Precision@N:", "precision@N")
    for k in [1, 3, 5, 10]:
        _print_metric(f"Precision@{k}:", f"precision@{k}")

    # Recall metrics
    print(f"\nRECALL METRICS:")
    _print_metric("Recall@N:", "recall@N")
    for k in [1, 3, 5, 10]:
        _print_metric(f"Recall@{k}:", f"recall@{k}")

    # F1 metrics
    print(f"\nF1 METRICS:")
    _print_metric("F1@N:", "f1@N")
    for k in [1, 3, 5, 10]:
        _print_metric(f"F1@{k}:", f"f1@{k}")

    # Ranking metrics
    print(f"\nRANKING METRICS:")
    _print_metric("Mean Average Precision:", "average_precision")

    # Retrieval statistics
    print(f"\nRETRIEVAL STATISTICS:")
    if "retrieved_passages_count" in avg_metrics:
        print(
            f"  {'Avg Passages Retrieved:':<28} "
            f"{avg_metrics['retrieved_passages_count']:.1f}"
        )
    if "retrieved_docs_count" in avg_metrics:
        print(
            f"  {'Avg Unique Docs (N):':<28} {avg_metrics['retrieved_docs_count']:.1f}"
        )
    if extra_retrieval_fields:
        for label, key, formatter in extra_retrieval_fields:
            if key not in avg_metrics:
                continue
            print(f"  {label:<28} {_format_value(avg_metrics[key], formatter)}")

    # Timing statistics
    print(f"\nTIMING:")
    if "retrieval_time" in avg_metrics:
        print(
            f"  {'Avg Retrieval Time:':<28} {avg_metrics['retrieval_time'] * 1000:.1f}ms"
        )
    if "reranking_time" in avg_metrics and avg_metrics.get("reranking_time", 0) > 0:
        print(
            f"  {'Avg Reranking Time:':<28} {avg_metrics['reranking_time'] * 1000:.1f}ms"
        )
    if "total_retrieval_time" in avg_metrics:
        print(
            f"  {'Avg Total Time:':<28} {avg_metrics['total_retrieval_time'] * 1000:.1f}ms"
        )
    elif "total_time" in avg_metrics:
        print(f"  {'Avg Total Time:':<28} {avg_metrics['total_time'] * 1000:.1f}ms")
    if extra_timing_fields:
        for label, key, formatter in extra_timing_fields:
            if key not in avg_metrics:
                continue
            print(f"  {label:<28} {_format_value(avg_metrics[key], formatter)}")

    print(f"{'=' * 80}\n")


def setup(args):
    # Add all standard parameters
    add_all_args(args)

    # Special handling for --eval argument
    for action in args._actions:
        if '--eval' in action.option_strings:
            action.type = lambda x: int(x) if x.isdigit() else True
            action.const = True
            break

    args = args.parse_args()

    # Set deterministic seeds
    set_deterministic_seeds(args.seed)

    # Setup LLM configuration with auto-detection
    llm_config = setup_llm_config(args)

    # Setup device-specific environment
    device_config = get_device_config()
    print(f"Device Config: {device_config}")

    # Initialize database
    if args.retrieval_method == "bm25":
        db_class = BM25DB
    else:
        db_class = VectorDB
    
    if args.database is None:
        args.database = db_class.get_default_db_name()
    
    db_file_path = args.database if args.database.endswith('.db') else f"{args.database}.db"
    db_base_name = args.database.replace('.db', '') if args.database.endswith('.db') else args.database
    
    rag_db = db_class(
        device=args.device,
        database=db_base_name,
        retriever_model=args.retriever_model, 
        reranker_model=args.reranker_model, 
        benchmark=args.benchmark,

        k1=args.bm25_k1, b=args.bm25_b, method=args.bm25_method, 
        delta=args.bm25_delta, backend=args.bm25_backend, 
        stopwords=args.bm25_stopwords,
        show_progress=args.bm25_show_progress, stemmer=args.bm25_stemmer,

        vector_index_method=args.vector_index_method, 
        ivf_nprobe=args.ivf_nprobe,
        load_embeddings=args.load_embeddings, 
        num_embedding_devices=args.num_embedding_devices,
    )
    
    # Load database
    if os.path.exists(db_file_path):
        print(f"Loading existing database from {db_file_path}")
        rag_db.from_serialized(db_file_path)
    else:
        if not args.ingest:
            raise ValueError("Either --database (existing) or --ingest (to create new) must be provided")
        
        # Ingest from file or folder
        tic = time.time()
        rag_db.ingest_from_path(args.ingest, num_threads=args.threads)
        
        # Get number of passages for timing calculation
        num_passages = len(rag_db._doc_list)  # This should be available after ingestion
        toc = time.time()
        ingestion_speed = num_passages/(toc-tic)
        print(f"Ingestion of {num_passages} passages took {toc - tic:.2f} seconds. {ingestion_speed:.2f} docs/sec")
        
        # Save the database (unless --no-save is specified)
        if not args.no_save:
            print(f"Saving database to {db_file_path}")
            rag_db.serialize(db_file_path)
        else:
            print("Skipping database save (--no-save specified)")
    
    # Build strategy parameters
    retrieval_config = {"max_results": args.max_results}
    retrieval_config["retrieval_strategy"] = args.retrieval_strategy
    if args.retrieval_strategy == "top_p":
        retrieval_config["p"] = args.top_p
    elif args.retrieval_strategy == "relative":
        retrieval_config["ratio"] = args.relative_ratio
    else:
        retrieval_config["top_k_retriever"] = args.top_k_retriever
        retrieval_config["top_k_reranking"] = args.top_k_reranking


    return args, llm_config, device_config, rag_db, retrieval_config
