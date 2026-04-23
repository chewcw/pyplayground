#!/usr/bin/env bash

set -e

CLAIM_AGENT_HOME="/home/ccw/Documents/code/rnd/flutter-app/claim-agent"
CLAIM_AGENT_EXAMPLE="playground/chroma"

source "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/.venv/bin/activate"

echo "Using Python at: $(which python3)"
which python3

case "$1" in
    ingest)
        # rm "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/chroma.sqlite3" || true
        echo "Running ingest..."
        python3 "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/indexer.py" \
            ingest \
            --input_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/test-claims-handwritten/" \
            --chroma_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}" \
            --embed_provider sentence-transformers \
            --embed_model all-MiniLM-L6-v2 \
            --chunk_size 1000 \
            --chunk_overlap 200
        ;;

    ingest_with_multimodal)
        # rm "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/chroma.sqlite3" || true
        echo "Running ingest..."
        python3 "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/indexer.py" \
            ingest \
            --input_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/test-claims-handwritten/" \
            --chroma_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}" \
            --embed_provider sentence-transformers \
            --embed_model Qwen/Qwen3-VL-Embedding-2B \
            --chunk_size 1000 \
            --chunk_overlap 200 \
            --use_multimodal
        ;;

    retrieve)
        echo "Running retrieve..."
        python3 "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/indexer.py" \
            retrieve \
            --retrieve_query "$2" \
            --input_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/test-claims-structured/" \
            --chroma_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}" \
            --embed_provider sentence-transformers \
            --embed_model all-MiniLM-L6-v2
        ;;

    debug)
        # Ingest (verbose)
        python indexer.py ingest \
        --input_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/sample_receipts" \
        --chroma_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}" \
        --embed_provider sentence-transformers \
        --chunk_size 500 \
        --chunk_overlap 100
        ;;

    inspect)
        echo "Running inspect..."
        python "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/indexer.py" \
            inspect \
            "${@:3}" \
            --chroma_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}"
        ;;

    ingest-hf)
        echo "Running ingest with Qwen3-VL..."
        . .env
        python3 "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/indexer.py" \
            ingest-hf \
            --input_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/test-claims-handwritten-hf/" \
            --chroma_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}"
        ;;

    retrieve-hf)
        echo "Running retrieve with Qwen3-VL..."
        python3 "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/indexer.py" \
            retrieve-hf \
            --retrieve_query "$2" \
            --input_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}/test-claims-structured/" \
            --chroma_dir "${CLAIM_AGENT_HOME}/${CLAIM_AGENT_EXAMPLE}"
        ;;

    *)
        echo "Usage: $0 {ingest|ingest_with_multimodal|retrieve|debug|inspect|qwen-hf|retrieve-hf}"
        exit 1
        ;;
esac
