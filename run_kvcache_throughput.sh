#!/bin/bash

MODEL="llama-2-7b.q4_0.gguf"
PROMPT_FILE="prompt_10_long.txt"
BIN="./build/bin/llama-cli"

LOG_PC="log_q4_0_pc_q4.txt"
LOG_BASE="log_q4_0_q4.txt"

# 기존 로그 삭제
rm -f $LOG_PC $LOG_BASE

IDX=0

while IFS= read -r PROMPT; do
  IDX=$((IDX+1))
  echo "================ Prompt $IDX =================" | tee -a $LOG_PC $LOG_BASE

  # ---------- q4_0_pc ----------
  echo "[q4_0_pc] Prompt $IDX" | tee -a $LOG_PC
  $BIN \
    -m $MODEL \
    -p "$PROMPT" \
    --cache-type-k q4_0_pc \
    --cache-type-k-scales scales_k.bin \
    -ngl 0 \
    -n 300 \
    2>&1 | tee -a $LOG_PC

  echo "" >> $LOG_PC

  # ---------- q4_0 ----------
  echo "[q4_0] Prompt $IDX" | tee -a $LOG_BASE
  $BIN \
    -m $MODEL \
    -p "$PROMPT" \
    --cache-type-k q4_0 \
    -ngl 0 \
    -n 300 \
    2>&1 | tee -a $LOG_BASE

  echo "" >> $LOG_BASE

done < "$PROMPT_FILE"

