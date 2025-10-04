# LLM-RAG-Chatbot
Production-ready RAG chatbot с векторным поиском (FAISS) и LLM (локальный qwen2.5 или OpenAI API). Демонстрирует ML Ops: FastAPI, Docker, CI/CD, pytest (74% coverage), автоматический eval. CPU-оптимизирован: 6.2s latency, 78% retrieval accuracy. Полная документация + метрики. Идеально для ML-портфолио. Python 3.11 | FastAPI | llama.cpp

# LLM RAG Chatbot - Портфолио проект

[![CI Pipeline](https://github.com/yourusername/llm-rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/sevastyanovilya/llm-rag-chatbot/actions)
[![Coverage](https://codecov.io/gh/yourusername/llm-rag-chatbot/branch/main/graph/badge.svg)](https://codecov.io/gh/sevastyanovilya/llm-rag-chatbot)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

**Production-ready RAG (Retrieval-Augmented Generation) чатбот**, демонстрирующий best practices ML-инженерии. Отвечает на вопросы, используя векторный поиск по базе знаний с локальным LLM inference (CPU-оптимизирован) или fallback на hosted API.

## 🎯 Ключевые возможности

- **Гибкий LLM Backend**: Локальный inference (llama.cpp + qwen2.5-7b) или hosted (OpenAI GPT-4o-mini)
- **Эффективный Retrieval**: FAISS векторный поиск с sentence transformers (CPU-оптимизирован)
- **Production паттерны**: FastAPI, Docker, CI/CD, автоматизированные evals, комплексные тесты
- **Готов для портфолио**: Полная документация, метрики и воспроизводимая настройка

---

## 🚀 Быстрый старт

### Требования
- Python 3.11+
- 8GB RAM (минимум 4GB)
- 5GB дискового пространства

### Установка
```bash
# Клонировать репозиторий
git clone https://github.com/sevastyanovilya/llm-rag-chatbot.git
cd llm-rag-chatbot

# Настроить окружение
make setup

# Скачать модель (опционально, для локального режима)
# Поместите qwen2.5-7b-instruct.Q4_K_M.gguf в ./models/

# Настроить окружение
cp .env.example .env
# Отредактируйте .env: установите LLM_MODE=local или hosted, добавьте API ключи если нужно

# Проиндексировать документы
make ingest

# Запустить сервер
make run
