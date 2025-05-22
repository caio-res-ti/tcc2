from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
import asyncio
import re
from bs4 import BeautifulSoup
import logging
import uvicorn
import json
import time

# Configuração de logging mais detalhada
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="EPUB Translation API", version="1.0.0")

# CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuração do Ollama
OLLAMA_BASE_URL = "http://10.159.10.50:11434/v1"
MODEL_NAME = "gemma3:4b"

# Modelo de dados de entrada
class EpubChapter(BaseModel):
    bookTitle: str
    author: str
    chapterTitle: str
    chapterIndex: int
    totalChapters: int
    htmlContent: str
    timestamp: str

# Modelo de resposta
class TranslationResponse(BaseModel):
    success: bool
    translatedHtml: Optional[str] = None
    originalHtml: Optional[str] = None
    error: Optional[str] = None
    processingTime: Optional[float] = None

class TranslationAgent:
    def __init__(self):
        self.ollama_url = f"{OLLAMA_BASE_URL}/chat/completions"
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minutos timeout
        logger.info(f"TranslationAgent inicializado. Ollama URL: {self.ollama_url}")
    
    async def translate_text_chunk(self, text_chunk: str, context: str = "") -> str:
        """Traduz um pedaço de texto usando Ollama"""
        
        logger.info(f"Iniciando tradução de chunk com {len(text_chunk)} caracteres")
        logger.debug(f"Texto a traduzir: {text_chunk[:100]}...")
        
        # Prompt para tradução humanizada
        system_prompt = """Você é um tradutor literário especializado em traduzir textos de forma natural e humanizada. 

INSTRUÇÕES:
- Traduza o texto do inglês para o português brasileiro
- Mantenha o sentido real e a intenção do autor
- Preserve nomes próprios, lugares, títulos de obras
- Mantenha termos técnicos que não devem ser traduzidos
- Use linguagem fluente e natural em português
- Preserve o estilo e tom do texto original
- NÃO adicione explicações ou comentários
- Retorne APENAS o texto traduzido"""

        user_prompt = f"""Contexto: {context}

Texto para traduzir:
{text_chunk}

Tradução em português brasileiro:"""

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 4000,
            "stream": False
        }
        
        try:
            logger.info(f"Enviando requisição para Ollama...")
            start_time = time.time()
            
            response = await self.client.post(self.ollama_url, json=payload)
            
            request_time = time.time() - start_time
            logger.info(f"Resposta do Ollama recebida em {request_time:.2f}s")
            
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Resposta completa do Ollama: {json.dumps(result, indent=2)}")
            
            translated_text = result["choices"][0]["message"]["content"].strip()
            
            # Remove possíveis prefixos de resposta
            original_translated = translated_text
            translated_text = re.sub(r'^(Tradução:|Português:|Resultado:)\s*', '', translated_text, flags=re.IGNORECASE)
            
            if original_translated != translated_text:
                logger.info("Removidos prefixos da resposta")
            
            logger.info(f"Tradução concluída. Resultado: {translated_text[:100]}...")
            return translated_text
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Erro HTTP na tradução: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=500, detail=f"Erro HTTP na tradução: {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("Timeout na requisição para Ollama")
            raise HTTPException(status_code=500, detail="Timeout na tradução")
        except Exception as e:
            logger.error(f"Erro geral na tradução: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Erro na tradução: {str(e)}")
    
    async def translate_html_content(self, html_content: str, book_context: str) -> str:
        """Traduz o conteúdo HTML preservando a estrutura"""
        try:
            logger.info(f"Iniciando tradução HTML. Tamanho: {len(html_content)} caracteres")
            
            soup = BeautifulSoup(html_content, 'html.parser')
            logger.info("HTML parseado com BeautifulSoup")
            
            # Encontra todos os elementos de texto que precisam ser traduzidos
            text_elements = soup.find_all(text=True)
            logger.info(f"Encontrados {len(text_elements)} elementos de texto")
            
            # Filtra apenas texto significativo (não whitespace, não scripts, etc.)
            meaningful_texts = []
            for element in text_elements:
                if element.parent.name not in ['script', 'style', 'meta', 'link']:
                    text = element.strip()
                    if text and len(text) > 2:  # Ignora textos muito pequenos
                        meaningful_texts.append(element)
            
            logger.info(f"Filtrados para {len(meaningful_texts)} textos significativos")
            
            if not meaningful_texts:
                logger.warning("Nenhum texto significativo encontrado para traduzir")
                return str(soup)
            
            # Log dos primeiros textos encontrados
            for i, elem in enumerate(meaningful_texts[:3]):
                logger.debug(f"Texto {i+1}: {elem.strip()[:50]}...")
            
            # Agrupa textos pequenos para traduzir em batch
            translation_tasks = []
            current_batch = []
            current_batch_length = 0
            
            for text_element in meaningful_texts:
                text_length = len(text_element.strip())
                
                # Se o texto é muito grande, traduz sozinho
                if text_length > 1500:
                    if current_batch:
                        translation_tasks.append(current_batch)
                        current_batch = []
                        current_batch_length = 0
                    translation_tasks.append([text_element])
                else:
                    # Se adicionar ao batch não ultrapassar 2000 chars
                    if current_batch_length + text_length <= 2000:
                        current_batch.append(text_element)
                        current_batch_length += text_length
                    else:
                        # Finaliza batch atual e inicia novo
                        if current_batch:
                            translation_tasks.append(current_batch)
                        current_batch = [text_element]
                        current_batch_length = text_length
            
            # Adiciona último batch se não estiver vazio
            if current_batch:
                translation_tasks.append(current_batch)
            
            logger.info(f"Criados {len(translation_tasks)} batches de tradução")
            
            # Executa traduções
            for batch_idx, batch in enumerate(translation_tasks):
                logger.info(f"Processando batch {batch_idx + 1}/{len(translation_tasks)}")
                
                if len(batch) == 1:
                    # Tradução individual
                    original_text = batch[0].strip()
                    logger.info(f"Traduzindo texto individual: {original_text[:50]}...")
                    translated_text = await self.translate_text_chunk(original_text, book_context)
                    batch[0].replace_with(translated_text)
                    logger.info(f"Texto traduzido para: {translated_text[:50]}...")
                else:
                    # Tradução em batch
                    combined_text = "\n\n---\n\n".join([elem.strip() for elem in batch])
                    logger.info(f"Traduzindo batch com {len(batch)} textos")
                    translated_combined = await self.translate_text_chunk(combined_text, book_context)
                    
                    # Divide a tradução de volta
                    translated_parts = translated_combined.split("\n\n---\n\n")
                    logger.info(f"Batch traduzido dividido em {len(translated_parts)} partes")
                    
                    # Aplica cada parte traduzida
                    for i, text_element in enumerate(batch):
                        if i < len(translated_parts):
                            translated_part = translated_parts[i].strip()
                            logger.debug(f"Aplicando tradução {i+1}: {translated_part[:30]}...")
                            text_element.replace_with(translated_part)
                
                # Pequena pausa entre batches para não sobrecarregar
                await asyncio.sleep(0.5)
            
            result_html = str(soup)
            logger.info(f"Tradução HTML concluída. Tamanho final: {len(result_html)} caracteres")
            
            return result_html
            
        except Exception as e:
            logger.error(f"Erro na tradução HTML: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Erro na tradução: {str(e)}")

# Instância do agente de tradução
translator = TranslationAgent()

@app.post("/translate", response_model=TranslationResponse)
async def translate_chapter(chapter: EpubChapter):
    """Endpoint principal para traduzir capítulos do EPUB"""
    
    logger.info(f"📚 Nova requisição de tradução recebida:")
    logger.info(f"  - Livro: {chapter.bookTitle}")
    logger.info(f"  - Autor: {chapter.author}")
    logger.info(f"  - Capítulo: {chapter.chapterTitle}")
    logger.info(f"  - Índice: {chapter.chapterIndex}/{chapter.totalChapters}")
    logger.info(f"  - Tamanho HTML: {len(chapter.htmlContent)} caracteres")
    logger.info(f"  - Timestamp: {chapter.timestamp}")
    
    start_time = time.time()
    
    try:
        # Contexto do livro para melhor tradução
        book_context = f"Livro: {chapter.bookTitle} por {chapter.author}. Capítulo: {chapter.chapterTitle}"
        
        # Traduz o conteúdo HTML
        logger.info("🔄 Iniciando processo de tradução...")
        translated_html = await translator.translate_html_content(
            chapter.htmlContent, 
            book_context
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"✅ Tradução concluída com sucesso em {processing_time:.2f}s")
        
        return TranslationResponse(
            success=True,
            translatedHtml=translated_html,
            originalHtml=chapter.htmlContent,
            processingTime=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"❌ Erro na tradução após {processing_time:.2f}s: {str(e)}")
        
        return TranslationResponse(
            success=False,
            error=str(e),
            processingTime=processing_time
        )

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    logger.info("🏥 Health check solicitado")
    try:
        # Testa conexão com Ollama
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/models")
            response.raise_for_status()
            
        logger.info("✅ Health check: Ollama conectado")
        return {"status": "healthy", "ollama": "connected"}
    except Exception as e:
        logger.error(f"❌ Health check falhou: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/")
async def root():
    return {
        "message": "EPUB Translation API",
        "version": "1.0.0",
        "endpoints": {
            "translate": "POST /translate",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    logger.info("🚀 Iniciando EPUB Translation API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")