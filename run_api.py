#!/usr/bin/env python3
"""
Script de arranque para la API del chatbot.
"""
import uvicorn
import sys
import os

# Añadir src al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    print("🚀 Iniciando API del Chatbot Universitario...")
    print("📚 Documentación interactiva disponible en:")
    print("   - Swagger UI: http://localhost:8000/docs")
    print("   - ReDoc: http://localhost:8000/redoc")
    print("\n⏹️  Presiona Ctrl+C para detener el servidor\n")
    
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=33209,
        reload=True,  # Auto-reload en desarrollo
        log_level="info"
    )
