#!/usr/bin/env python3
"""
Client simple pour interagir avec un serveur vLLM
Usage : python3 chat_vllm.py
"""

import json
import urllib.request
import urllib.error
import os

# ─── Configuration ────────────────────────────────────────────────────────────
VLLM_URL  = os.getenv("VLLM_URL",  "https://litis-vllm.univ-rouen.fr")
API_KEY   = os.getenv("VLLM_KEY",  "b5e611b779da8e6757a0860783af83d0afb148b9997bd898a7959c8c9fe52a64")
MODEL     = os.getenv("VLLM_MODEL", "")   # laissé vide = auto-détecté
MAX_TOKENS = 900
TEMPERATURE = 0.0
# ──────────────────────────────────────────────────────────────────────────────


def get_model_name() -> str:
    """Récupère le nom du modèle chargé sur le serveur."""
    req = urllib.request.Request(
        f"{VLLM_URL}/v1/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
        return data["data"][0]["id"]


def chat(messages: list) -> str:
    """Envoie une conversation et retourne la réponse."""
    model = MODEL or get_model_name()
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        f"{VLLM_URL}/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


def main():
    print("=" * 60)
    print("  Client vLLM — tapez 'quit' ou 'exit' pour quitter")
    print(f"  Serveur : {VLLM_URL}")
    print("=" * 60)

    # Vérifier la connexion
    try:
        model = get_model_name()
        print(f"  Modèle  : {model}")
    except Exception as e:
        print(f"\n❌ Impossible de joindre le serveur : {e}")
        print(f"   Vérifiez que vLLM tourne sur {VLLM_URL}")
        print(f"   Conseil : VLLM_URL=http://IP:8000 python3 chat_vllm.py")
        return

    print("=" * 60)
    print()

    messages = []
    system_prompt = input("Prompt système (Entrée pour ignorer) : ").strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    print()

    while True:
        try:
            user_input = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAu revoir !")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Au revoir !")
            break

        messages.append({"role": "user", "content": user_input})

        print("Assistant : ", end="", flush=True)
        try:
            response = chat(messages)
            print(response)
            messages.append({"role": "assistant", "content": response})
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            print(f"\n❌ Erreur HTTP {e.code} : {body}")
        except urllib.error.URLError as e:
            print(f"\n❌ Erreur réseau : {e.reason}")
        except Exception as e:
            print(f"\n❌ Erreur : {e}")

        print()


if __name__ == "__main__":
    main()