# Beispiel-System für die Integration von Language Models in Suchmaschinen und Dokumentenfindung - Nr. 1
Dieses Beispiel-System ist eine Python-Anwendung, die das OpenAI Embeddings Model **`text-embedding-ada-002`** mit einer vektorbasierten Datenbank (Redis Stack) integriert, um eine effiziente Dokumentensuche und -findung zu ermöglichen.

## Requirements

1. Installieren Sie die benötigten Python-Bibliotheken:
    ```bash
    pip install pandas openai redis streamlit tiktoken numpy
    ```
2. Starten Sie den Redis Stack Docker-Container:
    ```bash
    docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
    ```
3. Führen Sie embeddings.py aus, um die Vektoren der Text-Chunks zu erstellen und in der Datenbank zu speichern. Möglicherweise müssen Sie die Datei `embeddings.py` anpassen, um den Pfad zu den TXT-Dokumenten anzupassen.
    ```bash
    python embeddings.py
    ```
4. Starten Sie die streamlit-Anwendung:
    ```bash
    streamlit run path/2/this_dir/redis_test/search.py
    ```
    
## Architektur
Die Anwendung besteht aus zwei Hauptkomponenten:

1. **Language Model**: Das System verwendet das `text-embedding-ada-002` Modell von OpenAI, um Texte in semantisch aussagekräftige Vektoren umzuwandeln.
2. **Vektorbasierte Datenbank**: Ein in Docker bereitgestellter Redis Stack wird verwendet, um die Vektoren der Text-Chunks zu speichern und effiziente k-Nearest-Neighbor (kNN) Suchen durchzuführen.

## Funktionsweise
Das System funktioniert wie folgt:

1. Eine Sammlung von PDF-Dokumenten wird in Text-Chunks unterteilt und mithilfe des Language Models in Vektoren umgewandelt.
2. Die Vektoren werden in der Redis-Datenbank gespeichert.
3. Eine Suchanfrage wird ebenfalls in einen Vektor umgewandelt und an die Datenbank gesendet.
4. Die Datenbank führt eine kNN-Suche durch und gibt die am besten passenden Text-Chunks zurück.
5. Ein weiteres Language Model erstellt eine Zusammenfassung im Kontext der Anfrage und des Text-Chunks, die dem Nutzer als konkrete Antwort präsentiert wird.

## Disclaimer: Zugang zu Azure OpenAI Workspace erforderlich
Bitte beachten Sie, dass die Nutzung dieses Beispiel-Systems den Zugang zu einem Azure OpenAI Workspace voraussetzt. Ohne einen gültigen Zugang zu diesem Dienst können die Language Model Funktionen nicht genutzt werden.