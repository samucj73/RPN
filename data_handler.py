
import requests

API_URL = "https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest"
HEADERS = { "User-Agent": "Mozilla/5.0" }

def fetch_latest_result():
    try:
        response = requests.get(API_URL, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            game_data = data.get("data", {})
            result = game_data.get("result", {})
            outcome = result.get("outcome", {})
            lucky_list = result.get("luckyNumbersList", [])

            number = outcome.get("number")
            timestamp = game_data.get("startedAt")
            lucky_numbers = [item["number"] for item in lucky_list]

            return {
                "number": number,
                "timestamp": timestamp,
                "lucky_numbers": lucky_numbers
            }
    except:
        return None

def salvar_resultado_em_arquivo(resultados, caminho='historico_resultados.txt'):
    try:
        with open(caminho, 'a') as f:
            for r in resultados:
                linha = f"{r['number']} | {','.join(map(str, r['lucky_numbers']))} | {r['timestamp']}\n"
                f.write(linha)
    except Exception as e:
        print(f"[Erro ao salvar]: {e}")
