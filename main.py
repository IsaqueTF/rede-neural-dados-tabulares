"""
Ponto de entrada unificado para reprodução dos experimentos do artigo:
"Desenvolvimento da Inteligência Artificial: estruturação do banco de dados,
treinamento, codificação da IA, testes e validação"
UniEVANGÉLICA - 2026
"""
import argparse
from src.train import train
from src.evaluate import evaluate
from src.run_35_experiments import main as run_experiments

def main():
    parser = argparse.ArgumentParser(description="Pipeline de IA para Dados Tabulares - UniEVANGÉLICA")
    parser.add_argument(
        "--modo", 
        type=str, 
        choices=["treino", "avaliar", "bateria_artigo"], 
        default="bateria_artigo",
        help="Modo de execução: 'treino' (ciclo único), 'avaliar' (teste) ou 'bateria_artigo' (Fase 5 - 35 ciclos com AMP)."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Artigo: Desenvolvimento da Inteligência Artificial: estruturação do")
    print("banco de dados, treinamento, codificação da IA, testes e validação")
    print("Autores: Isaque Teles Ferreira & Igor Dalarmelino Borges (UniEVANGÉLICA)")
    print("=" * 70)

    if args.modo == "treino":
        print("\n>> Iniciando treinamento de ciclo único...")
        train()
    elif args.modo == "avaliar":
        print("\n>> Avaliando modelo no conjunto de teste independente...")
        evaluate()
    elif args.modo == "bateria_artigo":
        print("\n>> Executando Bateria da Fase 5 (35 ciclos estatísticos com AMP)...")
        print(">> Os resultados serão consolidados em 'data/processed/resultados_35_experimentos_alteracao5.csv'.")
        run_experiments()
        print("\n[SUCESSO] Bateria finalizada. Verifique a pasta data/processed/.")

if __name__ == "__main__":
    main()
