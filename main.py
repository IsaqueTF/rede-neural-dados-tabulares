"""
Ponto de entrada unificado para reprodução dos experimentos do artigo:
"Desenvolvimento da Inteligência Artificial: estruturação do banco de dados,
treinamento, codificação da IA, testes e validação"
Autores: Isaque Teles Ferreira & Igor Dalarmelino Borges (UniEVANGÉLICA, 2026)
"""
import argparse
import sys
from src.train import train
from src.evaluate import evaluate
from src.run_35_experiments import main as run_fase_5
from src.run_30_experiments import main as run_fase_4

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Experimental de IA para Dados Tabulares - UniEVANGÉLICA 2026"
    )
    parser.add_argument(
        "--modo", 
        type=str, 
        choices=["treino", "avaliar", "experimentos"], 
        default="experimentos",
        help="Modo de execução: 'treino' (ciclo único), 'avaliar' (avaliação no teste) ou 'experimentos' (bateria)."
    )
    parser.add_argument(
        "--fase",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=5,
        help="Fase experimental a executar: 4 (Focal+LayerNorm, 30 ciclos) ou 5 (AMP, 35 ciclos). Padrão: 5."
    )
    parser.add_argument(
        "--ciclos",
        type=int,
        default=None,
        help="Número de ciclos independentes (padrão: 35 para Fase 5, 30 para Fase 4; use --ciclos 1 para teste rápido)."
    )
    args = parser.parse_args()

    print("=" * 75)
    print("Artigo: Desenvolvimento da Inteligência Artificial: estruturação do banco")
    print("de dados, treinamento, codificação da IA, testes e validação")
    print("UniEVANGÉLICA - 2026 | Autores: Isaque Teles Ferreira & Igor Dalarmelino Borges")
    print("=" * 75)

    if args.modo == "treino":
        print("\n>> Executando treinamento de ciclo único...")
        train()
    elif args.modo == "avaliar":
        print("\n>> Avaliando modelo no conjunto de teste independente...")
        evaluate()
    elif args.modo == "experimentos":
        if args.fase == 5:
            ciclos = args.ciclos if args.ciclos is not None else 35
            print(f"\n>> Executando Bateria da Fase 5 (AMP e Otimização Final - {ciclos} ciclos)...")
            run_fase_5(total_cycles=ciclos)
        elif args.fase == 4:
            ciclos = args.ciclos if args.ciclos is not None else 30
            print(f"\n>> Executando Bateria da Fase 4 (Focal Loss + LayerNorm - {ciclos} ciclos)...")
            run_fase_4(total_cycles=ciclos)
        else:
            print(f"\n>> Nota: Os resultados históricos da Fase {args.fase} estão consolidados em:")
            mapa_fases = {
                1: "data/processed/resultados_30_experimentos.csv (Fase 1 - MLP basal)",
                2: "data/processed/resultados_30_experimentos_aprimorados.csv (Fase 2 - Atenção + Residual)",
                3: "data/processed/resultados_30_experimentos_alteracao3.csv (Fase 3 - Base enriquecida + ruído)"
            }
            print(f"   -> {mapa_fases.get(args.fase)}")
            print("   Para reexecutar as configurações finais do artigo, utilize --fase 4 ou --fase 5.")

if __name__ == "__main__":
    main()
