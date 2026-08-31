# Borra del dataset las letras cuya seña NO coincide con la LSP, para volver
# a grabarlas. Las demás muestras se conservan intactas.
#
# Uso:
#   python3 borrar_letras.py a f t          # quita las letras a, f y t
#   python3 borrar_letras.py --ver          # solo muestra qué hay, sin borrar
#
# Siempre guarda una copia de seguridad antes de tocar nada.

import csv
import shutil
import sys
from collections import Counter

CSV_ORIGEN = "asl_data_ANTIGUO.csv"   # el dataset grabado en ASL
CSV_DESTINO = "lsp_data.csv"          # el dataset que usará el entrenamiento


def contar(path):
    c = Counter()
    with open(path) as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if row:
                c[row[0]] += 1
    return c


def main():
    args = [a for a in sys.argv[1:]]

    if not args or "--ver" in args:
        c = contar(CSV_ORIGEN)
        print(f"{CSV_ORIGEN}: {sum(c.values())} muestras, {len(c)} etiquetas\n")
        for lab in sorted(c, key=lambda k: (not k.isalpha(), k)):
            print(f"  {lab:>6}  {c[lab]:>5} muestras")
        print("\nPara borrar:  python3 borrar_letras.py a f t")
        return

    borrar = {a.lower() for a in args}
    antes = contar(CSV_ORIGEN)

    desconocidas = borrar - set(antes)
    if desconocidas:
        print(f"[ERROR] Estas etiquetas no existen en el dataset: {sorted(desconocidas)}")
        print(f"        Etiquetas disponibles: {sorted(antes)}")
        raise SystemExit(1)

    shutil.copy(CSV_ORIGEN, CSV_ORIGEN + ".respaldo")
    print(f"Respaldo guardado en {CSV_ORIGEN}.respaldo")

    quitadas = 0
    with open(CSV_ORIGEN) as fin, open(CSV_DESTINO, "w", newline="") as fout:
        r = csv.reader(fin)
        w = csv.writer(fout)
        w.writerow(next(r))                  # cabecera
        for row in r:
            if not row:
                continue
            if row[0].lower() in borrar:
                quitadas += 1
            else:
                w.writerow(row)

    despues = contar(CSV_DESTINO)
    print(f"\nQuitadas {quitadas} muestras de: {', '.join(sorted(borrar))}")
    print(f"{CSV_DESTINO}: {sum(despues.values())} muestras, {len(despues)} etiquetas")
    print("\nAhora vuelve a grabar esas letras (y la Ñ) con:")
    print("   python3 collect_data.py")
    print("y luego entrena con:")
    print("   python3 train_model.py")


if __name__ == "__main__":
    main()
