# Cambio de ASL a LSP (Lengua de Señas Peruana)

## Por qué el cambio

La Lengua de Señas Peruana es reconocida oficialmente por el Estado peruano
mediante la **Ley N° 29535 (2010)** y su reglamento (D.S. 006-2017-MIMP). Es la
lengua que usan realmente las personas sordas en el Perú.

Para un proyecto sobre preservación cultural peruana en la WRO 2026 ("Robots
Meet Culture"), usar el alfabeto estadounidense (ASL) era una incoherencia que
un jurado podía señalar. Con LSP, el sistema le habla de verdad a su público.

## Qué cambia y qué no

| | ¿Cambia? |
|---|---|
| El código del reconocimiento | **No.** El modelo aprende las formas que se le graben, sin importar el idioma |
| El pipeline, la red, el plotter | **No.** Nada |
| **Los datos de entrenamiento** | **Sí. Este es todo el trabajo** |
| Cantidad de letras | Sí: se agrega la **Ñ** (27 letras en vez de 26) |

## Fuente obligatoria para las formas de mano

No inventar las señas ni copiarlas de videos sueltos de YouTube. Usar la guía
oficial del Ministerio de Educación:

- **"Lengua de señas peruana: Guía para el aprendizaje de la lengua de señas
  peruana, vocabulario básico"** — MINEDU, Dirección General de Educación
  Básica Especial (DIGEBE). Incluye el capítulo del alfabeto dactilológico.
- Marco legal y contexto: Ley 29535 y su reglamento.

**Mejor aún:** si el colegio o el MALI tienen contacto con un intérprete de LSP
o con la comunidad sorda de Lima, pedir que revisen las señas antes de grabar.
Eso además es material excelente para el informe (validación con usuarios
reales) y evita entrenar el modelo con señas mal hechas.

## HALLAZGO: comparación contra la guía oficial

Se descargó la guía del MINEDU y se comparó su alfabeto (páginas 69-70) contra
las señas que el modelo tenía grabadas en ASL. Resultado:

Se revisó **letra por letra**, con las etiquetas visibles en cada dibujo.
Resultado: **solo 2 señas hay que grabar.**

| Letra | Qué pasa |
|---|---|
| **U** | **DISTINTA.** En LSP es **índice + meñique** extendidos (forma de "cuernos"), con medio y anular doblados. En ASL es índice + medio juntos. Hay que borrar la U vieja y grabarla de nuevo |
| **Ñ** | **NUEVA.** Es la **N con movimiento lateral** (la guía lo marca con una flecha ↔) |
| Las otras 25 | **Coinciden** con lo que ya estaba grabado: A B C D E F G H I J K L M N O P Q R S T V W X Y Z |

Esto tiene sentido histórico: el alfabeto unimanual usado hoy por la comunidad
sorda peruana comparte raíz con el internacional. Existe además un "alfabeto
antiguo" mixto (de dos manos) que los sordos adultos consideran patrimonio, pero
no es el que enseña la guía oficial ni el que se usa para deletrear hoy.

**Implicación práctica: no hace falta regrabar las 26 letras.** Son ~15 minutos:

```bash
python3 borrar_letras.py u     # quita la U vieja (forma de ASL)
python3 collect_data.py        # graba la U nueva (cuernos) y la Ñ (tecla ";")
python3 train_model.py
```

Al grabar la **Ñ**, mover la mano de lado a lado mientras graba: ese movimiento
es lo que la separa de la N (igual que la J y la Z).

> Aun así, conviene que un intérprete de LSP o alguien de la comunidad sorda
> revise las señas antes de la competencia. La comparación se hizo sobre los
> dibujos de la guía, y una validación humana es más confiable — además de ser
> material excelente para el informe.

### Detalle técnico ya resuelto

Como la Ñ es "N + movimiento", tiene el mismo problema que I/J: misma forma de
mano, distinto movimiento. Ya se agregó a `lsp_app.py` un desempate por
movimiento total de la mano (`ENIE_MOTION_THRESH`), igual al que existía para
I/J. Probado: mano quieta devuelve N, mano en movimiento devuelve Ñ.

## Cómo re-grabar el dataset

1. Imprimir o tener a la mano el alfabeto dactilológico de LSP.
2. Borrar/ignorar el dataset viejo. El archivo `asl_data_ANTIGUO.csv` y
   `model_asl_ANTIGUO.pkl` quedan guardados por si hace falta volver atrás.
3. Correr el grabador:

   ```bash
   python3 collect_data.py
   ```

   Teclas: `a-z` para cada letra, **`;` para la Ñ**, `0-9` números,
   `ESPACIO` / `TAB` / `SUPR` para los gestos de control.
   Cada vez que se presiona una tecla graba 100 muestras.

4. **Meta: mínimo 400–500 muestras por letra** (o sea, presionar cada tecla
   4 o 5 veces). El dataset anterior tenía 12,500 muestras en total; apunten a
   algo parecido: unas 13,500 con la Ñ incluida.

5. Grabar con **variedad**, o el modelo solo funcionará en condiciones
   idénticas a la grabación:
   - Cambiar de distancia a la cámara entre tanda y tanda.
   - Girar un poco la mano (unos grados a cada lado).
   - Cambiar la iluminación (día, noche, luz artificial).
   - Si pueden, que **más de una persona** grabe las mismas letras. Esto es lo
     que más mejora la precisión con jurados o visitantes desconocidos.

6. Entrenar:

   ```bash
   python3 train_model.py
   ```

   Fijarse en el número de "Honest accuracy". El sistema ASL llegó a ~96%.

7. Probar en vivo:

   ```bash
   python3 lsp_app.py local
   ```

   Señar el abecedario completo y anotar qué letras se confunden.

8. Las letras que fallen: volver a grabarlas con más cuidado y consistencia,
   y re-entrenar. Así se corrigieron V/W/G/Q en la versión ASL.

## Puntos a verificar propios de LSP

- **La Ñ.** Es la letra nueva. Verificar dos cosas: que el modelo la distinga
  de la N, y que al escribirse llegue bien a la computadora receptora
  (`receiver.py` avisa por consola si `pyautogui` no logra escribirla).
- **Letras con movimiento.** En ASL, la I y la J tienen la misma forma de mano
  y solo cambia el movimiento, por eso `lsp_app.py` tiene un desempate especial
  para ese par. En LSP hay que revisar con la guía cuáles son las letras con
  movimiento y ajustar ese desempate si no coincide.
- **Letras parecidas entre sí.** Al probar en vivo va a aparecer su propia lista
  de confusiones, distinta a la de ASL (que era V/W/G/Q e I/J).

## Qué actualizar además del código

- El **informe (dossier)**: cambiar todas las menciones de "lengua de señas"
  genérica o ASL por **LSP**, y agregar la Ley 29535 como respaldo. Es un
  párrafo que suma en impacto social.
- La **página web** y el README del proyecto.
- La presentación del equipo.
