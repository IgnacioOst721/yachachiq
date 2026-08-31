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
