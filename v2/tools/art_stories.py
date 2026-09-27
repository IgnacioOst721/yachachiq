"""Test stories for the ART evaluation (Spanish, Quechua, English; Andean and not).

Each story lists `expect`: things a faithful drawing must show (used to judge plans by
hand and to compare with the VLM check). The Quechua texts are Southern Quechua
(Qusqu-Qullaw style); their Spanish versions play the role of the voice translation.
"""
from __future__ import annotations

STORIES = [
    {"id": "condor-apu", "lang": "spa_Latn", "expect": ["condor", "snowy mountain"],
     "text": "Mi abuelo me contó que el cóndor vive en los Apus nevados. Cada mañana sale volando "
             "sobre las montañas y cuida a los pueblos desde el cielo."},
    {"id": "nina-llama", "lang": "spa_Latn", "expect": ["girl", "llama", "river"],
     "text": "Una niña pastora caminaba con su llama por el camino del cerro. Llevaban papas a la feria "
             "del pueblo y se detuvieron a tomar agua en el río."},
    {"id": "zorro-luna", "lang": "spa_Latn", "expect": ["fox", "moon", "lake"],
     "text": "Dicen que una noche el zorro quiso subir a la luna. Se colgó de una cuerda que bajaba del "
             "cielo, pero la cuerda se rompió y cayó en la laguna."},
    {"id": "andenes-maiz", "lang": "spa_Latn", "expect": ["terraces", "maize", "farmer"],
     "text": "En los andenes de mi comunidad sembramos maíz y quinua. Mi mamá trabaja la tierra con la "
             "chakitaclla mientras canta en quechua."},
    {"id": "kuntur-qu", "lang": "quy_Latn", "expect": ["condor", "mountain", "llamas"],
     "text": "Ñawpa pachapi huk kunturmi karqan. Payqa sapa p'unchaw hatun urqukunapa hawanta "
             "phawarqan, llamakunata qhawaspa.",
     "text_es": "Hace mucho tiempo había un cóndor. Cada día volaba sobre las grandes montañas mirando "
                "a las llamas."},
    {"id": "ukuku-qu", "lang": "quy_Latn", "expect": ["bear", "dancer", "snow"],
     "text": "Qoyllur Rit'i raymipi ukukukunaqa rit'iman seqaspa tusunku.",
     "text_es": "En la fiesta del Qoyllur Rit'i los ukukus suben a la nieve y bailan."},
    {"id": "vizcacha-qu", "lang": "quy_Latn", "expect": ["vizcacha", "rocks"],
     "text": "Wisk'achaqa rumikuna ukhupi tiyan. Intiq k'anchayninpi q'uñikun.",
     "text_es": "La vizcacha vive entre las piedras. Se calienta con la luz del sol."},
    {"id": "grandma-weaving", "lang": "eng_Latn", "expect": ["grandmother", "loom", "house"],
     "text": "My grandmother weaves ponchos on a backstrap loom in front of her adobe house, "
             "while her little dog sleeps next to her."},
    {"id": "lighthouse", "lang": "eng_Latn", "expect": ["lighthouse", "boat", "waves"],
     "text": "An old fisherman in Scotland rows his small boat home through the waves, guided by "
             "the light of the lighthouse on the cliff."},
    {"id": "dragon-kite", "lang": "spa_Latn", "expect": ["child", "kite", "dragon"],
     "text": "En China, para el año nuevo, un niño hizo volar una cometa con forma de dragón junto a "
             "su abuelo en el parque."},
]
