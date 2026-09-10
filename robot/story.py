"""Story analysis: transcript -> title, elements, scene, image prompt, narration.

    analyze(text, lang=None, backends=None) -> dict

Backends (config.STORY_BACKENDS, tried in order until one works):
    remote     laptop/ai_server.py  POST /analyze
    ollama     Ollama directly       POST /api/generate (JSON mode)
    anthropic  Claude API            needs internet + ANTHROPIC_API_KEY
    rules      offline keyword rules, never fails
"""
import json
import logging
import re

import config
from language import detect_language, find_elements

log = logging.getLogger("story")

ENGLISH = {
    "person": "a child in a poncho", "condor": "a condor flying", "llama": "a llama", "puma": "a puma",
    "cat": "a cat", "dog": "a dog", "bird": "a bird", "fish": "a fish", "bear": "a spectacled bear",
    "fox": "a fox", "snake": "a snake", "mountain": "Andean mountains", "river": "a river",
    "lake": "a lake", "sea": "the sea", "sun": "the sun", "moon": "the moon", "star": "stars",
    "cloud": "clouds", "rain": "rain", "tree": "a tree", "flower": "flowers", "corn": "corn plants",
    "potato": "potatoes", "field": "farm fields", "house": "an adobe house", "bridge": "a rope bridge",
    "path": "a mountain path", "temple": "an Inca temple", "boat": "a reed boat", "fire": "a bonfire",
    "music": "a quena flute",
}
SPANISH = {
    "person": "una niña con poncho", "condor": "un cóndor", "llama": "una llama", "puma": "un puma",
    "cat": "un gato", "dog": "un perro", "bird": "un pájaro", "fish": "un pez", "bear": "un oso de anteojos",
    "fox": "un zorro", "snake": "una serpiente", "mountain": "las montañas", "river": "un río",
    "lake": "una laguna", "sea": "el mar", "sun": "el sol", "moon": "la luna", "star": "las estrellas",
    "cloud": "las nubes", "rain": "la lluvia", "tree": "un árbol", "flower": "flores", "corn": "maíz",
    "potato": "papas", "field": "la chacra", "house": "una casa de adobe", "bridge": "un puente",
    "path": "un camino", "temple": "un templo", "boat": "un bote de totora", "fire": "una fogata",
    "music": "una quena",
}

_JSON_INSTRUCTIONS = (
    "You analyse a short oral story told to a drawing robot in Peru. Reply with ONLY a JSON object with keys: "
    '"title" (short Spanish title), "elements" (list of keys from this set: %s), '
    '"scene" (one Spanish sentence describing the picture to draw), '
    '"image_prompt" (one English sentence describing a simple black-ink line drawing of the scene, no text), '
    '"narration" (2-3 warm Spanish sentences retelling the story).' % ", ".join(ENGLISH.keys())
)


# Concrete words that actually appear in the stories people tell the robot, so the offline
# analysis can draw what was said instead of a generic Andean postcard. Keys are accent-free
# and matched on word stems, values are what goes into the English drawing prompt.
LEXICON = {
    # cada palabra -> (para el dibujo en ingles, para mostrar en pantalla en espanol)
    # familia y gente
    "abuelo": ("an old man", "un abuelo"), "abuela": ("an old woman", "una abuela"),
    "abuelita": ("an old woman", "una abuelita"), "madre": ("a mother", "una madre"),
    "mama": ("a mother", "una mamá"), "padre": ("a father", "un padre"),
    "nino": ("a child", "un niño"), "nina": ("a girl", "una niña"),
    "hijo": ("a child", "un hijo"), "hija": ("a girl", "una hija"),
    "hermano": ("a brother", "un hermano"), "hermana": ("a sister", "una hermana"),
    "nieto": ("a grandchild", "un nieto"), "nieta": ("a grandchild", "una nieta"),
    "familia": ("a family", "una familia"), "amigo": ("a friend", "un amigo"),
    "hombre": ("a man", "un hombre"), "mujer": ("a woman", "una mujer"),
    "pastor": ("a shepherd", "un pastor"), "viajero": ("a traveller", "un viajero"),
    "tejedora": ("a weaver", "una tejedora"),
    # animales
    "llama": ("a llama", "una llama"), "alpaca": ("an alpaca", "una alpaca"),
    "vicuna": ("a vicuna", "una vicuña"), "oveja": ("sheep", "ovejas"),
    "condor": ("a condor", "un cóndor"), "zorro": ("a fox", "un zorro"),
    "puma": ("a puma", "un puma"), "oso": ("a spectacled bear", "un oso"),
    "perro": ("a dog", "un perro"), "gato": ("a cat", "un gato"),
    "pajaro": ("a bird", "un pájaro"), "ave": ("a bird", "un ave"),
    "pez": ("a fish", "un pez"), "serpiente": ("a snake", "una serpiente"),
    "culebra": ("a snake", "una culebra"), "vaca": ("a cow", "una vaca"),
    "toro": ("a bull", "un toro"), "caballo": ("a horse", "un caballo"),
    "burro": ("a donkey", "un burro"), "gallina": ("a hen", "una gallina"),
    "rana": ("a frog", "una rana"), "mariposa": ("a butterfly", "una mariposa"),
    "colibri": ("a hummingbird", "un colibrí"), "aguila": ("an eagle", "un águila"),
    "raton": ("a mouse", "un ratón"), "jaguar": ("a jaguar", "un jaguar"),
    "otorongo": ("a jaguar", "un otorongo"), "mono": ("a monkey", "un mono"),
    "tortuga": ("a turtle", "una tortuga"), "cuy": ("a guinea pig", "un cuy"),
    "chancho": ("a pig", "un chancho"), "pato": ("a duck", "un pato"),
    "abeja": ("bees", "abejas"), "arana": ("a spider", "una araña"),
    # paisaje
    "montana": ("Andean mountains", "las montañas"), "cerro": ("a hill", "un cerro"),
    "nevado": ("a snowy peak", "un nevado"), "cordillera": ("a mountain range", "la cordillera"),
    "rio": ("a river", "un río"), "laguna": ("a lagoon", "una laguna"),
    "lago": ("a lake", "un lago"), "mar": ("the sea", "el mar"),
    "playa": ("a beach", "una playa"), "valle": ("a valley", "un valle"),
    "selva": ("the jungle", "la selva"), "bosque": ("a forest", "un bosque"),
    "desierto": ("a desert", "el desierto"), "arbol": ("a tree", "un árbol"),
    "flor": ("flowers", "flores"), "piedra": ("stones", "piedras"),
    "cueva": ("a cave", "una cueva"), "camino": ("a path", "un camino"),
    "puente": ("a rope bridge", "un puente"), "chacra": ("farm fields", "la chacra"),
    "campo": ("fields", "el campo"), "pueblo": ("a village", "un pueblo"),
    "ciudad": ("a town", "una ciudad"), "sierra": ("the highlands", "la sierra"),
    "pampa": ("an open plain", "la pampa"),
    # cielo y clima
    "sol": ("the sun", "el sol"), "luna": ("the moon", "la luna"),
    "estrella": ("stars", "estrellas"), "nube": ("clouds", "nubes"),
    "lluvia": ("rain", "la lluvia"), "viento": ("wind", "el viento"),
    "nieve": ("snow", "la nieve"), "tormenta": ("a storm", "una tormenta"),
    "arcoiris": ("a rainbow", "un arcoíris"), "noche": ("night", "la noche"),
    "amanecer": ("sunrise", "el amanecer"), "atardecer": ("sunset", "el atardecer"),
    # cosas
    "casa": ("an adobe house", "una casa"), "choza": ("a hut", "una choza"),
    "puerta": ("a door", "una puerta"), "ventana": ("a window", "una ventana"),
    "olla": ("a clay pot", "una olla"), "pan": ("bread", "pan"),
    "manta": ("a woven blanket", "una manta"), "poncho": ("a poncho", "un poncho"),
    "sombrero": ("a hat", "un sombrero"), "canasta": ("a basket", "una canasta"),
    "telar": ("a loom", "un telar"), "quena": ("a flute", "una quena"),
    "tambor": ("a drum", "un tambor"), "charango": ("a charango", "un charango"),
    "bote": ("a reed boat", "un bote"), "barco": ("a boat", "un barco"),
    "fuego": ("a fire", "fuego"), "fogata": ("a bonfire", "una fogata"),
    "vela": ("a candle", "una vela"), "libro": ("a book", "un libro"),
    "maiz": ("corn plants", "maíz"), "quinua": ("quinoa", "quinua"),
    "trigo": ("wheat", "trigo"), "coca": ("coca leaves", "hojas de coca"),
    "semilla": ("seeds", "semillas"),
    # lo sagrado y la tierra (historias de la sierra)
    "apu": ("a sacred mountain", "el Apu"), "pachamama": ("the earth", "la Pachamama"),
    "tierra": ("the earth", "la tierra"), "ofrenda": ("an offering", "una ofrenda"),
    "pago": ("an offering", "un pago a la tierra"), "ancestro": ("ancestors", "los ancestros"),
    "antepasado": ("ancestors", "los antepasados"), "abuelos": ("grandparents", "los abuelos"),
    "altiplano": ("the high plateau", "el altiplano"), "altoandin": ("the high plateau", "lo altoandino"),
    "puna": ("the high plateau", "la puna"), "cosecha_": ("a harvest", "la cosecha"),
    "espiritu": ("a spirit", "un espíritu"), "inca": ("an Inca", "un inca"),
    "templo": ("an Inca temple", "un templo"), "iglesia": ("a church", "una iglesia"),
    "escuela": ("a school", "una escuela"), "mercado": ("a market", "un mercado"),
    # mas animales, gente, cosas y lugares que aparecen en las historias
    "leon": ("a lion", "un león"), "leona": ("a lioness", "una leona"), "tigre": ("a tiger", "un tigre"),
    "elefante": ("an elephant", "un elefante"), "jirafa": ("a giraffe", "una jirafa"), "mono": ("a monkey", "un mono"),
    "lobo": ("a wolf", "un lobo"), "conejo": ("a rabbit", "un conejo"), "raton": ("a mouse", "un ratón"),
    "delfin": ("a dolphin", "un delfín"), "ballena": ("a whale", "una ballena"), "tiburon": ("a shark", "un tiburón"),
    "pinguino": ("a penguin", "un pingüino"), "buho": ("an owl", "un búho"), "loro": ("a parrot", "un loro"),
    "paloma": ("a dove", "una paloma"), "gallo": ("a rooster", "un gallo"), "cerdo": ("a pig", "un cerdo"),
    "cabra": ("a goat", "una cabra"), "dinosaurio": ("a dinosaur", "un dinosaurio"), "dragon": ("a dragon", "un dragón"),
    "unicornio": ("a unicorn", "un unicornio"), "monstruo": ("a monster", "un monstruo"),
    "princesa": ("a princess", "una princesa"), "principe": ("a prince", "un príncipe"), "rey": ("a king", "un rey"),
    "reina": ("a queen", "una reina"), "bruja": ("a witch", "una bruja"), "mago": ("a wizard", "un mago"),
    "robot": ("a robot", "un robot"), "astronauta": ("an astronaut", "un astronauta"), "pirata": ("a pirate", "un pirata"),
    "soldado": ("a soldier", "un soldado"), "doctor": ("a doctor", "un doctor"), "maestra": ("a teacher", "una maestra"),
    "profesor": ("a teacher", "un profesor"), "bebe": ("a baby", "un bebé"), "chico": ("a boy", "un chico"),
    "chica": ("a girl", "una chica"), "joven": ("a young person", "un joven"), "anciano": ("an old man", "un anciano"),
    "anciana": ("an old woman", "una anciana"), "tio": ("an uncle", "un tío"), "tia": ("an aunt", "una tía"),
    "primo": ("a cousin", "un primo"), "vecino": ("a neighbour", "un vecino"), "gente": ("people", "gente"),
    "carro": ("a car", "un carro"), "auto": ("a car", "un auto"), "camion": ("a truck", "un camión"),
    "bus": ("a bus", "un bus"), "avion": ("an airplane", "un avión"), "tren": ("a train", "un tren"),
    "bicicleta": ("a bicycle", "una bicicleta"), "moto": ("a motorcycle", "una moto"), "cohete": ("a rocket", "un cohete"),
    "pelota": ("a ball", "una pelota"), "juguete": ("a toy", "un juguete"), "muneca": ("a doll", "una muñeca"),
    "guitarra": ("a guitar", "una guitarra"), "piano": ("a piano", "un piano"), "computadora": ("a computer", "una computadora"),
    "telefono": ("a phone", "un teléfono"), "reloj": ("a clock", "un reloj"), "llave": ("a key", "una llave"),
    "espada": ("a sword", "una espada"), "corona": ("a crown", "una corona"), "tesoro": ("a treasure chest", "un tesoro"),
    "torta": ("a cake", "una torta"), "pastel": ("a cake", "un pastel"), "helado": ("an ice cream", "un helado"),
    "manzana": ("an apple", "una manzana"), "platano": ("a banana", "un plátano"), "fruta": ("fruit", "fruta"),
    "comida": ("food", "comida"), "mesa": ("a table", "una mesa"), "silla": ("a chair", "una silla"),
    "cama": ("a bed", "una cama"), "cocina": ("a kitchen", "una cocina"), "jardin": ("a garden", "un jardín"),
    "parque": ("a park", "un parque"), "calle": ("a street", "una calle"), "colegio": ("a school", "un colegio"),
    "hospital": ("a hospital", "un hospital"), "tienda": ("a shop", "una tienda"), "castillo": ("a castle", "un castillo"),
    "isla": ("an island", "una isla"), "volcan": ("a volcano", "un volcán"), "cascada": ("a waterfall", "una cascada"),
    "planeta": ("a planet", "un planeta"), "cometa": ("a comet", "un cometa"), "nave": ("a spaceship", "una nave"),
    "futbol": ("a football", "fútbol"), "partido": ("a football match", "un partido"),
    "fiesta": ("a party", "una fiesta"), "cumpleanos": ("a birthday", "un cumpleaños"), "regalo": ("a gift", "un regalo"),
    "sonrisa": ("smiling", "sonriendo"), "abrazo": ("a hug", "un abrazo"), "beso": ("a kiss", "un beso"),
    "juega": ("playing", "jugando"), "jugaba": ("playing", "jugando"), "salta": ("jumping", "saltando"),
    "come": ("eating", "comiendo"), "comia": ("eating", "comiendo"), "lee": ("reading", "leyendo"),
    "escribe": ("writing", "escribiendo"), "dibuja": ("drawing", "dibujando"), "pinta": ("painting", "pintando"),
    "grita": ("shouting", "gritando"), "pelea": ("fighting", "peleando"), "abraza": ("hugging", "abrazando"),
    # acciones
    "tejer": ("weaving", "tejiendo"), "tejia": ("weaving", "tejiendo"),
    "camina": ("walking", "caminando"), "corr": ("running", "corriendo"),
    "vola": ("flying", "volando"), "nada": ("swimming", "nadando"),
    "duerme": ("sleeping", "durmiendo"), "dormia": ("sleeping", "durmiendo"),
    "canta": ("singing", "cantando"), "baila": ("dancing", "bailando"),
    "siembra": ("planting", "sembrando"), "cosechando": ("harvesting", "cosechando"),
    "entierra": ("kneeling on the ground", "arrodillada en la tierra"),
    "enterrar": ("kneeling on the ground", "arrodillada en la tierra"),
    "despierta": ("waking up", "despertando"), "despertaba": ("waking up", "despertando"),
    "saluda": ("waving", "saludando"), "reza": ("praying", "rezando"), "agradece": ("giving thanks", "agradeciendo"),
    "pasta": ("grazing", "pastando"), "cocina": ("cooking", "cocinando"),
    "llora": ("crying", "llorando"), "rie": ("laughing", "riendo"),
    "sube": ("climbing", "subiendo"), "baja": ("going down", "bajando"),
    "pesca": ("fishing", "pescando"), "monta": ("riding", "montando"),
}
# ñ -> n too: the lexicon keys are written without it, and \b[a-zn]+ used to split "montaña" into
# "monta" + "a", which then matched "monta" = riding instead of the mountain.
_ACENTOS = str.maketrans("áéíóúüñ", "aeiouun")


# What to draw, for the subjects people ask for most: a full description beats a bare noun.
# The model is told the pose and composition, so "un león" is a whole lion, not a face.
PROMPT_LIBRARY = {
    "a lion": "a friendly lion standing in profile, full body, big mane, simple outline",
    "a lioness": "a lioness walking in profile, full body, simple outline",
    "a llama": "a llama standing in profile, full body, fluffy, simple outline",
    "an alpaca": "an alpaca standing in profile, full body, fluffy, simple outline",
    "a condor": "an Andean condor with wings spread wide, seen from below, simple outline",
    "a fox": "a fox sitting in profile with a bushy tail, full body, simple outline",
    "a puma": "a puma walking in profile, full body, simple outline",
    "a spectacled bear": "a bear standing on four legs in profile, simple outline",
    "a dog": "a happy dog sitting in profile, full body, simple outline",
    "a cat": "a cat sitting in profile with its tail curled, simple outline",
    "a bird": "a small bird perched on a branch, in profile, simple outline",
    "a fish": "a fish swimming in profile with a few bubbles, simple outline",
    "a snake": "a snake curled in an S shape, simple outline",
    "a cow": "a cow standing in profile, full body, simple outline",
    "a horse": "a horse standing in profile, full body, simple outline",
    "a donkey": "a donkey standing in profile, full body, simple outline",
    "a hen": "a hen standing in profile, simple outline",
    "a frog": "a frog sitting on a lily pad, simple outline",
    "a butterfly": "a butterfly with open wings seen from above, symmetric, simple outline",
    "a hummingbird": "a hummingbird hovering next to a flower, simple outline",
    "an eagle": "an eagle with wings spread, simple outline",
    "a jaguar": "a jaguar walking in profile, full body, spotted, simple outline",
    "a monkey": "a monkey hanging from a branch by one arm, simple outline",
    "a turtle": "a turtle walking in profile, simple outline",
    "a guinea pig": "a guinea pig in profile, round and fluffy, simple outline",
    "a duck": "a duck swimming in profile, simple outline",
    "a tiger": "a tiger walking in profile, full body, striped, simple outline",
    "an elephant": "an elephant standing in profile, full body, simple outline",
    "a giraffe": "a giraffe standing in profile, full body, simple outline",
    "a wolf": "a wolf howling at the moon, in profile, simple outline",
    "a rabbit": "a rabbit sitting in profile with long ears, simple outline",
    "a dolphin": "a dolphin jumping over a wave, in profile, simple outline",
    "a whale": "a whale in profile with a water spout, simple outline",
    "a shark": "a shark swimming in profile, simple outline",
    "a penguin": "a penguin standing, front view, simple outline",
    "an owl": "an owl perched on a branch, front view, simple outline",
    "a parrot": "a parrot perched on a branch, in profile, simple outline",
    "a dinosaur": "a friendly dinosaur standing in profile, full body, simple outline",
    "a dragon": "a friendly dragon standing with small wings, full body, in profile, simple outline",
    "a unicorn": "a unicorn standing in profile, full body, simple outline",
    "a monster": "a friendly round monster with big eyes, front view, simple outline",
    "an old man": "an old man standing, full body, front view, poncho and hat, simple outline",
    "an old woman": "an old woman standing, full body, front view, long braids and hat, simple outline",
    "a child": "a child standing, full body, front view, simple outline",
    "a girl": "a girl standing, full body, front view, simple outline",
    "a boy": "a boy standing, full body, front view, simple outline",
    "a mother": "a mother standing with a child, full body, front view, simple outline",
    "a father": "a father standing with a child, full body, front view, simple outline",
    "a princess": "a princess standing in a long dress with a small crown, full body, simple outline",
    "a prince": "a prince standing with a small crown, full body, simple outline",
    "a king": "a king standing with a crown and cape, full body, simple outline",
    "a queen": "a queen standing with a crown, full body, simple outline",
    "a witch": "a witch with a pointed hat riding a broom, in profile, simple outline",
    "a wizard": "a wizard with a pointed hat and a staff, full body, simple outline",
    "a robot": "a friendly boxy robot standing, front view, simple outline",
    "an astronaut": "an astronaut floating in a spacesuit, full body, simple outline",
    "a pirate": "a pirate standing with a hat and a sword, full body, simple outline",
    "a shepherd": "a shepherd standing with a staff and a sheep, full body, simple outline",
    "a weaver": "a woman weaving on a backstrap loom, in profile, simple outline",
    "Andean mountains": "three mountain peaks with a sun, simple outline",
    "a river": "a winding river between two banks, simple outline",
    "a lake": "a lake with a small boat and hills behind, simple outline",
    "a tree": "a single tree with a round crown, simple outline",
    "flowers": "three flowers with leaves, simple outline",
    "the sun": "a sun with rays in the top corner, simple outline",
    "the moon": "a crescent moon with a few stars, simple outline",
    "stars": "a night sky with a few big stars, simple outline",
    "clouds": "two fluffy clouds, simple outline",
    "rain": "a cloud with rain drops falling, simple outline",
    "a rainbow": "a rainbow with clouds at each end, simple outline",
    "an adobe house": "a small house with a door, a window and a tiled roof, front view, simple outline",
    "a hut": "a small hut with a straw roof, front view, simple outline",
    "a castle": "a castle with two towers and a gate, front view, simple outline",
    "a school": "a school building with a flag, front view, simple outline",
    "a church": "a church with a bell tower, front view, simple outline",
    "a rope bridge": "a rope bridge between two cliffs, side view, simple outline",
    "a reed boat": "a reed boat on a lake, in profile, simple outline",
    "a boat": "a small boat on the water, in profile, simple outline",
    "a car": "a car in profile, simple outline",
    "a truck": "a truck in profile, simple outline",
    "a bus": "a bus in profile, simple outline",
    "an airplane": "an airplane flying, in profile, simple outline",
    "a train": "a train with two wagons, in profile, simple outline",
    "a bicycle": "a bicycle in profile, simple outline",
    "a rocket": "a rocket flying up with a flame, simple outline",
    "a spaceship": "a flying saucer, in profile, simple outline",
    "a ball": "a football, simple outline",
    "a guitar": "an acoustic guitar, front view, simple outline",
    "a flute": "a quena flute, simple outline",
    "a drum": "a drum with two sticks, simple outline",
    "a cake": "a birthday cake with candles, simple outline",
    "an apple": "an apple with a leaf, simple outline",
    "a gift": "a gift box with a bow, simple outline",
    "a fire": "a campfire with logs, simple outline",
    "a bonfire": "a campfire with logs, simple outline",
    "a treasure chest": "an open treasure chest, simple outline",
    "a crown": "a crown, front view, simple outline",
    "a sword": "a sword, simple outline",
    "a young person": "a young woman standing in a poncho and hat, full body, front view, simple outline",
    "a sacred mountain": "a tall snow-capped mountain with a gentle face in the peak, simple outline",
    "coca leaves": "three coca leaves, simple outline",
    "an offering": "a woven cloth on the ground with leaves and corn on it, simple outline",
    "ancestors": "an old man and an old woman in ponchos standing side by side, full body, simple outline",
    "the high plateau": "a wide open plain with mountains far away, simple outline",
    "a harvest": "a basket full of corn and potatoes, simple outline",
    "wind": "curved wind lines in the sky, simple outline",
    "a spirit": "a gentle glowing figure with a simple face, simple outline",
    "an Inca": "an Inca standing with a feathered headdress, full body, front view, simple outline",
}


_POSES = ("standing", "sitting", "walking", "perched", "swimming", "curled", "hanging", "howling",
          "hovering", "jumping", "floating", "riding a broom", "weaving", "wings spread", "round monster")


def _is_creature(w):
    """People and animals: their library description says how they pose."""
    d = PROMPT_LIBRARY.get(w, "")
    return any(p in d for p in _POSES)


def describe(words):
    """Turn the lexicon words into what the model should draw: the first noun that has a
    library entry becomes the main subject (with pose and composition), the rest come after."""
    if not words:
        return ""
    # a person or an animal is the main subject when there is one ("la joven ... el Apu" is a
    # drawing of the young woman, not of the mountain), otherwise the first known thing
    main = next((w for w in words if w in PROMPT_LIBRARY and _is_creature(w)), None) \
        or next((w for w in words if w in PROMPT_LIBRARY), None)
    if main is None:
        return ", ".join(words[:4])
    rest = [w for w in words if w != main][:3]     # main + three: with more the model draws a crowd
    actions = [w for w in rest if w.split()[0].endswith("ing")]      # walking, kneeling on the ground...
    things = [w for w in rest if not w.split()[0].endswith("ing")]
    out = PROMPT_LIBRARY[main]
    if actions:
        out += ", " + " and ".join(actions)
    if things:
        out += ", with " + ", ".join(things) + " in the background"
    return out


def _content(text, limit=4):
    """Words from the story itself, in the order they appear.
    Returns (para_el_dibujo_en_ingles, para_mostrar_en_espanol)."""
    plain = text.lower().translate(_ACENTOS)
    words = re.findall(r"[a-zn]+", plain)
    en, es, seen = [], [], set()
    for w in words:
        for key, (ingles, espanol) in LEXICON.items():
            k = key.rstrip("_")
            if (w == k or w == k + "s" or w == k + "es" or (len(k) > 4 and w.startswith(k))) and ingles not in seen:
                seen.add(ingles)
                en.append(ingles)
                es.append(espanol)
                break
        if len(en) >= limit:
            break
    return en, es


def _title_from(text):
    words = re.findall(r"[\wñáéíóúü']+", text)
    t = " ".join(words[:6]).strip()
    return (t[0].upper() + t[1:]) if t else "Una historia"


def _join_es(names):
    if not names:
        return "un paisaje andino"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " y " + names[-1]


def _join_en(names):
    if not names:
        return "an Andean landscape"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _rules(text, lang):
    """Offline analysis. Builds the picture from what the story actually says; only falls back
    to a generic Andean scene when no known word appears at all."""
    found = find_elements(text)
    words, palabras = _content(text, limit=6)
    elements = found or ["mountain"]
    es = [SPANISH.get(e, e) for e in found]

    if words:
        # the main subject gets its library description (pose, composition), the rest are added;
        # describe() keeps at most four things: with seven the model draws a crowd.
        image_prompt = describe(words)
        # what the screen shows: the same things that went into the drawing, in Spanish
        scene = _join_es(palabras).capitalize() + "."
    else:
        # nothing in the vocabulary: hand the model the story's own words rather than a made-up
        # default scene ("un león" used to become "a child in a poncho under the sun")
        crudo = " ".join(re.findall(r"[\wáéíóúüñ]+", text.lower()))
        image_prompt = crudo[:80] if crudo else "an Andean landscape"
        scene = "Lo que contaste: " + (text.strip()[:70] or "un paisaje andino") + "."

    short = text.strip()
    if len(short) > 220:
        short = short[:217].rsplit(" ", 1)[0] + "..."
    # Without a language model the honest retelling is the story itself: adding "una historia de
    # X" invented characters (a story about a grandmother came back as "una niña con poncho").
    narration = f"Contada en quechua. {short}" if lang == "quechua" else short
    return {"title": _title_from(text), "elements": elements, "scene": scene,
            "image_prompt": image_prompt, "narration": narration}


def _validate(d, text, lang):
    base = _rules(text, lang)
    if not isinstance(d, dict):
        raise ValueError("not a JSON object")
    out = dict(base)
    for k in ("title", "scene", "image_prompt", "narration"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            out[k] = v.strip()
    els = d.get("elements")
    if isinstance(els, list):
        els = [e for e in els if isinstance(e, str) and e in ENGLISH]
        if els:
            out["elements"] = els
    return out


def _remote(text, lang):
    import requests
    import laptop
    u = laptop.url(8600)
    if not u:
        raise RuntimeError("ai_server no está corriendo en la laptop")
    r = requests.post(f"{u}/analyze", json={"text": text, "lang": lang},
                      timeout=(3, config.AI_SERVER_TIMEOUT))
    r.raise_for_status()
    return _validate(r.json(), text, lang)


def _extract_json(s):
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    return json.loads(m.group(0))


def _ollama(text, lang):
    import requests
    import laptop
    u = laptop.url(11434)
    if not u:
        raise RuntimeError("Ollama no está corriendo en la laptop")
    prompt = f"{_JSON_INSTRUCTIONS}\n\nStory ({lang}):\n{text}"
    r = requests.post(f"{u}/api/generate",
                      json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
                      timeout=(3, config.AI_SERVER_TIMEOUT))
    r.raise_for_status()
    return _validate(_extract_json(r.json().get("response", "")), text, lang)


def _anthropic(text, lang):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(model=config.ANTHROPIC_MODEL, max_tokens=600,
                                 system=_JSON_INSTRUCTIONS,
                                 messages=[{"role": "user", "content": f"Story ({lang}):\n{text}"}])
    return _validate(_extract_json("".join(b.text for b in msg.content if hasattr(b, "text"))), text, lang)


BACKENDS = {"remote": _remote, "ollama": _ollama, "anthropic": _anthropic, "rules": _rules}


def analyze(text, lang=None, backends=None, on_status=None):
    text = (text or "").strip()
    if lang is None:
        lang, _ = detect_language(text)
    errors = []
    for name in (backends or config.STORY_BACKENDS):
        fn = BACKENDS.get(name)
        if fn is None:
            continue
        try:
            if on_status:
                on_status(name)
            out = fn(text, lang)
            out.update({"backend": name, "lang": lang, "text": text, "errors": errors})
            return out
        except Exception as e:
            log.warning("story backend '%s' failed: %s", name, e)
            errors.append(f"{name}: {e}")
    out = _rules(text, lang)
    out.update({"backend": "rules", "lang": lang, "text": text, "errors": errors})
    return out
