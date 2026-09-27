"""Voice self-check for Ignacio: what works right now on this machine?

    .venvs/voice/bin/python -m yq.voice.check          # add --say to hear the speaker
"""
from __future__ import annotations

import sys


def _ok(flag: bool, text: str) -> bool:
    print(("  OK   " if flag else "  FALTA ") + text)
    return flag


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--say", action="store_true", help="speak a test sentence")
    a = ap.parse_args(argv)
    from yq.common import config, languages
    from . import asr, audio as au, settings, tts, vad
    print("Voz de Yachachiq - revisión")
    c = languages.meta().get("counts", {})
    _ok(bool(c), "lista de idiomas: %s reconocer, %s hablar, %s traducir" % (c.get("asr_any"), c.get("tts_any"),
                                                                           c.get("translate")))
    devs = au.list_devices()
    mic = au.find_device("input", settings.MIC_DEVICE, settings.MIC_NAMES)
    spk = au.find_device("output", settings.SPEAKER_DEVICE, settings.SPEAKER_NAMES)
    names = {d["index"]: d["name"] for d in devs}
    _ok(any(d["inputs"] for d in devs), "micrófono: %s" % (names.get(mic) or "el de por defecto (KAYSUDA no encontrado)"))
    _ok(any(d["outputs"] for d in devs), "parlante: %s" % (names.get(spk) or "el de por defecto"))
    v = vad.make_vad()
    _ok(isinstance(v, vad.SileroVAD), "detector de voz Silero (%s)" % settings.vad_model_path())
    voices = [l for l in ("spa_Latn", "eng_Latn", "por_Latn") if tts.local_voice(l)]
    _ok(bool(voices), "voces Piper instaladas: %s (%s)" % (", ".join(voices) or "ninguna", settings.piper_dir()))
    _ok(asr.cuda_available(), "GPU CUDA para Whisper local (si falta, se usa la CPU)")
    try:
        import faster_whisper  # noqa: F401
        fw = True
    except ImportError:
        fw = False
    _ok(fw, "faster-whisper instalado (respaldo local de reconocimiento)")
    mac = False
    if not config.mock("mac"):
        try:
            from yq.common.macclient import client
            h = client().health()
            mac = h.get("domains", {}).get("routes_voice") == "ok"
            _ok(mac, "Mac con voz lista (%s)" % client()._good)
        except Exception as e:
            _ok(False, "Mac no responde (%s): quechua/aimara y traducción no funcionarán" % e)
    if a.say:
        print("  hablando...", tts.say("Hola, soy Yachachiq. Cuéntame tu historia.", "spa_Latn", wait=True),
              tts.last_engine)
    return 0


if __name__ == "__main__":
    sys.exit(main())
