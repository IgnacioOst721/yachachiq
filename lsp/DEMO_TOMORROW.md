# ASL Demo — Tomorrow Checklist

## ✅ Already done (this Mac = "Mac A", the camera one)
- Camera + hand tracking working
- Model trained: ~96% honest accuracy. Knows 26 letters + 👍 space + 👎 backspace
- App auto-finds the other Mac on the network and types onto it
- Everything tested

---

## 🖥️ Mac B one-time setup  (DO THIS TODAY if Mac B is handy)
"Mac B" = the Mac that will be typed into.

1. **AirDrop** `receiver.py` from this Mac (in the `asl-camera` folder) to Mac B.
2. On Mac B, install the typing library:
   ```
   pip3 install --user pyautogui
   ```
3. Give Python permission to type:
   **System Settings → Privacy & Security → Accessibility → turn ON Terminal.**
   (If Terminal isn't listed, just run `python3 receiver.py` once and macOS will ask — click Allow.)

---

## ▶️ TOMORROW — the whole demo is 2 commands

1. Put **BOTH Macs on the SAME WiFi** (or a phone hotspot).

2. On **Mac B**, in Terminal:
   ```
   python3 receiver.py
   ```
   - It prints **RECEIVER READY** and Mac B's IP.
   - **Click into a TextEdit window** so the letters land there.

3. On **Mac A** (this one), in Terminal:
   ```
   cd ~/asl-camera
   python3 asl_app.py
   ```
   - It **auto-finds Mac B**. The window shows **"→ Mac B"** when connected.
   - If it says *"No other Mac found"*, run this instead (IP is shown on Mac B):
     ```
     python3 asl_app.py <Mac B IP>
     ```

4. **Sign!** Letters appear in TextEdit on Mac B. 🎉

---

## 🎮 Controls while signing
- **Letters A–Z**: sign and HOLD steady until the green bar fills → it types.
- **Same letter twice** (e.g. the LL in HELLO): drop your hand, then sign again.
- 👍 **thumbs-up** = space
- 👎 **thumbs-down** = backspace
- **I vs J**: hold still = I, swoop the pinky = J
- **Q** (with the window clicked) = quit

---

## 🧯 If something's off
- **"No other Mac found"** → both on the same WiFi? Use the manual IP: `python3 asl_app.py <Mac B IP>`
- **Nothing types on Mac B** → did you click into TextEdit? Is **Accessibility ON** for Terminal on Mac B?
- **macOS asks "allow incoming connections?"** on Mac B → click **Allow**.
- **A letter misreads** → sign it more deliberately. M and R are the weakest letters.
- **Camera window won't open** → Privacy & Security → Camera → Terminal ON, then Cmd+Q Terminal and reopen.

---

## 📁 Files in this folder
- `asl_app.py` ...... the main app (run this on Mac A)
- `receiver.py` ..... runs on Mac B, types what it receives
- `model.pkl` ....... the trained "decoding" model
- `collect_data.py` . record more sign samples (if you ever want to improve a letter)
- `train_model.py` .. rebuild model.pkl after collecting more
- `asl_data.csv` .... all your collected sign data
